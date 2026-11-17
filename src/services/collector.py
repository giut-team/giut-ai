import asyncio
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
import re
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.llm.dto import CategoryEnum
from src.config import settings
from src.services.constants import UrlType
from src.services.extraction import (
    InvalidContestError,
    check_valid_activity_after_text,
    check_valid_activity_after_image,
    filter_official_url,
)
from src.llm.field_extractor import extract_fields_from_text, extract_fields_from_images
from src.llm.prompts import FIELDS_DESCRIPTION
from src.parsers.uos_rss_parser import parse_uos_rss
from src.parsers.contest_korea_parser import parse_contest_korea

UOS_RSS_GENERAL = "https://www.uos.ac.kr/rss/gBoard.do"
UOS_RSS_HAKSA = "https://www.uos.ac.kr/rss/hBoard.do"

CONTEST_KOREA_CONTEST_LIST = (
    "https://www.contestkorea.com/sub/list.php?int_gbn=1"  # 공모전/대회
)
CONTEST_KOREA_ACTIVITY_LIST = (
    "https://www.contestkorea.com/sub/list.php?int_gbn=2"  # 대외활동
)
CONTEST_KOREA_SCAN_LIMIT = 30  # 각 목록 앞 30개에서 새 게시물만 선택
CONTEST_KOREA_FILTERS = {  # 공모전 검색 조건
    "Txt_sortkey": "a.int_sort",
    "Txt_sortword": "desc",
    "Txt_code1[0]": "26",  # 누구나
    "Txt_code1[1]": "30",  # 대학생
    "Txt_code1[2]": "76",  # 대학원생
    "Txt_area[0]": "75",  # 전국
    "Txt_area[1]": "97",  # 온라인
    "Txt_area[2]": "31",  # 서울
}

CONTEST_KOREA_LISTS = {
    "contest": CONTEST_KOREA_CONTEST_LIST,
    "activity": CONTEST_KOREA_ACTIVITY_LIST,
}

PAGE_EXTRACTION_CONCURRENCY = 5

logger = logging.getLogger(__name__)


def _published_in_range(page: dict, start: datetime, end: datetime) -> bool:
    """게시 시각이 [start, end) 범위에 속하는지 확인"""
    published_date = page.get("published_date", "")
    try:
        try:
            published_at = parsedate_to_datetime(published_date)
        except ValueError:
            published_at = datetime.fromisoformat(published_date)
    except (TypeError, ValueError):
        logger.warning("시립대 RSS 게시일을 읽을 수 없음: %r", published_date)
        return False

    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=settings.TIMEZONE)
    return start <= published_at < end


def _uos_link_context_blocks(full_text: str) -> list[dict]:
    blocks = []
    for line in full_text.split("\n"):
        urls_iter = {
            *re.finditer(
                r"(https?:\/\/)(?:[\w-]+\.)+[\w-]+(?::\d+)?(?:[\/?#][^\s]*)?", line
            ),
        }
        next_idx = 0
        for i in sorted(urls_iter, key=lambda i: i.start()):
            start, end = i.span()
            blocks.append(
                {
                    "url": line[start:end].strip(),
                    "text": line[next_idx:start].strip(),
                    "line": line,
                    "context": "",
                }
            )
            next_idx = end
    return blocks


async def extract_uos_page(page: dict) -> dict:
    page_text = page["title"] + "\n" + page["body"]
    # 제목을 RSS피드에서 이미 제공하므로 추출할 키에서 제외
    keys = [*FIELDS_DESCRIPTION][:]
    keys.remove("title")

    extracted_fields = await extract_fields_from_text(
        page_text, page["source_url"], keys
    )
    extracted_fields["title"] = page["title"]
    extracted_fields["images"] = page["images"]
    extracted_fields["urls"] = [
        {"type": UrlType.SOURCE_ORIGINAL, "url": page["source_url"]},
        {
            "type": UrlType.OFFICIAL,
            "url": filter_official_url(
                _uos_link_context_blocks(page_text), page["source_url"]
            ),
        },
    ]

    check_valid_activity_after_text(extracted_fields)

    missing_keys = [key for key, value in extracted_fields.items() if value is None]
    if missing_keys:
        image_extracted_fields = await extract_fields_from_images(
            page["images"], page["source_url"], missing_keys
        )
        for key, value in image_extracted_fields.items():
            if value is not None:
                extracted_fields[key] = value

    check_valid_activity_after_image(extracted_fields)

    if not extracted_fields["category"]:
        # 카테고리 분류에 실패했을 경우 '기타'로 분류
        extracted_fields["category"] = [CategoryEnum.ETC]

    return extracted_fields


async def collect_uos(start: datetime, end: datetime) -> list[dict]:
    """[start, end) 사이에 게시된 공지만 수집"""
    general, haksa = await asyncio.gather(
        asyncio.to_thread(parse_uos_rss, UOS_RSS_GENERAL),
        asyncio.to_thread(parse_uos_rss, UOS_RSS_HAKSA),
    )
    semaphore = asyncio.Semaphore(PAGE_EXTRACTION_CONCURRENCY)

    async def extract_limited(page: dict) -> dict | None:
        async with semaphore:
            try:
                return await extract_uos_page(page)
            except InvalidContestError as exc:
                # 등록 기준에 맞지 않는 활동은 정상 필터링이므로 traceback 없이 기록
                logger.info(
                    "시립대 공지 필터링됨: %s (%s)", page.get("source_url"), str(exc)
                )
                return None
            except Exception:
                # 한 페이지 실패가 전체 수집을 중단시키지 않도록 로그만 남기고 제외
                logger.exception("시립대 공지 추출 실패: %s", page.get("source_url"))
                return None

    results = await asyncio.gather(
        *(
            extract_limited(page)
            for page in general + haksa
            if _published_in_range(page, start, end)
        )
    )
    return [result for result in results if result is not None]


def contest_korea_str_no(url: str) -> int | None:
    """상세 URL의 str_no(YYYYMMDD + 일련번호)를 정수로 반환"""
    str_no = parse_qs(urlparse(url).query).get("str_no", [""])[0]
    return int(str_no) if str_no.isdigit() else None


async def _contest_korea_detail_urls(list_url: str, last_str_no: int) -> list[str]:
    """목록에서 str_no가 last_str_no보다 큰 게시물의 상세 URL 반환"""
    page_url = list_url.split("?", maxsplit=1)[0]
    int_gbn = parse_qs(urlparse(list_url).query)["int_gbn"][0]
    first_page = True
    detail_urls = []
    seen_urls = set()
    scanned_entries = 0

    while page_url and scanned_entries < CONTEST_KOREA_SCAN_LIMIT:
        response = await asyncio.to_thread(
            requests.get,
            page_url,
            params=(
                {"int_gbn": int_gbn, **CONTEST_KOREA_FILTERS} if first_page else None
            ),
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        first_page = False
        soup = BeautifulSoup(response.content, "html.parser")
        entries = soup.select(".list_style_2 .title a[href*='str_no=']")
        if not entries:
            break

        # 사이트의 목록 순서는 게시일 순서가 아니므로 오래된 글이 나와도 CONTEST_KOREA_SCAN_LIMIT 범위 내에서 계속 확인
        candidates = entries[: CONTEST_KOREA_SCAN_LIMIT - scanned_entries]
        scanned_entries += len(candidates)
        for entry in candidates:
            url = urljoin(response.url, entry["href"])
            str_no = contest_korea_str_no(url)
            if str_no is None:
                logger.warning("콘테스트코리아 게시물 번호를 읽을 수 없음: %s", url)
                continue

            if str_no > last_str_no and url not in seen_urls:
                seen_urls.add(url)
                detail_urls.append(url)

        if scanned_entries >= CONTEST_KOREA_SCAN_LIMIT:
            break

        current_page = int(parse_qs(urlparse(response.url).query).get("page", ["1"])[0])
        next_page = None
        for link in soup.select(".pagination a[href]"):
            candidate = urljoin(response.url, link["href"])
            if parse_qs(urlparse(candidate).query).get("page") == [
                str(current_page + 1)
            ]:
                next_page = candidate
                break
        page_url = next_page

    return detail_urls


async def extract_contest_korea_page(page: dict) -> dict:
    extracted_fields = {key: page.get(key, None) for key in FIELDS_DESCRIPTION.keys()}
    extracted_fields["urls"] = page["urls"]

    missing_keys = [
        key for key in FIELDS_DESCRIPTION.keys() if page.get(key, None) is None
    ]
    extracted_missing_fields = await extract_fields_from_text(
        page["full_text"], page["source_url"], missing_keys
    )
    for key, value in extracted_missing_fields.items():
        extracted_fields[key] = value

    check_valid_activity_after_text(extracted_fields)

    missing_keys = [key for key, value in extracted_fields.items() if value is None]
    if missing_keys:
        image_extracted_fields = await extract_fields_from_images(
            [page["main_image_link"], *page["detail_image_links"]],
            page["source_url"],
            missing_keys,
        )
        for key, value in image_extracted_fields.items():
            if value is not None:
                extracted_fields[key] = value

    check_valid_activity_after_image(extracted_fields)

    if not extracted_fields["category"]:
        # 카테고리 분류에 실패했을 경우 '기타'로 분류
        extracted_fields["category"] = [CategoryEnum.ETC]

    return extracted_fields


async def collect_contest_korea(
    last_str_nos: dict[str, int],
) -> tuple[list[dict], dict[str, int]]:
    """목록별로 마지막으로 수집한 str_no보다 큰 게시물만 수집

    반환값: (추출 결과, 목록별 갱신된 마지막 str_no)
    """
    names = list(CONTEST_KOREA_LISTS)
    url_lists = await asyncio.gather(
        *(
            _contest_korea_detail_urls(CONTEST_KOREA_LISTS[name], last_str_nos[name])
            for name in names
        )
    )
    new_last_str_nos = {
        name: max(
            [last_str_nos[name], *(contest_korea_str_no(url) for url in urls)]
        )
        for name, urls in zip(names, url_lists)
    }
    detail_urls = list(dict.fromkeys(url for urls in url_lists for url in urls))

    results = []
    for url in detail_urls:
        try:
            page = await asyncio.to_thread(parse_contest_korea, url)
        except Exception:
            logger.exception("콘테스트코리아 상세 페이지 파싱 실패: %s", url)
            continue
        page["source_url"] = url
        results.append(page)

    semaphore = asyncio.Semaphore(PAGE_EXTRACTION_CONCURRENCY)

    async def extract_limited(page: dict) -> dict:
        async with semaphore:
            try:
                return await extract_contest_korea_page(page)
            except InvalidContestError as exc:
                # 등록 기준에 맞지 않는 활동은 정상 필터링이므로 traceback 없이 기록
                logger.info(
                    "콘테스트코리아 필터링됨: %s (%s)", page.get("source_url"), str(exc)
                )
                return None
            except Exception:
                # 한 페이지 실패가 전체 수집을 중단시키지 않도록 로그만 남기고 제외
                logger.exception("콘테스트코리아 추출 실패: %s", page.get("source_url"))
                return None

    extracted = await asyncio.gather(*(extract_limited(page) for page in results))
    return [result for result in extracted if result is not None], new_last_str_nos
