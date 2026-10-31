import asyncio
import logging
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
import re
from urllib.parse import parse_qs, urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from src.llm.field_extractor import extract_fields_from_text, extract_fields_from_images
from src.llm.prompts import FIELDS_DESCRIPTION
from src.parsers.uos_rss_parser import parse_uos_rss
from src.parsers.contest_korea_parser import parse_contest_korea

UOS_RSS_GENERAL = "https://www.uos.ac.kr/rss/gBoard.do"
UOS_RSS_HAKSA = "https://www.uos.ac.kr/rss/hBoard.do"
PAGE_EXTRACTION_CONCURRENCY = 5

CONTEST_KOREA_CONTEST_LIST = (
    "https://www.contestkorea.com/sub/list.php?int_gbn=1"  # 공모전/대회
)
CONTEST_KOREA_ACTIVITY_LIST = (
    "https://www.contestkorea.com/sub/list.php?int_gbn=2"  # 대외활동
)
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

SEOUL_TIMEZONE = ZoneInfo("Asia/Seoul")
logger = logging.getLogger(__name__)


def _published_on_date(page: dict, target_date: date) -> bool:
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
        published_at = published_at.replace(tzinfo=SEOUL_TIMEZONE)
    return published_at.astimezone(SEOUL_TIMEZONE).date() == target_date


async def extract_uos_page(page: dict) -> dict:
    page_text = page["title"] + "\n" + page["body"]
    extracted_fields, usage = await extract_fields_from_text(page_text)
    extracted_fields = extracted_fields.to_dict()

    missing_keys = [key for key, value in extracted_fields.items() if value is None]
    if missing_keys:
        image_extracted_fields, usage = await extract_fields_from_images(
            page["images"], missing_keys
        )
        for key, value in image_extracted_fields.to_dict().items():
            if value is not None:
                extracted_fields[key] = value

    extracted_fields["links"] = {
        page["link"],
        *re.findall(r"https://[^\s]+", page_text),
    }
    return extracted_fields


async def collect_uos() -> list[dict]:
    # 자정 기준으로 전일 게시된 공지만 수집
    yesterday = datetime.now(SEOUL_TIMEZONE).date() - timedelta(days=1)
    general, haksa = await asyncio.gather(
        asyncio.to_thread(parse_uos_rss, UOS_RSS_GENERAL),
        asyncio.to_thread(parse_uos_rss, UOS_RSS_HAKSA),
    )
    semaphore = asyncio.Semaphore(PAGE_EXTRACTION_CONCURRENCY)

    async def extract_limited(page: dict):
        async with semaphore:
            return await extract_uos_page(page)

    return await asyncio.gather(
        *(
            extract_limited(page)
            for page in general + haksa
            if _published_on_date(page, yesterday)
        )
    )


async def _contest_korea_detail_urls(list_url: str, yesterday: date) -> list[str]:
    page_url = list_url.split("?", maxsplit=1)[0]
    int_gbn = parse_qs(urlparse(list_url).query)["int_gbn"][0]
    first_page = True
    detail_urls = []
    seen_urls = set()

    while page_url:
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

        reached_older_entry = False
        for entry in entries:
            url = urljoin(response.url, entry["href"])
            str_no = parse_qs(urlparse(url).query).get("str_no", [""])[0]
            try:
                posted_date = datetime.strptime(str_no[:8], "%Y%m%d").date()
            except ValueError:
                logger.warning(
                    "콘테스트코리아 게시물 번호에서 날짜를 읽을 수 없음: %s", url
                )
                continue

            if posted_date < yesterday:
                reached_older_entry = True
                break
            if posted_date == yesterday and url not in seen_urls:
                seen_urls.add(url)
                detail_urls.append(url)

        if reached_older_entry:
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


async def extract_contest_korea_page(page: dict):
    extracted_fields = {key: page.get(key, None) for key in FIELDS_DESCRIPTION.keys()}
    missing_keys = [
        key for key in FIELDS_DESCRIPTION.keys() if page.get(key, None) is None
    ]
    extracted_missing_fields, usage = await extract_fields_from_text(
        page["full_text"], missing_keys
    )
    extracted_missing_fields = extracted_missing_fields.to_dict()
    for key, value in extracted_missing_fields.items():
        extracted_fields[key] = value

    missing_keys = [key for key, value in extracted_fields.items() if value is None]
    if missing_keys:
        image_extracted_fields, usage = await extract_fields_from_images(
            [page["main_image_link"], *page["detail_image_links"]], missing_keys
        )
        for key, value in image_extracted_fields.to_dict().items():
            if value is not None:
                extracted_fields[key] = value

    extracted_fields["links"] = {
        page["homepage_link"],
        page["application_link"],
    }.remove(None)

    return extracted_fields


async def collect_contest_korea() -> list[dict]:
    # 자정 기준으로 전일 게시된 공지만 수집
    yesterday = (datetime.now(SEOUL_TIMEZONE) - timedelta(days=1)).date()
    competition_urls, activity_urls = await asyncio.gather(
        _contest_korea_detail_urls(CONTEST_KOREA_CONTEST_LIST, yesterday),
        _contest_korea_detail_urls(CONTEST_KOREA_ACTIVITY_LIST, yesterday),
    )
    detail_urls = list(dict.fromkeys(competition_urls + activity_urls))

    results = []
    for url in detail_urls:
        try:
            page = await asyncio.to_thread(parse_contest_korea, url)
        except Exception:
            logger.exception("콘테스트코리아 상세 페이지 파싱 실패: %s", url)
            continue
        page["url"] = url
        results.append(page)

    semaphore = asyncio.Semaphore(PAGE_EXTRACTION_CONCURRENCY)

    async def extract_limited(page: dict):
        async with semaphore:
            return await extract_contest_korea_page(page)

    return await asyncio.gather(*(extract_limited(page) for page in results))
