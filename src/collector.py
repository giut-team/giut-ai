import asyncio
import logging
import os
from datetime import date, datetime, time, timedelta
from urllib.parse import parse_qs, urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from src.services.extraction import is_eligible_activity
from src.llm.field_extractor import extract_fields_from_text, extract_fields_from_images
from src.llm.prompts import FIELDS_DESCRIPTION
from src.parsers.uos_rss_parser import parse_uos_rss
from src.parsers.contest_korea_parser import parse_contest_korea
from src.backend_client import post_competition

UOS_RSS_GENERAL = "https://www.uos.ac.kr/rss/gBoard.do"
UOS_RSS_HAKSA = "https://www.uos.ac.kr/rss/hBoard.do"
PAGE_EXTRACTION_CONCURRENCY = 5

CONTEST_KOREA_CONTEST_LIST = "https://www.contestkorea.com/sub/list.php?int_gbn=1"
CONTEST_KOREA_ACTIVITY_LIST = "https://www.contestkorea.com/sub/list.php?int_gbn=2"
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


async def _extract_uos_page(page: dict, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        page_text = page["title"] + "\n" + page["body"]
        extracted_fields, usage = await extract_fields_from_text(page_text)
        extracted_fields = extracted_fields.to_dict()

        if not is_eligible_activity(extracted_fields):
            return None

        missing_keys = [key for key, value in extracted_fields.items() if value is None]
        if missing_keys:
            image_extracted_fields, usage = await extract_fields_from_images(
                page["images"], missing_keys
            )
            for key, value in image_extracted_fields.to_dict().items():
                if value is not None:
                    extracted_fields[key] = value

        return extracted_fields


async def collect_uos() -> list[dict]:
    # 자정 기준으로 전일 게시된 공지만 수집

    general, haksa = await asyncio.gather(
        asyncio.to_thread(parse_uos_rss, UOS_RSS_GENERAL),
        asyncio.to_thread(parse_uos_rss, UOS_RSS_HAKSA),
    )
    semaphore = asyncio.Semaphore(PAGE_EXTRACTION_CONCURRENCY)

    extracted_pages = await asyncio.gather(
        *(_extract_uos_page(page, semaphore) for page in general + haksa)
    )
    return [page for page in extracted_pages if page is not None]


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


async def _extract_contest_korea_page(page: dict, semaphore: asyncio.Semaphore):
    async with semaphore:
        missing_keys = [
            key for key in FIELDS_DESCRIPTION.keys() if page.get(key, None) is None
        ]
        if missing_keys:
            extracted_fields, usage = await extract_fields_from_text(
                page["full_text"], missing_keys
            )
            extracted_fields = extracted_fields.to_dict()

            if not is_eligible_activity(extracted_fields):
                return None

            missing_keys = [
                key for key, value in extracted_fields.items() if value is None
            ]
            if missing_keys:
                image_extracted_fields, usage = await extract_fields_from_images(
                    [page["main_image_link"], *page["detail_image_links"]], missing_keys
                )
                for key, value in image_extracted_fields.to_dict().items():
                    if value is not None:
                        extracted_fields[key] = value

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

    extracted_pages = await asyncio.gather(
        *(_extract_contest_korea_page(page, semaphore) for page in results)
    )
    return [page for page in extracted_pages if page is not None]


async def collect_and_post(bearer_token: str) -> None:
    # 시립대 공지 및 콘테스트코리아에서 추출한 결과 리스트 전체 전송
    uos_results, contest_korea_results = await asyncio.gather(
        collect_uos(), collect_contest_korea()
    )
    payload = uos_results + contest_korea_results
    await asyncio.to_thread(post_competition, payload, bearer_token)


async def run_daily_collector(bearer_token: str) -> None:
    # 매일 자정에 공모전 정보 수집 후 전송
    if not bearer_token or not bearer_token.strip():
        raise ValueError("bearer_token must be provided")

    while True:
        now = datetime.now(SEOUL_TIMEZONE)
        next_midnight = datetime.combine(
            now.date() + timedelta(days=1), time.min, tzinfo=SEOUL_TIMEZONE
        )
        await asyncio.sleep((next_midnight - now).total_seconds())

        try:
            await collect_and_post(bearer_token)
        except Exception:
            logger.exception("공모전 수집 또는 전송 실패")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(run_daily_collector(os.getenv("GIUT_BEARER_TOKEN", "")))
