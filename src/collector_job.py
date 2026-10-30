import asyncio
from datetime import datetime, time, timedelta
import logging
import os
from zoneinfo import ZoneInfo

from src.backend_client import post_competition
from src.services.collector import collect_contest_korea, collect_uos

SEOUL_TIMEZONE = ZoneInfo("Asia/Seoul")
logger = logging.getLogger(__name__)


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
