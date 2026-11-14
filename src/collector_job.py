import asyncio
from datetime import datetime, time, timedelta
import logging

from src.config import settings
from src.backend_client import post_competition
from src.services.collector import collect_contest_korea, collect_uos

logger = logging.getLogger(__name__)


async def collect_and_post(x_ai_api_key: str) -> None:
    """시립대 공지 및 콘테스트코리아에서 추출한 결과 리스트 전체 전송"""
    uos_results, contest_korea_results = await asyncio.gather(
        collect_uos(), collect_contest_korea()
    )
    payload = [result for result in uos_results + contest_korea_results]
    await asyncio.to_thread(post_competition, payload, x_ai_api_key)


async def run_daily_collector(x_ai_api_key: str) -> None:
    """매일 자정에 시립대 공지 및 콘테스트코리아에서 공모전 정보 수집 후 전송"""
    if not x_ai_api_key or not x_ai_api_key.strip():
        raise ValueError("X-AI-API-Key must be provided")

    while True:
        now = datetime.now(settings.TIMEZONE)
        next_midnight = datetime.combine(
            now.date() + timedelta(days=1), time.min, tzinfo=settings.TIMEZONE
        )
        await asyncio.sleep((next_midnight - now).total_seconds())

        try:
            logger.info("공모전 정보 수집 시작")
            await collect_and_post(x_ai_api_key)
            logger.info("공모전 수집 및 전송 성공")
        except Exception as exc:
            logger.exception("공모전 수집 또는 전송 실패: %r", exc)


if __name__ == "__main__":
    asyncio.run(run_daily_collector(settings.X_AI_API_KEY))
