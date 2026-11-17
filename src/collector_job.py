import asyncio
from datetime import datetime, time, timedelta
import json
import logging
from pathlib import Path

from src.config import settings
from src.backend_client import post_competition
from src.services.collector import (
    CONTEST_KOREA_LISTS,
    collect_contest_korea,
    collect_uos,
)

logger = logging.getLogger(__name__)

COLLECT_HOURS = (0, 12)  # 서울 기준 00시, 12시에 수집
COLLECT_INTERVAL = timedelta(hours=12)


def _next_run_at(now: datetime) -> datetime:
    """now 이후 가장 가까운 수집 시각"""
    for day_offset in (0, 1):
        day = now.date() + timedelta(days=day_offset)
        for hour in COLLECT_HOURS:
            run_at = datetime.combine(day, time(hour), tzinfo=settings.TIMEZONE)
            if run_at > now:
                return run_at
    raise AssertionError("unreachable")


def _load_contest_korea_state(window_start: datetime) -> dict[str, int]:
    """목록별 마지막으로 수집한 콘테스트코리아 str_no 로드

    저장된 값이 없으면 수집 구간 시작일 0번으로 간주(해당 날짜 게시물부터 수집)
    """
    default = int(window_start.strftime("%Y%m%d") + "0000")
    try:
        state = json.loads(Path(settings.COLLECTOR_STATE_PATH).read_text("utf-8"))
        saved = state.get("contest_korea_last_str_no", {})
    except FileNotFoundError:
        saved = {}
    except (OSError, ValueError, AttributeError):
        logger.exception("수집 상태 파일을 읽을 수 없음: %s", settings.COLLECTOR_STATE_PATH)
        saved = {}

    return {
        name: int(saved[name]) if str(saved.get(name, "")).isdigit() else default
        for name in CONTEST_KOREA_LISTS
    }


def _save_contest_korea_state(last_str_nos: dict[str, int]) -> None:
    path = Path(settings.COLLECTOR_STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"contest_korea_last_str_no": last_str_nos}, indent=2), "utf-8"
    )
    tmp_path.replace(path)


async def collect_and_post(x_ai_api_key: str, run_at: datetime) -> None:
    """시립대 공지 및 콘테스트코리아에서 추출한 결과 리스트 전체 전송

    - 시립대: 직전 수집 시각 ~ run_at 사이에 게시된 공지
    - 콘테스트코리아: 마지막으로 저장한 str_no보다 큰 게시물
    """
    window_start = run_at - COLLECT_INTERVAL
    last_str_nos = _load_contest_korea_state(window_start)

    uos_results, (contest_korea_results, new_last_str_nos) = await asyncio.gather(
        collect_uos(window_start, run_at), collect_contest_korea(last_str_nos)
    )
    payload = uos_results + contest_korea_results
    if payload:
        await asyncio.to_thread(post_competition, payload, x_ai_api_key)
    else:
        logger.info("수집된 공모전이 없어 전송하지 않음")

    # 전송 성공 후에만 갱신하여 실패 시 다음 수집에서 다시 시도
    _save_contest_korea_state(new_last_str_nos)


async def run_daily_collector(x_ai_api_key: str) -> None:
    """매일 00시, 12시에 시립대 공지 및 콘테스트코리아에서 공모전 정보 수집 후 전송"""
    if not x_ai_api_key or not x_ai_api_key.strip():
        raise ValueError("X-AI-API-Key must be provided")

    while True:
        now = datetime.now(settings.TIMEZONE)
        run_at = _next_run_at(now)
        await asyncio.sleep((run_at - now).total_seconds())

        try:
            logger.info("공모전 정보 수집 시작 (%s)", run_at.isoformat())
            await collect_and_post(x_ai_api_key, run_at)
            logger.info("공모전 수집 및 전송 성공")
        except Exception as exc:
            logger.exception("공모전 수집 또는 전송 실패: %r", exc)


if __name__ == "__main__":
    asyncio.run(run_daily_collector(settings.X_AI_API_KEY))
