from datetime import date, datetime, tzinfo
import glob
import json
import logging
from logging.handlers import BaseRotatingHandler
import os
from pathlib import Path
import shutil
from threading import Lock
from typing import Any, Literal

from openai.types import CompletionUsage
from pydantic import BaseModel
import requests

from src.config import BASE_DIR, settings

logger = logging.getLogger(__name__)

_usage_logger = logging.getLogger(f"{__name__}.gpt_usage")
_usage_logger_lock = Lock()
_request_logger = logging.getLogger(f"{__name__}.backend_requests")
_request_logger_lock = Lock()
_api_logger = logging.getLogger(f"{__name__}.api_requests")
_api_logger_lock = Lock()


class DailyRotatingFileHandler(BaseRotatingHandler):
    """지정한 타임존의 날짜가 바뀌면 기존 파일을 `<파일명>.YYYY-MM-DD`로 넘기는 핸들러

    TimedRotatingFileHandler는 UTC/서버 로컬 시간만 지원하므로 타임존을 직접 받도록 구현함
    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        tz: tzinfo,
        backup_count: int = 30,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(filename, "a", encoding=encoding)
        self.tz = tz
        self.backup_count = backup_count
        # 기존 파일이 있으면 마지막 수정 날짜를 현재 파일의 날짜로 간주
        if os.path.exists(self.baseFilename):
            mtime = os.stat(self.baseFilename).st_mtime
            self._current_date = datetime.fromtimestamp(mtime, tz).date()
        else:
            self._current_date = datetime.now(tz).date()
        self._next_date: date = self._current_date

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        self._next_date = datetime.fromtimestamp(record.created, self.tz).date()
        return self._next_date != self._current_date

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None

        if os.path.exists(self.baseFilename):
            dfn = self.rotation_filename(
                f"{self.baseFilename}.{self._current_date.isoformat()}"
            )
            if os.path.exists(dfn):
                # 같은 날짜 파일이 이미 있으면 덮어쓰지 않고 이어 붙임
                with open(self.baseFilename, "rb") as src, open(dfn, "ab") as dst:
                    shutil.copyfileobj(src, dst)
                os.remove(self.baseFilename)
            else:
                self.rotate(self.baseFilename, dfn)

        if self.backup_count > 0:
            backups = sorted(
                glob.glob(f"{glob.escape(self.baseFilename)}.[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
            )
            for old in backups[: -self.backup_count]:
                os.remove(old)

        self._current_date = self._next_date
        self.stream = self._open()


def _ensure_file_handler(
    target: logging.Logger, lock: Lock, log_dir: str, filename: str
) -> None:
    """최초 호출 시 한 번만 날짜별 파일 핸들러를 붙임

    핸들러 생성을 직렬화하여 스레드에서 호출해도 중복 등록 방지됨
    """
    with lock:
        if target.handlers:
            return
        path = Path(log_dir).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        handler = DailyRotatingFileHandler(
            path / filename,
            tz=settings.TIMEZONE,
            backup_count=30,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        target.setLevel(logging.INFO)
        target.propagate = False
        target.addHandler(handler)


def write_gpt_usage_log(
    usage: CompletionUsage | None,
    *,
    source_url: str,
    request_type: Literal["text", "image"],
    # run_id: str,
    model: str = settings.OPENAI_MODEL,
    started_at: datetime,
    ended_at: datetime,
    # response_id: str | None = None,
    status: Literal["success", "error"] = "success",
    error_type: str | None = None,
) -> None:
    """시작·종료 시각으로 elapsed_ms를 계산하고 GPT 호출 한 건을 JSONL로 기록

    잘못된 인자는 ValueError로 알리지만, 파일 기록 실패는 GPT 호출 결과에 영향을 주지 않도록
    로그만 남기고 삼킴
    """
    if started_at.utcoffset() is None or ended_at.utcoffset() is None:
        raise ValueError("started_at, ended_at에 시간대가 필요합니다.")
    if request_type not in {"text", "image"}:
        raise ValueError("request_type은 text 또는 image여야 합니다.")

    started_at = started_at.astimezone(settings.TIMEZONE)
    ended_at = ended_at.astimezone(settings.TIMEZONE)
    elapsed_ms = (ended_at - started_at).total_seconds() * 1000
    if elapsed_ms < 0:
        # datetime.now()는 벽시계라 NTP 보정 등으로 뒤로 갈 수 있으므로 0으로 기록
        logger.warning(
            "GPT 호출 종료 시각이 시작 시각보다 빠름: %s < %s", ended_at, started_at
        )
        elapsed_ms = 0.0

    prompt_details = usage.prompt_tokens_details if usage else None
    completion_details = usage.completion_tokens_details if usage else None
    record = {
        # "run_id": run_id,
        "source_url": source_url,
        "type": request_type,
        "model": model,
        # "response_id": response_id,
        "status": status,
        "error_type": error_type,
        "started_at": started_at.isoformat(timespec="milliseconds"),
        "ended_at": ended_at.isoformat(timespec="milliseconds"),
        "elapsed_ms": round(elapsed_ms, 3),
        "input_tokens": usage.prompt_tokens if usage else None,
        "cached_input_tokens": prompt_details.cached_tokens if prompt_details else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "reasoning_tokens": (
            completion_details.reasoning_tokens if completion_details else None
        ),
    }
    try:
        message = json.dumps(record, ensure_ascii=False, allow_nan=False)
        _ensure_file_handler(
            _usage_logger,
            _usage_logger_lock,
            settings.GPT_USAGE_LOG_DIR,
            "gpt_usage.jsonl",
        )
        _usage_logger.info(message)
    except Exception:
        # 기록 실패가 GPT 추출 결과에 영향을 주지 않도록 로그만 남김
        logger.exception("GPT 사용량 기록 실패")


def _response_body_for_log(response: requests.Response | None) -> Any:
    """201 응답은 JSON 객체로, 그 외에는 원문 문자열로 반환"""
    if response is None:
        return None
    if response.status_code == 201:
        try:
            return response.json()
        except ValueError:
            pass
    return response.text


def write_backend_request_log(
    payload: list[dict],
    *,
    url: str,
    sent_at: datetime,
    response: requests.Response | None,
    error: Exception | None,
) -> None:
    """백엔드로 보낸 요청 본문과 응답 결과를 JSON 파일로 기록"""
    try:
        record = {
            "sent_at": sent_at.isoformat(timespec="seconds"),
            "url": url,
            "count": len(payload),
            "status_code": response.status_code if response is not None else None,
            "response_body": _response_body_for_log(response),
            "error": repr(error) if error is not None else None,
            "payload": payload,
        }
        message = json.dumps(record, ensure_ascii=False)

        _ensure_file_handler(
            _request_logger,
            _request_logger_lock,
            settings.BACKEND_REQUEST_LOG_DIR,
            "backend_requests.json",
        )
        _request_logger.info(message)
    except Exception:
        # 기록 실패가 전송 결과에 영향을 주지 않도록 로그만 남김
        logger.exception("백엔드 요청 기록 실패")


def _body_for_log(body: Any) -> Any:
    """요청·응답 본문을 JSON으로 기록할 수 있는 형태로 변환

    pydantic 모델은 dict로, bytes는 JSON이면 파싱하고 아니면 문자열로 변환
    """
    if isinstance(body, BaseModel):
        return body.model_dump(mode="json")
    if isinstance(body, (bytes, bytearray)):
        text = body.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except ValueError:
            return text
    return body


def write_api_request_log(
    *,
    method: str,
    path: str,
    request_body: Any,
    received_at: datetime,
    responded_at: datetime,
    status_code: int | None,
    response_body: Any,
    error: Exception | None = None,
) -> None:
    """API로 받은 요청과 응답 결과를 JSONL 한 줄로 기록"""
    try:
        received_at = received_at.astimezone(settings.TIMEZONE)
        responded_at = responded_at.astimezone(settings.TIMEZONE)
        record = {
            "received_at": received_at.isoformat(timespec="milliseconds"),
            "responded_at": responded_at.isoformat(timespec="milliseconds"),
            "elapsed_ms": round(
                (responded_at - received_at).total_seconds() * 1000, 3
            ),
            "method": method,
            "path": path,
            "status_code": status_code,
            "error": repr(error) if error is not None else None,
            "request_body": _body_for_log(request_body),
            "response_body": _body_for_log(response_body),
        }
        # 응답에 date·Enum 등이 섞여 있어도 기록이 실패하지 않도록 문자열로 변환
        message = json.dumps(record, ensure_ascii=False, default=str)

        _ensure_file_handler(
            _api_logger,
            _api_logger_lock,
            settings.API_REQUEST_LOG_DIR,
            "api_requests.jsonl",
        )
        _api_logger.info(message)
    except Exception:
        # 기록 실패가 API 응답에 영향을 주지 않도록 로그만 남김
        logger.exception("API 요청 기록 실패")
