from datetime import datetime
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import requests
from pydantic import TypeAdapter

from src.api.dto import CompetitionExtractResponse
from src.config import BASE_DIR, settings

COMPETITIONS_API_URL = settings.COMPETITIONS_API_URL
logger = logging.getLogger(__name__)

_competition_list_adapter = TypeAdapter(list[CompetitionExtractResponse])
_request_log_lock = Lock()


def _write_request_log(
    payload: list[dict],
    *,
    sent_at: datetime,
    response: requests.Response | None,
    error: Exception | None,
) -> None:
    """백엔드로 보낸 요청 본문과 응답 결과를 JSONL 한 줄로 기록"""
    try:
        log_dir = Path(settings.BACKEND_REQUEST_LOG_DIR).expanduser()
        if not log_dir.is_absolute():
            log_dir = BASE_DIR / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "sent_at": sent_at.isoformat(timespec="seconds"),
            "url": COMPETITIONS_API_URL,
            "count": len(payload),
            "status_code": response.status_code if response is not None else None,
            "response_body": response.text if response is not None else None,
            "error": repr(error) if error is not None else None,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _request_log_lock:
            with open(log_dir / "backend_requests.jsonl", "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        # 기록 실패가 전송 결과에 영향을 주지 않도록 로그만 남김
        logger.exception("백엔드 요청 기록 실패")


def post_competition(
    extracted_fields: list[Mapping[str, Any]],
    x_ai_api_key: str,
    *,
    timeout: float = 10.0,
) -> requests.Response:
    """추출된 필드를 list[CompetitionExtractResponse] 형식으로 검증 후 전송"""
    if not x_ai_api_key or not x_ai_api_key.strip():
        raise ValueError("X-AI-API-Key must be provided")

    competitions = _competition_list_adapter.validate_python(extracted_fields)
    payload = _competition_list_adapter.dump_python(competitions, mode="json")

    sent_at = datetime.now(settings.TIMEZONE)
    response = None
    error = None
    try:
        response = requests.post(
            COMPETITIONS_API_URL,
            json=payload,
            headers={
                "X-AI-API-Key": f"{x_ai_api_key.strip()}",
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        error = exc
        raise
    finally:
        _write_request_log(payload, sent_at=sent_at, response=response, error=error)

    if not response.ok:
        logger.error(
            "백엔드 전송 실패 (%s): %s", response.status_code, response.text[:2000]
        )
    response.raise_for_status()
    return response
