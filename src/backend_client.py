from datetime import datetime
import logging
from typing import Any, Mapping

import requests
from pydantic import TypeAdapter

from src.api.dto import CompetitionExtractResponse
from src.config import settings
from src.utils.log_handler import write_backend_request_log

COMPETITIONS_API_URL = settings.COMPETITIONS_API_URL
logger = logging.getLogger(__name__)

_competition_list_adapter = TypeAdapter(list[CompetitionExtractResponse])


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
        write_backend_request_log(
            payload,
            url=COMPETITIONS_API_URL,
            sent_at=sent_at,
            response=response,
            error=error,
        )

    if not response.ok:
        logger.error(
            "백엔드 전송 실패 (%s): %s", response.status_code, response.text[:2000]
        )
    response.raise_for_status()
    return response
