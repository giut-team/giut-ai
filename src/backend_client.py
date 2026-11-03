import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping

import requests

COMPETITIONS_URL = "https://api.dev.giut.store/api/integrations/competitions/batch"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def post_competition(
    extracted_fields: Mapping[str, Any] | list[Mapping[str, Any]],
    x_ai_api_key: str,
    *,
    timeout: float = 10.0,
) -> requests.Response:
    """추출된 필드를 JSON 본문으로 전송"""
    if not x_ai_api_key or not x_ai_api_key.strip():
        raise ValueError("X-AI-API-Key must be provided")

    if isinstance(extracted_fields, list):
        fields = [dict(item) for item in extracted_fields]
    else:
        fields = dict(extracted_fields)
    payload = json.loads(json.dumps(fields, default=_json_default))
    response = requests.post(
        COMPETITIONS_URL,
        json=payload,
        headers={
            "X-AI-API-Key": f"{x_ai_api_key.strip()}",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response
