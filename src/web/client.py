import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping

import requests

COMPETITIONS_URL = "https://api.dev.giut.store/api/integrations/competitions"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def post_competition(
    extracted_fields: Mapping[str, Any],
    bearer_token: str,
    *,
    timeout: float = 10.0,
) -> requests.Response:
    # 추출된 필드를 JSON 본문으로 전송
    if not bearer_token or not bearer_token.strip():
        raise ValueError("bearer_token must be provided")

    payload = json.loads(json.dumps(dict(extracted_fields), default=_json_default))
    response = requests.post(
        COMPETITIONS_URL,
        json=payload,
        headers={"Authorization": f"Bearer {bearer_token.strip()}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response
