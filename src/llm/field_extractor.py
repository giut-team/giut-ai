from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlparse

from openai import BadRequestError
from openai.types import CompletionUsage
import requests

from src.config import BASE_DIR, settings
from src.llm.openai_client import create_json_completion
from src.llm.dto import ExtractedFields
from src.llm.prompts import (
    FIELDS_DESCRIPTION,
    CONTEST_LLM_PROMPT,
    CONTEST_VLM_PROMPT,
    UOS_LLM_PROMPT,
    UOS_VLM_PROMPT,
)
from src.utils.image import to_base64_data_url

logger = logging.getLogger(__name__)
# OpenAI가 이미지 URL을 가져오거나 해석하지 못했을 때 반환하는 에러 코드
_IMAGE_FETCH_ERROR_CODES = {
    "invalid_image_url",
    "invalid_image_format",
    "invalid_image",
}
_usage_logger = logging.getLogger(f"{__name__}.usage")
_usage_logger_lock = Lock()


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
    """시작·종료 시각으로 elapsed_ms를 계산하고 GPT 호출 한 건을 JSONL로 기록"""
    if started_at.utcoffset() is None or ended_at.utcoffset() is None:
        raise ValueError("started_at, ended_at에 시간대가 필요합니다.")
    if request_type not in {"text", "image"}:
        raise ValueError("request_type은 text 또는 image여야 합니다.")

    started_at = started_at.astimezone(timezone.utc)
    ended_at = ended_at.astimezone(timezone.utc)
    elapsed_ms = (ended_at - started_at).total_seconds() * 1000
    if elapsed_ms < 0:
        raise ValueError("ended_at은 started_at보다 빠를 수 없습니다.")

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
    message = json.dumps(record, ensure_ascii=False, allow_nan=False)

    # 최초 핸들러 생성을 직렬화하여 스레드에서 호출해도 중복 등록 방지됨
    with _usage_logger_lock:
        if not _usage_logger.handlers:
            log_dir = Path(settings.GPT_USAGE_LOG_DIR).expanduser()
            if not log_dir.is_absolute():
                log_dir = BASE_DIR / log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = TimedRotatingFileHandler(
                log_dir / "gpt_usage.jsonl",
                when="midnight",
                backupCount=30,
                encoding="utf-8",
                utc=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            _usage_logger.setLevel(logging.INFO)
            _usage_logger.propagate = False
            _usage_logger.addHandler(handler)

    _usage_logger.info(message)


async def extract_fields_from_text(
    page_text: str,
    source_url: str,
    keys: list[str] = FIELDS_DESCRIPTION.keys(),
) -> dict:
    hostname = urlparse(source_url).hostname or ""
    if hostname == "uos.ac.kr" or hostname.endswith(".uos.ac.kr"):
        # 시립대(uos.ac.kr) 공지는 전용 프롬프트 사용
        prompt = UOS_LLM_PROMPT
    else:
        prompt = CONTEST_LLM_PROMPT

    # 추출할 키 및 관련 설명만 기본 프롬프트에 추가
    descriptions = [FIELDS_DESCRIPTION[key] for key in keys]
    prompt += "\n".join(descriptions)

    try:
        started_at = datetime.now(timezone.utc)
        response = await create_json_completion(
            messages=[
                {
                    "role": "developer",
                    "content": [{"type": "text", "text": prompt}],
                },
                {"role": "user", "content": [{"type": "text", "text": page_text}]},
            ],
            response_format=ExtractedFields.select_fields(keys),
        )
        ended_at = datetime.now(timezone.utc)

        write_gpt_usage_log(
            response.usage,
            source_url=source_url,
            request_type="text",
            started_at=started_at,
            ended_at=ended_at,
            status="success",
        )
        return response.choices[0].message.parsed.to_dict()

    except Exception as exc:
        logger.exception(exc)
        write_gpt_usage_log(
            None,
            source_url=source_url,
            request_type="text",
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
            status="error",
            error_type=type(exc).__name__,
        )
        return {}


async def extract_fields_from_images(
    image_urls: list[str],
    source_url: str,
    keys: list[str] = FIELDS_DESCRIPTION.keys(),
) -> dict:
    hostname = urlparse(source_url).hostname or ""
    if hostname == "uos.ac.kr" or hostname.endswith(".uos.ac.kr"):
        # 시립대(uos.ac.kr) 공지는 전용 프롬프트 사용
        prompt = UOS_VLM_PROMPT
    else:
        prompt = CONTEST_VLM_PROMPT

    # 추출할 키 및 관련 설명만 기본 프롬프트에 추가
    descriptions = [FIELDS_DESCRIPTION[key] for key in keys]
    prompt += "\n".join(descriptions)
    try:
        response_format = ExtractedFields.select_fields(keys)
        return await _request_image_fields(
            prompt, image_urls, source_url, response_format
        )
    except BadRequestError as exc:
        if not _is_image_fetch_error(exc):
            logger.exception(exc)
            return {}
        logger.warning(
            "[이미지 재시도] OpenAI 이미지 로드 실패, Base64로 변환해 재요청: %s (%s)",
            source_url,
            exc.message,
        )
    except Exception as exc:
        logger.exception(exc)
        return {}

    # OpenAI가 URL 이미지를 가져오지 못한 경우 직접 다운로드해 Base64로 전달
    data_urls = []
    for url in image_urls:
        try:
            data_urls.append(await asyncio.to_thread(to_base64_data_url, url))
        except (requests.RequestException, OSError, ValueError) as exc:
            logger.warning("[SKIP] 이미지 Base64 변환 실패: %s (%s)", url, exc)

    if not data_urls:
        return {}

    try:
        return await _request_image_fields(
            prompt, data_urls, source_url, response_format
        )
    except Exception as exc:
        logger.exception(exc)
        return {}


def _is_image_fetch_error(exc: BadRequestError) -> bool:
    """OpenAI가 이미지 URL을 다운로드·해석하지 못해 발생한 400 에러인지 판별"""
    return exc.code in _IMAGE_FETCH_ERROR_CODES or "downloading" in exc.message


async def _request_image_fields(
    prompt: str,
    image_urls: list[str],
    source_url: str,
    response_format: type,
) -> dict:
    """이미지 필드 추출 요청 1회 수행 및 사용량 기록 (실패 시 기록 후 예외 전파)"""
    started_at = datetime.now(timezone.utc)
    try:
        response = await create_json_completion(
            messages=[
                {
                    "role": "developer",
                    "content": [{"type": "text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ],
                },
            ],
            response_format=response_format,
        )
    except Exception as exc:
        write_gpt_usage_log(
            None,
            source_url=source_url,
            request_type="image",
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
            status="error",
            error_type=type(exc).__name__,
        )
        raise

    write_gpt_usage_log(
        response.usage,
        source_url=source_url,
        request_type="image",
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        status="success",
    )
    return response.choices[0].message.parsed.to_dict()
