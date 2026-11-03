from __future__ import annotations

from openai.types import CompletionUsage
import logging

from src.llm.openai_client import create_json_completion
from src.llm.dto import ExtractedFields, SelectedExtractedFields
from src.llm.prompts import FIELDS_DESCRIPTION, OPENAI_LLM_PROMPT, OPENAI_VLM_PROMPT

logger = logging.getLogger(__name__)

# TODO: usage 로그 작성 기능 추가
#       - usage를 함께 반환 안하고 로그 문서에만 따로 적기
#       - 파이프라인, 테스트 코드 수정


async def write_gpt_usage_log(usage: CompletionUsage):
    # 파일 열어서 동시에 기록할 수 있으므로 어떻게 동기화 처리할지 고민 필요.
    pass


async def extract_fields_from_text(
    page_text: str, keys: list[str] = FIELDS_DESCRIPTION.keys()
) -> tuple[ExtractedFields, CompletionUsage]:
    response = await create_json_completion(
        messages=[
            {
                "role": "developer",
                "content": [{"type": "text", "text": OPENAI_LLM_PROMPT}],
            },
            {"role": "user", "content": [{"type": "text", "text": page_text}]},
        ],
        response_format=ExtractedFields.select_fields(keys),
    )

    return (response.choices[0].message.parsed, response.usage)


async def extract_fields_from_images(
    image_urls: list[str], keys: list[str]
) -> tuple[SelectedExtractedFields, CompletionUsage]:
    # 추출할 키 및 관련 설명만 기본 프롬프트에 추가
    descriptions = [FIELDS_DESCRIPTION[key] for key in keys]
    prompt = OPENAI_VLM_PROMPT + "\n".join(descriptions)

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
            response_format=ExtractedFields.select_fields(keys),
        )

        return (response.choices[0].message.parsed, response.usage)

    except Exception as exc:
        logger.exception(exc)

    finally:
        return SelectedExtractedFields(), {}
