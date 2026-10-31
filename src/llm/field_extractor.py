from __future__ import annotations

from openai.types import CompletionUsage

from src.llm.openai_client import create_json_completion
from src.llm.dto import ExtractedFields
from src.llm.prompts import FIELDS_DESCRIPTION, OPENAI_LLM_PROMPT, OPENAI_VLM_PROMPT


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
) -> tuple[ExtractedFields, CompletionUsage]:
    # 추출할 키 및 관련 설명만 기본 프롬프트에 추가
    descriptions = [FIELDS_DESCRIPTION[key] for key in keys]
    prompt = OPENAI_VLM_PROMPT + "\n".join(descriptions)

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
