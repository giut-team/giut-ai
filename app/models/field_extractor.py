from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
import re
import sys
import datetime
from PIL import Image

from app.models.openai_client import create_json_completion
from app.models.dto import ExtractedFields
from app.models.prompts import OPENAI_LLM_PROMPT, OPENAI_VLM_PROMPT


def _load_page_text(blocks: list[dict]) -> str:
    page_text = ""

    for b in blocks:
        block_text = b.get("text", None)
        if block_text is not None:
            page_text.append(block_text + "\n")

    return page_text


async def extract_fields_from_text(page_text: str) -> ExtractedFields:
    response = await create_json_completion(
        messages=[
            {
                "role": "developer",
                "content": {"type": "text", "text": OPENAI_LLM_PROMPT},
            },
            {"role": "user", "content": {"type": "text", "text": page_text}},
        ],
        response_format=ExtractedFields,
    )

    return response.choices[0].message.parsed


def convert_images_to_base64(images_paths: list[str]) -> list[str]:
    # 로컬 이미지 파일을 Base64 데이터 URL로 변환
    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }
    data_urls: list[str] = []

    for image_path in images_paths:
        image_bytes = Path(image_path).read_bytes()

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            mime_type = mime_types.get(image.format)

            if mime_type is None:
                # 이미지 형식이 mime_types 안에 없을 경우 PNG로 변환
                with BytesIO() as buffer:
                    image.convert("RGBA").save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                mime_type = "image/png"

        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_urls.append(f"data:{mime_type};base64,{encoded}")

    return data_urls


async def extract_fields_from_images(images: list[Image.Image]) -> ExtractedFields:
    response = await create_json_completion(
        messages=[
            {
                "role": "developer",
                "content": {"type": "text", "text": OPENAI_VLM_PROMPT},
            },
            {"role": "user", "content": {"type": "image", "image_url": ""}},
        ],
        response_format=ExtractedFields,
    )

    return response.choices[0].message.parsed


images_paths = [
    ["campuspick.jpg"],
    ["dacon.jpg"],
    ["thinkyou_1.png"],
    ["thinkyou_2.jpg"],
    ["wevity_1.jpg", "wevity_2.jpg"],
]


if __name__ == "__main__":

    PAGE_ID = 0

    # text
    with open("html_data/test/test_blocks.json", "r") as f:
        blocks = json.load(f)["text"]
        page_blocks = []

        for b in blocks:
            if b["page_id"] > PAGE_ID:
                break
            if b["page_id"] == PAGE_ID:
                page_blocks.append(b)

    page_text = _load_page_text(page_blocks)

    ## image
    # images = [Image.open(path).convert("RGB") for path in images_paths[4]]

    started_at = datetime.now()
    fields_result = extract_fields_from_text(page_text)
    # fields_result = extract_fields_from_images(images)
    ended_at = datetime.now()

    print(fields_result.format())
    print("time elapsed:", ended_at - started_at)
