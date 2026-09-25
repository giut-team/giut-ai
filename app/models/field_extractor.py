from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

import requests
from PIL import Image

from app.models.openai_client import create_json_completion
from app.models.dto import ExtractedFields
from app.models.prompts import OPENAI_LLM_PROMPT, OPENAI_VLM_PROMPT


def load_page_text(blocks: list[dict]) -> str:
    page_text = ""

    for b in blocks:
        block_text = b.get("text", None)
        if block_text is not None:
            page_text += block_text + "\n"

    return page_text


async def extract_fields_from_text(page_text: str) -> ExtractedFields:
    response = await create_json_completion(
        messages=[
            {
                "role": "developer",
                "content": [{"type": "text", "text": OPENAI_LLM_PROMPT}],
            },
            {"role": "user", "content": [{"type": "text", "text": page_text}]},
        ],
        response_format=ExtractedFields,
    )

    return response.choices[0].message.parsed


def _download_image_bytes(url: str, timeout: float = 30.0) -> bytes:
    # HTTP(S) 이미지 주소에서 바이트 데이터 다운로드
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("이미지 주소가 HTTP(S) URL이 아님")

    with requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout
    ) as response:
        response.raise_for_status()
        return response.content


def convert_images_to_base64(images_paths: list[str]) -> list[str]:
    # 로컬 이미지 파일 또는 HTTP(S) 이미지를 Base64 데이터 URL로 변환
    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }
    data_urls: list[str] = []

    for image_path in images_paths:
        if image_path.lower().startswith(("http://", "https://")):
            image_bytes = _download_image_bytes(image_path)
        else:
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
            {
                "role": "user",
                "content": [{"type": "image", "image_url": url} for url in images],
            },
        ],
        response_format=ExtractedFields,
    )

    return response.choices[0].message.parsed
