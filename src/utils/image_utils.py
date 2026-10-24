import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
import requests


def download_image_bytes(url: str, timeout: float = 30.0) -> bytes:
    # HTTP(S) 이미지 주소에서 바이트 데이터 다운로드
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("이미지 주소가 HTTP(S) URL이 아님")

    with requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout
    ) as response:
        response.raise_for_status()
        return response.content


def _convert_image_to_base64(image_path: str) -> str:
    # 로컬 이미지 파일 또는 HTTP(S) 이미지를 Base64 데이터 URL로 변환
    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }

    if image_path.lower().startswith(("http://", "https://")):
        image_bytes = download_image_bytes(image_path)
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
    base64_url = f"data:{mime_type};base64,{encoded}"

    return base64_url


def convert_images_to_base64(images_paths: list[str]) -> list[str]:
    # 이미지 리스트를 Base64 데이터 URL 리스트로 변환
    base64_urls: list[str] = []

    for image_path in images_paths:
        url = _convert_image_to_base64(image_path)
        base64_urls.append(url)

    return base64_urls
