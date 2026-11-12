import base64
from io import BytesIO
from urllib.parse import urlparse

from PIL import Image
import requests

# OpenAI 비전 입력이 지원하는 포맷
_SUPPORTED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
# OpenAI 입력될 때 긴 변이 2048px로 줄어들므로 그보다 크면 미리 축소해 페이로드를 줄임
_MAX_SIDE = 2048


def download_image_bytes(url: str, timeout: float = 30.0) -> bytes:
    """HTTP(S) 이미지 주소 또는 Base64 데이터 URL에서 원본 바이트를 로드"""
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() == "data":
        header, separator, encoded = url.partition(",")
        if (
            not separator
            or not header.lower().startswith("data:image/")
            or not header.lower().endswith(";base64")
        ):
            raise ValueError("이미지 주소가 Base64 이미지 데이터 URL이 아님")
        return base64.b64decode(encoded, validate=True)

    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(
            "이미지 주소가 HTTP(S) URL 또는 Base64 이미지 데이터 URL이 아님"
        )

    with requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout
    ) as response:
        response.raise_for_status()
        return response.content


def download_image(url: str, timeout: float = 30.0) -> Image.Image:
    """이미지 주소를 PIL 이미지로 로드"""
    with Image.open(BytesIO(download_image_bytes(url, timeout))) as image:
        image.load()
        return image


def to_base64_data_url(url: str, timeout: float = 30.0) -> str:
    """이미지를 직접 다운로드해 OpenAI에 넣을 수 있는 Base64 데이터 URL로 변환"""
    if urlparse(url).scheme.lower() == "data":
        return url

    image_bytes = download_image_bytes(url, timeout)
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()  # 이미지가 아닌 응답(HTML 등)이면 여기서 예외 발생
        mime = _SUPPORTED_FORMATS.get(image.format)

        if mime is None or max(image.size) > _MAX_SIDE:
            # 미지원 포맷(GIF, BMP 등)이거나 너무 큰 경우 재인코딩
            image.thumbnail((_MAX_SIDE, _MAX_SIDE))
            buffer = BytesIO()
            if image.mode in ("RGBA", "LA", "P"):
                image.save(buffer, format="PNG")
                mime = "image/png"
            else:
                image.convert("RGB").save(buffer, format="JPEG", quality=90)
                mime = "image/jpeg"
            image_bytes = buffer.getvalue()

    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
