import re
import asyncio
import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
import joblib
import requests
import torch


from src.llm.dto import ActivityKindEnum, TargetParticipantsEnum
from src.llm.field_extractor import extract_fields_from_images, extract_fields_from_text
from src.ml.image_classifier import ImageClassifier
from src.ml.text_classifier import TextClassifier, tokenize
from src.parsers.html_parser import HtmlParser

from src.services.constants import (
    CAMPUSPICK_HOSTNAME,
    UOS_HOSTNAME,
    UrlType,
    SNS_DOMAINS,
    APPLY_PATH_KEYWORDS,
    IRRELEVANT_KEYWORDS,
    OFFICIAL_PAGE_KEYWORDS,
    DACON_HOSTNAME,
    DAKER_HOSTNAME,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KEYS = ["link_density", "rel_pos"]
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
TEXT_SCALER_PATH = MODEL_DIR / "text_classifier" / "num_scaler_v1.joblib"
TEXT_CLASSIFIER_PATH = MODEL_DIR / "text_classifier" / "text_classifier_head_v1.pt"
IMAGE_CLASSIFIER_PATH = MODEL_DIR / "image_classifier" / "image_classifier.pt"


@lru_cache(maxsize=1)
def get_models():
    """텍스트/이미지 분류기 모델 생성 및 가중치 로드"""
    scaler = joblib.load(TEXT_SCALER_PATH)
    text_classifier = TextClassifier(n_num=len(NUM_KEYS)).to(DEVICE)
    text_classifier.head.load_state_dict(
        torch.load(
            TEXT_CLASSIFIER_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )
    text_classifier.eval()

    image_classifier = ImageClassifier().to(DEVICE)
    image_classifier.model.load_state_dict(
        torch.load(
            IMAGE_CLASSIFIER_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )
    image_classifier.eval()

    return scaler, text_classifier, image_classifier


def is_eligible_activity(extracted_fields: dict) -> bool:
    """참여할 수 없는 활동이거나 팀 참가가 불가능한 경우 필터링 함수"""
    target_participants = extracted_fields.get("target_participants", None)
    activity_kind = extracted_fields.get("activity_kind", None)
    is_team = extracted_fields.get("is_team", None)

    if (
        target_participants == TargetParticipantsEnum.NONE.value
        or activity_kind == ActivityKindEnum.INVALID_ACTIVITY.value
        or is_team == False
    ):
        return False

    return True


def _link_context_blocks(link_blocks: list[dict]) -> list[dict]:
    blocks = []
    for block in link_blocks:
        urls = block.get("urls")
        if urls is None:
            continue
        for text, url in urls.items():
            if not re.match(r"^https?:\/\/", url):
                continue
            blocks.append(
                {
                    "url": url,
                    "text": text,
                    "line": block["text"],
                    "context": block["context"],
                }
            )
    return blocks


def _normalize_hostname(url: str) -> str:
    """도메인 비교를 위해 호스트를 소문자로 바꾸고 선행 www. 제거"""
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def filter_official_url(blocks: list[dict], source_url: str) -> str | None:
    """링크 목록 중 주최사 공식 홈페이지를 찾아 반환 (연관 공모전, 신청 폼, 주최사 SNS 등 제외)"""
    candidate_urls = []
    source_domain = _normalize_hostname(source_url)

    for block in blocks:
        url = block["url"]
        domain = _normalize_hostname(url)
        if re.search(domain, source_domain) or re.search(source_domain, domain):
            # 같은 도메인(연관공모전, 공모전 사이트 자체 게시판 페이지 등) 제외
            if domain == CAMPUSPICK_HOSTNAME:
                if url.find("linkclick") == -1:
                    continue  # 캠퍼스픽 redirect 경로는 포함
            elif domain == UOS_HOSTNAME or source_domain == UOS_HOSTNAME:
                if domain == source_domain:
                    continue  # UOS 공지는 도메인 완전 일치할 경우에만 제외
            else:
                continue
        if (source_domain == DACON_HOSTNAME and domain == DAKER_HOSTNAME) or (
            source_domain == DAKER_HOSTNAME and domain == DACON_HOSTNAME
        ):
            continue  # 데이콘-데이커 (연관 플랫폼) 필터링

        if any(re.search(sns, _normalize_hostname(url)) for sns in SNS_DOMAINS):
            continue  # SNS 주소 제외
        if any(re.search(keyword, url) for keyword in APPLY_PATH_KEYWORDS):
            continue  # 신청폼 주소 제외

        has_official_keyword = any(
            re.search(keyword, block["text"]) for keyword in OFFICIAL_PAGE_KEYWORDS
        )
        has_official_inline_keyword = any(
            re.search(keyword, block["line"]) for keyword in OFFICIAL_PAGE_KEYWORDS
        )
        has_official_context_keyword = any(
            re.search(keyword, block["context"]) for keyword in OFFICIAL_PAGE_KEYWORDS
        )

        has_irrelevant_keyword = any(
            re.search(keyword, block["text"]) for keyword in IRRELEVANT_KEYWORDS
        )
        has_irrelevant_inline_keyword = any(
            re.search(keyword, block["line"]) for keyword in IRRELEVANT_KEYWORDS
        )
        has_irrelevant_context_keyword = any(
            re.search(keyword, block["context"]) for keyword in IRRELEVANT_KEYWORDS
        )

        if (
            has_irrelevant_keyword
            or (not has_official_keyword and has_irrelevant_inline_keyword)
            or (
                not has_official_keyword
                and not has_irrelevant_inline_keyword
                and has_irrelevant_context_keyword
            )
        ):
            # irrelevant 키워드가 있으면 추출하지 않음(a 태그 자체 텍스트 -> 해당 라인의 텍스트 -> 이전 문맥 순으로 비교)
            continue

        candidate_level = (
            1
            if has_official_keyword
            else (
                2
                if (not has_official_keyword and has_official_inline_keyword)
                else (
                    3
                    if (
                        not has_official_keyword
                        and not has_official_inline_keyword
                        and has_official_context_keyword
                    )
                    else 4
                )
            )
        )

        candidate_urls.append({"url": url, "candidate_level": candidate_level})

    for level in range(1, 5):
        result = list(
            {url["url"] for url in candidate_urls if url["candidate_level"] == level}
        )
        if result:
            break

    if len(result) == 0:
        return None
    elif len(result) > 1:
        result = sorted(result, key=len)
    return result[0]


def _load_page_text(blocks: list[dict]) -> str:
    page_text = ""
    for b in blocks:
        block_text = b.get("text", None)
        if block_text:
            page_text += block_text + "\n"
    return page_text


def _download_image(url: str, timeout: float = 30.0) -> Image:
    """HTTP(S) 이미지 주소에서 바이트 데이터 다운로드"""
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("이미지 주소가 HTTP(S) URL이 아님")

    with requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout
    ) as response:
        response.raise_for_status()
        bytes = response.content

    with Image.open(BytesIO(bytes)) as image:
        image.load()
        return image


def run_text_classifier(text_blocks: list[dict]):
    relevant_blocks = []
    scaler, classifier, _ = get_models()

    with torch.inference_mode():
        for start in range(0, len(text_blocks), 64):
            batch = text_blocks[start : start + 64]
            enc = tokenize(batch)
            # 분류기에 필요한 수치 피처를 학습 스케일로 변환
            raw_num = [
                [block["num_features"][key] for key in NUM_KEYS] for block in batch
            ]
            num = torch.tensor(
                scaler.transform(raw_num), dtype=torch.float, device=DEVICE
            )
            predictions = (
                classifier.logits(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                    num,
                )
                .argmax(dim=-1)
                .tolist()
            )
            relevant_blocks.extend(
                block
                for block, prediction in zip(batch, predictions)
                if prediction == 1
            )

    return relevant_blocks


def run_image_classifier(image_blocks: list[dict]):
    valid_blocks = []
    downloaded_images = []
    for image in image_blocks:
        try:
            downloaded = _download_image(image["src"])
        except (requests.RequestException, OSError, ValueError) as exc:
            logger.warning("[SKIP] 이미지 다운로드 실패: %s (%s)", image["src"], exc)
            continue
        valid_blocks.append(image)
        downloaded_images.append(downloaded)

    if not downloaded_images:
        return []

    _, _, classifier = get_models()

    with torch.inference_mode():
        predictions = (classifier.logits(downloaded_images).sigmoid() >= 0.5).tolist()
        return [
            image
            for image, prediction in zip(valid_blocks, predictions)
            if prediction == 1
        ]


async def run_pipeline(url: str, use_classifier: bool = True):
    """URL의 텍스트 블록 중 relevant 본문을 모아 필드 추출 결과와 토큰 사용량 반환"""
    async with HtmlParser() as parser:
        parsed_data = await parser.parse_url(url, render=True)
        blocks = parsed_data.text
        images = parsed_data.images
        links = parsed_data.links

    # html 텍스트를 추출하지 못하거나 브라우저 접속 오류의 경우 예외 발생
    if blocks is None or len(blocks) == 0:
        raise ValueError("페이지 텍스트 추출에 실패했습니다.")

    relevant_blocks = run_text_classifier(blocks) if use_classifier else blocks

    page_text = _load_page_text(relevant_blocks)

    extracted_fields, usage = await extract_fields_from_text(page_text)
    extracted_fields = extracted_fields.to_dict()

    extracted_fields["urls"] = [
        {"type": UrlType.SOURCE_ORIGINAL, "url": url},
        {
            "type": UrlType.OFFICIAL,
            "url": filter_official_url(_link_context_blocks(links), url),
        },
    ]  # 수집 경로 및 공식 홈페이지 URL 추가

    if not is_eligible_activity(extracted_fields):
        return extracted_fields, usage

    missing_keys = [key for key, value in extracted_fields.items() if value is None]

    image_urls = []
    if missing_keys and images:
        # 픽셀 단위 면적, 브라우저 내 상대 사이즈, 종횡비 룰 기반 분류
        size_filtered = []
        for image in images:
            size = image["size"]
            if (
                size["display_area"] > 400 * 400
                and size["relative_display_area"] > 0.6
                and size["display_height"] / size["display_width"] > 0.7
            ):
                size_filtered.append(image)

        # CNN으로 포스터 사진만 분리
        if size_filtered != [] and use_classifier:
            relevant_images = await asyncio.to_thread(
                run_image_classifier, size_filtered
            )
        else:
            relevant_images = size_filtered

        image_urls = [i["src"] for i in relevant_images]

    if image_urls != []:
        image_extracted_fields, usage = await extract_fields_from_images(
            image_urls, missing_keys
        )
        image_extracted_fields = image_extracted_fields.to_dict()

        # 이미지에서 추출에 성공한 데이터만 기존 extracted_fields에 추가
        for key, value in image_extracted_fields.items():
            if value is not None:
                extracted_fields[key] = value

    return extracted_fields, usage


if __name__ == "__main__":
    import asyncio

    TEST_URL = (
        "https://www.wevity.com/?c=find&s=1&gub=1&cidx=21&gbn=view&gp=1&ix=110623"
    )

    print(asyncio.run(run_pipeline(TEST_URL)))
