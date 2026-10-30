from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
import joblib
import requests
import torch

from src.llm.dto import TargetParticipantsEnum
from src.llm.field_extractor import extract_fields_from_images, extract_fields_from_text
from src.ml.image_classifier import ImageClassifier
from src.ml.text_classifier import TextClassifier, tokenize
from src.parsers.html_parser import HtmlParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KEYS = ["link_density", "rel_pos"]


def load_page_text(blocks: list[dict]) -> str:
    page_text = ""

    for b in blocks:
        block_text = b.get("text", None)
        if block_text is not None:
            page_text += block_text + "\n"

    return page_text


def is_eligible_activity(extracted_fields: dict) -> bool:
    # 참여할 수 없는 활동이거나 팀 참가가 불가능한지 확인
    target_participants = extracted_fields.get("target_participants", None)
    activity_kind = extracted_fields.get("activity_kind", None)
    is_team = extracted_fields.get("is_team", None)

    if (
        target_participants is None
        or target_participants == TargetParticipantsEnum.NONE.value
        or activity_kind is None
        or is_team == False
    ):
        return False

    return True


def download_image(url: str, timeout: float = 30.0) -> Image:
    # HTTP(S) 이미지 주소에서 바이트 데이터 다운로드
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

    model_dir = Path(__file__).resolve().parents[1] / "models" / "text_classifier"
    scaler = joblib.load(model_dir / "num_scaler_v1.joblib")
    classifier = TextClassifier(n_num=len(NUM_KEYS)).to(DEVICE)
    classifier.head.load_state_dict(
        torch.load(
            model_dir / "text_classifier_head_v1.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )
    classifier.eval()

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


def run_image_classifier(image_blocks: list[str]):
    relevant_images = []

    cnn_model_dir = Path(__file__).resolve().parents[1] / "models" / "image_classifier"
    classifier = ImageClassifier().to(DEVICE)
    classifier.model.load_state_dict(
        torch.load(
            cnn_model_dir / "image_classifier.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )
    classifier.eval()

    with torch.inference_mode():
        downloaded_images = [download_image(image["src"]) for image in image_blocks]
        predictions = (classifier.logits(downloaded_images).sigmoid() >= 0.5).tolist()
        relevant_images.extend(
            image
            for image, prediction in zip(image_blocks, predictions)
            if prediction == 1
        )


async def run_pipeline(url: str, use_classifier: bool = True):
    # URL의 텍스트 블록 중 relevant 본문을 모아 필드 추출 결과와 토큰 사용량 반환
    async with HtmlParser() as parser:
        parsed_data = await parser.parse_url(url, render=True)
        blocks = parsed_data.text
        images = parsed_data.images
        links = parsed_data.links

    if blocks is None or len(blocks) == 0:
        return None, None

    if use_classifier:
        relevant_blocks = run_text_classifier(blocks)

    page_text = load_page_text(relevant_blocks)

    extracted_fields, usage = await extract_fields_from_text(page_text)
    extracted_fields = extracted_fields.to_dict()

    missing_keys = [key for key, value in extracted_fields.items() if value is None]

    if missing_keys == []:
        # 텍스트만으로 모두 추출되었을 경우
        return extracted_fields, usage

    if images != [] and image_urls == []:
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
            relevant_images = run_image_classifier(size_filtered)
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
