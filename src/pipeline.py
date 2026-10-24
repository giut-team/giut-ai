from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import joblib
import torch

from src.llm.field_extractor import (
    extract_fields_from_images,
    extract_fields_from_text,
    load_page_text,
)
from src.ml.image_classifier import ImageClassifier
from src.ml.text_classifier import TextClassifier, tokenize
from src.parsers.html_parser import HtmlParser
from src.parsers.contest_korea_parser import parse_contest_korea
from src.parsers.uos_rss_parser import parse_uos_rss
from src.utils.image_utils import download_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KEYS = ["link_density", "rel_pos"]


async def run_pipeline(url: str, use_classifier: bool = True):
    # URL의 텍스트 블록 중 relevant 본문을 모아 필드 추출 결과와 토큰 사용량 반환
    domain = (urlparse(url).hostname or "").lower()

    if domain in {"contestkorea.com", "www.contestkorea.com"}:
        page_text, main_image_link, detail_image_links = parse_contest_korea(url)
        image_urls = detail_image_links.insert(0, main_image_link)

    elif domain == "rss.uos.ac.kr":
        page_text = parse_uos_rss(url)

    else:
        async with HtmlParser() as parser:
            parsed_data = await parser.parse_url(url, render=True)
            blocks = parsed_data.text
            images = parsed_data.images
            links = parsed_data.links
        image_urls = []

        ##### 파싱 결과 기록
        # start_at = datetime.now()
        # with open(
        #     f"results/html_extracted_blocks_{str(start_at).replace(" ","_").replace(":","").replace(r"\.\d+","")}.json",
        #     "w",
        # ) as f:
        #     import json

        #     json.dump([parsed_data.text, parsed_data.images], f, ensure_ascii=False)
        #####

        relevant_blocks = []
        if blocks is None or len(blocks) == 0:
            return None

        if use_classifier:
            model_dir = (
                Path(__file__).resolve().parents[1] / "models" / "text_classifier"
            )
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
                for start in range(0, len(blocks), 64):
                    batch = blocks[start : start + 64]
                    enc = tokenize(batch)
                    # 분류기에 필요한 수치 피처를 학습 스케일로 변환
                    raw_num = [
                        [block["num_features"][key] for key in NUM_KEYS]
                        for block in batch
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

        page_text = load_page_text(relevant_blocks)

        ##### 분류기 결과 기록
        # with open(
        #     f"results/relevant_page_text_{str(start_at).replace(" ","_").replace(":","").replace(r"\.\d+","")}.txt",
        #     "w",
        # ) as f2:
        #     f2.write(page_text)
        #####

    extracted_fields, usage = await extract_fields_from_text(page_text)
    extracted_fields = extracted_fields.to_dict()

    # print()
    # print("Text:")
    # print(*(f"{k}: {v}\n" for k, v in extracted_fields.items()))
    # print()

    extract_failed_keys = []
    for key, value in extracted_fields.items():
        if value is None:
            extract_failed_keys.append(key)

    if extract_failed_keys == []:
        # 텍스트만으로 모두 추출되었을 경우
        return extracted_fields, usage

    # print("Extract Failed Keys:")
    # print(*extract_failed_keys)
    # print()

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

        # print("Size filtered images:")
        # print(*(image["src"] for image in size_filtered), sep="\n")
        # print()

        # CNN으로 포스터 사진만 분리
        relevant_images = []
        if size_filtered != [] and use_classifier:
            cnn_model_dir = (
                Path(__file__).resolve().parents[1] / "models" / "image_classifier"
            )
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
                downloaded_images = [
                    download_image(image["src"]) for image in size_filtered
                ]
                predictions = (
                    classifier.logits(downloaded_images).sigmoid() >= 0.5
                ).tolist()
                relevant_images.extend(
                    image
                    for image, prediction in zip(size_filtered, predictions)
                    if prediction == 1
                )
        else:
            relevant_images = size_filtered

        image_urls = [i["src"] for i in relevant_images]

        ##### 이미지 분류기 결과 기록
        # with open(
        #     f"results/relevant_images_{str(start_at).replace(" ","_").replace(":","").replace(r"\.\d+","")}.txt",
        #     "w",
        # ) as f3:
        #     for url in image_urls:
        #         f3.write(url)
        #####

    if image_urls:
        image_extracted_fields, usage = await extract_fields_from_images(
            image_urls, extract_failed_keys
        )
        image_extracted_fields = image_extracted_fields.to_dict()

        # 이미지에서 추출에 성공한 데이터만 기존 extracted_fields에 추가
        for key, value in image_extracted_fields.items():
            if value is not None:
                extracted_fields[key] = value

        # print("Images:")
        # print(*(f"{k}: {v}\n" for k, v in extracted_fields.items()))
        # print()

    return extracted_fields, usage


if __name__ == "__main__":
    import asyncio

    TEST_URL = (
        "https://www.wevity.com/?c=find&s=1&gub=1&cidx=21&gbn=view&gp=1&ix=110623"
    )

    print(asyncio.run(run_pipeline(TEST_URL)))
