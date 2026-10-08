from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import joblib
import torch

from src.llm.field_extractor import extract_fields_from_text, load_page_text
from src.ml.text_classifier import TextClassifier, tokenize
from src.parsers.html_parser import HtmlParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KEYS = ["link_density", "rel_pos"]


async def run_pipeline(url: str, use_classifier: bool = True):
    # URL의 텍스트 블록 중 relevant 본문을 모아 필드 추출 결과와 토큰 사용량 반환
    domain = (urlparse(url).hostname or "").lower()
    if domain in {"rss.uos.ac.kr", "contestkorea.com", "www.contestkorea.com"}:
        # 별도 전처리 연결 예정
        return None

    async with HtmlParser() as parser:
        blocks = (await parser.parse_url(url, render=True)).text

    ##### 파싱 결과 기록
    # start_at = datetime.now()
    # with open(
    #     f"results/html_extracted_blocks_{str(start_at).replace(" ","_").replace(":","")}.json",
    #     "w",
    # ) as f:
    #     import json
    #     json.dump(blocks, f, ensure_ascii=False)
    #####

    relevant_blocks = []
    if blocks is None or len(blocks) == 0:
        return None

    if use_classifier:
        model_dir = Path(__file__).resolve().parents[1] / "models" / "text_classifier"
        scaler = joblib.load(model_dir / "num_scaler.joblib")
        classifier = TextClassifier(n_num=len(NUM_KEYS)).to(DEVICE)
        classifier.head.load_state_dict(
            torch.load(
                model_dir / "text_classifier_head.pt",
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

    page_text = load_page_text(relevant_blocks)

    ##### 분류기 결과 기록
    # with open(
    #     f"results/relevant_page_text_{str(start_at).replace(" ","_").replace(":","")}.txt",
    #     "w",
    # ) as f2:
    #     f2.write(page_text)
    #####

    return await extract_fields_from_text(page_text)


if __name__ == "__main__":
    import asyncio

    TEST_URL = (
        "https://www.wevity.com/?c=find&s=1&gub=1&cidx=21&gbn=view&gp=1&ix=110623"
    )

    print(asyncio.run(run_pipeline(TEST_URL)))
