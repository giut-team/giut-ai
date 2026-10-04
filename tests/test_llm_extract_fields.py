import json
from datetime import datetime
import pytest

from src.llm.dto import ExtractedFields
from src.llm.field_extractor import (
    load_page_text,
    extract_fields_from_text,
)
from src.pipeline import run_pipeline

MAX_PAGE_ID = 34  # 공모전 상세페이지만. 네거티브 모두 제거 (+ page id 19번 제거해야 함)
# MAX_PAGE_ID = 42  # 공모전 주최사 공홈까지만. 하드 네거티브 포함


@pytest.mark.skip(reason="결과 파일 저장용 헬퍼 함수")
def save_results(id, fields_result, usage, comparison, elapsed_time, path: str):
    with open(path, "w") as f_result:
        try:
            f_result.write("[")

            input_tokens = usage.prompt_tokens
            cached_input_tokens = usage.prompt_tokens_details.cached_tokens
            output_tokens = usage.completion_tokens

            result_block = {
                "page_id": id,
                "elapsed_time": str(elapsed_time),
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "cache_rate": cached_input_tokens / input_tokens,
                "output_tokens": output_tokens,
                "response": ExtractedFields.to_dict(fields_result),
                "comparison": comparison,
            }

            add_data = json.dumps(result_block, ensure_ascii=False)
            f_result.write(add_data + ",")

        finally:
            f_result.write("]")


async def test_extract_fields():
    with open("data/test/test_blocks.json", "r") as f_blocks:
        blocks = json.load(f_blocks)["text"]

    ### 노이즈 제거 코드
    # with open("data/test/test_labels.json", "r") as f_rel_labels:
    #     rel_labels = {l["id"]: l["is_relevant"] for l in json.load(f_rel_labels)}

    # for b in blocks[:]:
    #     # 인덱싱용으로 리스트를 복사해서 원본 리스트에서 삭제해도 인덱스가 밀리지 않게 함
    #     if rel_labels[b["id"]] != "RELEVANT":
    #         blocks.remove(b)
    ###

    page_blocks = {}

    page_id = 0
    for b in blocks:
        if b["page_id"] > page_id:
            page_id = b["page_id"]
        ### 네거티브 제거 코드
        if b["page_id"] == 19:
            continue
        if page_id > MAX_PAGE_ID:
            break
        ###
        if b["page_id"] == page_id:
            page_blocks.setdefault(page_id, []).append(b)

    with open("data/test/llm_answer_labels.json", "r") as f_labels:
        labels = json.load(f_labels)[: len(blocks)]

        page_labels = {}
        for l in labels:
            page_labels[l["page_id"]] = l

        for id, blocks in page_blocks.items():
            page_text = load_page_text(blocks)

        started_at = datetime.now()
        fields_result, usage = await extract_fields_from_text(page_text)
        ended_at = datetime.now()
        elapsed_time = str(ended_at - started_at)

        print("\n" + fields_result.format())
        print("elapsed_time:", elapsed_time)

        comparison = fields_result.compare(ExtractedFields.from_dict(page_labels[id]))

        save_results(
            id=id,
            fields_result=fields_result,
            usage=usage,
            comparison=comparison,
            elapsed_time=elapsed_time,
            path=f"results/field_extractor/text/llm_results_{str(started_at).replace(" ", "_").replace(":", "")}",
        )


async def test_pipeline():
    from scripts.url_list import TEST_URLS

    URLS = []
    for v in TEST_URLS.values():
        URLS += v
    TEST_PAGE_ID = 0

    started_at = datetime.now()
    fields_result, usage = await run_pipeline(URLS[TEST_PAGE_ID])
    ended_at = datetime.now()
    elapsed_time = ended_at - started_at

    print("\n" + fields_result.format())
    print("elapsed_time:", elapsed_time)

    with open("data/test/llm_answer_labels.json", "r") as f_labels:
        labels = json.load(f_labels)
    page_labels = {}
    for l in labels:
        page_labels[l["page_id"]] = l

    comparison = fields_result.compare(
        ExtractedFields.from_dict(page_labels[TEST_PAGE_ID])
    )

    save_results(
        id=TEST_PAGE_ID,
        fields_result=fields_result,
        usage=usage,
        comparison=comparison,
        elapsed_time=elapsed_time,
        path=f"results/pipeline/text/pipeline_results_{str(started_at).replace(" ", "_").replace(":", "")}.json",
    )
