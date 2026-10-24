import json
from datetime import datetime

import pytest

from src.pipeline import run_pipeline
from src.llm.dto import ExtractedFields
from src.llm.field_extractor import (
    extract_fields_from_images,
)
from tests.test_llm_extract_fields import make_result_block


async def test_extract_fields_from_images():
    image_paths = {}

    PAGE_ID = 0

    with open("data/test/html_blocks.json", "r") as f_blocks:
        blocks = json.load(f_blocks)["images"]

    with open("data/test/labels.json", "r") as f_rel:
        labels = json.load(f_rel)
        ids = [l["id"] for l in labels if l["is_relevant"]]

    for b in blocks:
        if b["id"] in ids:
            image_paths.setdefault(b["page_id"], []).append(b["src"])

    with open(
        "data/test/vlm_answer_labels_participants_double_enum.json", "r"
    ) as f_labels:
        labels = json.load(f_labels)

        page_labels = {}
        for l in labels:
            page_labels[l["page_id"]] = l

    result_blocks = []

    try:
        images = image_paths[PAGE_ID]

        started_at = datetime.now()
        fields_result, usage = await extract_fields_from_images(images)
        ended_at = datetime.now()
        elapsed_time = str(ended_at - started_at)

        print(fields_result.format())
        print("elapsed_time:", elapsed_time)

        comparison = fields_result.compare(ExtractedFields.from_dict(page_labels[id]))

        result = make_result_block(
            id=id,
            fields_result=fields_result.to_dict(),
            usage=usage,
            comparison=comparison,
            elapsed_time=elapsed_time,
        )
        result_blocks.append(result)

    finally:
        path = f"results/field_extractor/images/vlm_results_{str(started_at).replace(" ", "_").replace(":", "")}.json"
        with open(path, "w") as f_result:
            json.dump(result_blocks, f_result, ensure_ascii=False)


async def test_pipeline():
    from scripts.url_list import TEST_URLS

    URLS = []
    for v in TEST_URLS.values():
        URLS += v
    TEST_PAGE_ID = 6

    started_at = datetime.now()
    fields_result, usage = await run_pipeline(URLS[TEST_PAGE_ID])
    ended_at = datetime.now()
    elapsed_time = ended_at - started_at

    print()
    print(*(f"{k}: {v}\n" for k, v in fields_result.items()))
    print("elapsed_time:", elapsed_time)

    with open("data/text/test/vlm_answer_labels.json", "r") as f_labels:
        labels = json.load(f_labels)
    page_labels = {}
    for l in labels:
        page_labels[l["page_id"]] = l

    comparison = ExtractedFields.from_dict(fields_result).compare(
        ExtractedFields.from_dict(page_labels[TEST_PAGE_ID])
    )

    result = make_result_block(
        id=TEST_PAGE_ID,
        fields_result=fields_result,
        usage=usage,
        comparison=comparison,
        elapsed_time=elapsed_time,
    )

    path = f"results/pipeline/pipeline_results_{str(started_at).replace(" ", "_").replace(":", "")}.json"
    with open(path, "w") as f_result:
        json.dump(result, f_result, ensure_ascii=False)
