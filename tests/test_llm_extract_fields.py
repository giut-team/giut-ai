import json
from datetime import datetime

from app.models.dto import ExtractedFields
from app.models.field_extractor import (
    load_page_text,
    extract_fields_from_text,
)


async def test_extract_fields():
    with open("html_data/test/test_blocks.json", "r") as f_blocks:
        blocks = json.load(f_blocks)["text"]

    page_blocks = {}

    page_id = 0
    for b in blocks:
        if b["page_id"] > page_id:
            page_id = b["page_id"]
            # break
        if b["page_id"] == page_id:
            page_blocks.setdefault(page_id, []).append(b)

    with open("html_data/test/llm_answer_labels.json", "r") as f_labels:
        labels = json.load(f_labels)[: len(blocks)]

        page_labels = {}
        for l in labels:
            page_labels[l["page_id"]] = l

    with open(
        f"tests/results/llm_results_{datetime.now()}.json".replace(":", "_"),
        "w",
    ) as f_result:
        for id, blocks in page_blocks.items():
            page_text = load_page_text(blocks)

            started_at = datetime.now()
            fields_result = await extract_fields_from_text(page_text)
            ended_at = datetime.now()

            print(fields_result.format())
            print("elapsed_time:", ended_at - started_at)

            result_block = {
                "page_id": id,
                "elapsed_time": str(ended_at - started_at),
                "response": ExtractedFields.to_dict(fields_result),
                "comparison": fields_result.compare(
                    ExtractedFields.from_dict(page_labels[id])
                ),
            }

            add_data = json.dumps(result_block, ensure_ascii=False)
            f_result.write(add_data)
