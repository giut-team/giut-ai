import json
from scripts.url_list import TEST_URLS

with open("data/test/test_blocks.json", "r") as tb:
    test_blocks = json.load(tb)["text"]

last_block_page_id = test_blocks[-1]["page_id"]

answer_labels = []

for id in range(last_block_page_id):
    if id < len(TEST_URLS["COMPETITION"]):
        answer_labels.append(
            {
                "page_id": id,
                "title": "",
                "host_organization": "",
                "target_participants": "",
                "application_start_at": "",
                "application_end_at": "",
                "prize": "",
                "activity_kind": "",
                "category": "",
                "is_team": True,
                "is_team_evidence": "",
            }
        )
    else:
        answer_labels.append(
            {
                "page_id": id,
                "title": None,
                "host_organization": None,
                "target_participants": None,
                "application_start_at": None,
                "application_end_at": None,
                "prize": None,
                "activity_kind": None,
                "category": None,
                "is_team": None,
                "is_team_evidence": None,
            }
        )

with open("data/test/llm_answer_labels.json", "w") as lbs:
    json.dump(answer_labels, lbs, ensure_ascii=False)
