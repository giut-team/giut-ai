import json
from src.llm.dto import CategoryEnum

RESULT_FILE_PATH = "tests/results_260929_8/llm_results_2026-09-29 14_30_39.494627.json"
NEW_FILE_PATH = "tests/results_260929_8/llm_results_2026-09-29 14_30_39.494627_1.json"
ANSWER_LABEL_PATH = "data/test/llm_answer_labels_1_category.json"

with open(ANSWER_LABEL_PATH, "r") as f_answer:
    answers = {
        a["page_id"]: (
            "None" if a["category"] == None else str(CategoryEnum(a["category"]))
        )
        for a in json.load(f_answer)
    }

with open(RESULT_FILE_PATH, "r") as f_result:
    results = json.load(f_result)

for r in results:
    response_before = r["response"]["category"]
    answer = str(answers[r["page_id"]])

    print(response_before, "/", answer)

    r["comparison"]["category"] = "EXACT" if response_before == answer else "MISS"

with open(NEW_FILE_PATH, "w") as f_new:
    json.dump(results, f_new, ensure_ascii=False)
