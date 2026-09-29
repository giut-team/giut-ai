import json


def load_text_blocks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["text"]


def load_text_labels(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {
            row["id"]: row["is_relevant"]
            for row in json.load(f)
            if row["id"].startswith("T")
        }
