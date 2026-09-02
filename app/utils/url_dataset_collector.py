import json

from app.parsers.html_parser import HtmlParser
from app.utils.url_list import TRAIN_URLS, TEST_URLS

if __name__ == "__main__":
    htmlParser = HtmlParser()

    with open("urls/train.jsonl", "a", encoding="utf-8") as f:
        for label, urls in TRAIN_URLS.items():
            for url in urls:
                try:
                    parsed = htmlParser.parse_url(url)
                    record = {
                        "text": parsed.text,
                        "images": parsed.images,
                        "links": parsed.links,
                        "is_competition": label,
                        "is_team": label.replace("COMPETITION", "TEAM"),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[skip] {url} -> {type(e).__name__}: {e}")

    with open("urls/test.jsonl", "a", encoding="utf-8") as f:
        for label, urls in TEST_URLS.items():
            for url in urls:
                try:
                    parsed = htmlParser.parse_url(url)
                    record = {
                        "text": parsed.text,
                        "images": parsed.images,
                        "links": parsed.links,
                        "is_competition": label,
                        "is_team": label.replace("COMPETITION", "TEAM"),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[skip] {url} -> {type(e).__name__}: {e}")
