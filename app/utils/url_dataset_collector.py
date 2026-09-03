import json

from app.parsers.html_parser import HtmlParser
from app.utils.url_list import TRAIN_URLS, TEST_URLS

if __name__ == "__main__":
    htmlParser = HtmlParser()

    with open("urls/train.json", "w", encoding="utf-8") as f:
        text = []
        images = []
        links = []

        for label, urls in TRAIN_URLS.items():
            for url in urls:
                try:
                    parsed = htmlParser.parse_url(url)
                    for t in parsed.text:
                        text.append(
                            {
                                **t,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["rel_pos"] >= 0.1
                                    and t["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )
                    for i in parsed.images:
                        images.append(
                            {
                                **i,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["rel_pos"] >= 0.1
                                    and t["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )
                    for l in parsed.links:
                        links.append(
                            {
                                **l,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["rel_pos"] >= 0.1
                                    and t["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                except Exception as e:
                    print(f"[skip] {url} -> {type(e).__name__}: {e}")

        f.write(
            json.dumps(
                {"text": text, "images": images, "links": links}, ensure_ascii=False
            )
        )

    with open("urls/test.json", "w", encoding="utf-8") as f:
        text = []
        images = []
        links = []

        for label, urls in TEST_URLS.items():
            for url in urls:
                try:
                    parsed = htmlParser.parse_url(url)
                    for t in parsed.text:
                        text.append(
                            {
                                **t,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["rel_pos"] >= 0.1
                                    and t["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )
                    for i in parsed.images:
                        images.append(
                            {
                                **i,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["rel_pos"] >= 0.1
                                    and t["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )
                    for l in parsed.links:
                        links.append(
                            {
                                **l,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["rel_pos"] >= 0.1
                                    and t["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                except Exception as e:
                    print(f"[skip] {url} -> {type(e).__name__}: {e}")

        f.write(
            json.dumps(
                {"text": text, "images": images, "links": links}, ensure_ascii=False
            )
        )
