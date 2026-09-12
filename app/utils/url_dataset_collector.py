import json

from app.parsers.html_parser import HtmlParser
from app.utils.url_list import TRAIN_URLS, TEST_URLS

if __name__ == "__main__":
    htmlParser = HtmlParser()

    train_labels = []
    test_labels = []

    with open("urls/train/train_blocks.json", "w", encoding="utf-8") as f:
        text = []
        images = []
        links = []

        text_labels = []
        image_labels = []
        link_labels = []

        page_id = 0

        for label, urls in TRAIN_URLS.items():
            for url in urls:
                try:
                    parsed = htmlParser.parse_url(url)

                    for t in parsed.text:
                        id = f"T{len(text)+1:06d}"
                        text.append({"id": id, "page_id": page_id, **t})
                        text_labels.append(
                            {
                                "id": id,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["features"]["rel_pos"] >= 0.1
                                    and t["features"]["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                    for i in parsed.images:
                        id = f"I{len(images)+1:06d}"
                        images.append({"id": id, "page_id": page_id, **i})
                        image_labels.append(
                            {
                                "id": id,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and i["features"]["rel_pos"] >= 0.1
                                    and i["features"]["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                    for l in parsed.links:
                        id = f"L{len(links)+1:06d}"
                        links.append({"id": id, "page_id": page_id, **l})
                        link_labels.append(
                            {
                                "id": id,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and l["features"]["rel_pos"] >= 0.1
                                    and l["features"]["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                except Exception as e:
                    print(f"[skip] {url} -> {type(e).__name__}: {e}")

                page_id += 1
                print(f"[ok] [{page_id}] {url}")

        f.write(
            json.dumps(
                {"text": text, "images": images, "links": links}, ensure_ascii=False
            )
        )

        train_labels = text_labels + image_labels + link_labels

    with open("urls/train/train_labels.json", "w") as f:
        f.write(json.dumps(train_labels, ensure_ascii=False))

    with open("urls/test/test_blocks.json", "w", encoding="utf-8") as f:
        text = []
        images = []
        links = []

        text_labels = []
        image_labels = []
        link_labels = []

        page_id = 0

        for label, urls in TEST_URLS.items():
            for url in urls:
                try:
                    parsed = htmlParser.parse_url(url)
                    for t in parsed.text:
                        id = f"T{len(text)+1:06d}"
                        text.append({"id": id, "page_id": page_id, **t})
                        text_labels.append(
                            {
                                "id": id,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and t["features"]["rel_pos"] >= 0.1
                                    and t["features"]["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                    for i in parsed.images:
                        id = f"I{len(images)+1:06d}"
                        images.append({"id": id, "page_id": page_id, **i})
                        image_labels.append(
                            {
                                "id": id,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and i["features"]["rel_pos"] >= 0.1
                                    and i["features"]["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                    for l in parsed.links:
                        id = f"L{len(links)+1:06d}"
                        links.append({"id": id, "page_id": page_id, **l})
                        link_labels.append(
                            {
                                "id": id,
                                "is_relevant": (
                                    "RELEVANT"
                                    if label == "COMPETITION"
                                    and l["features"]["rel_pos"] >= 0.1
                                    and l["features"]["rel_pos"] < 0.8
                                    else "IRRELEVANT"
                                ),
                            }
                        )

                except Exception as e:
                    print(f"[skip] {url} -> {type(e).__name__}: {e}")

                page_id += 1
                print(f"[ok] [{page_id}] {url}")

        f.write(
            json.dumps(
                {"text": text, "images": images, "links": links}, ensure_ascii=False
            )
        )

        test_labels = text_labels + image_labels + link_labels

    with open("urls/test/test_labels.json", "w") as f:
        f.write(json.dumps(test_labels, ensure_ascii=False))
