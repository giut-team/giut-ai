import json


def remove_domain(src: str) -> str:
    src = src.removeprefix("https://").removeprefix("http://")
    return src[src.find("/") :]


def mask_file_name(src: str) -> str:
    # /uploads/competition/xxxx.jpg 와 같은 형식에서 파일명 xxxx.jpg 앞 부분 모두, .jpg와 같은 확장자명 유지
    # -> /uploads/competition/[file_name].jpg 처럼 마스킹
    dirs = src[: src.rfind("/") + 1]
    extension = src[src.rfind(".") :]  #
    return dirs + "[file_name]" + extension


if __name__ == "__main__":
    with open("data/train/train_blocks.json", "r") as f_blocks:
        image_blocks = json.load(f_blocks)["images"]

    masked_blocks = []
    src_labels = []

    for b in image_blocks:
        src: str = b.get("src", None)
        if src is None or len(src.strip()) == 0:
            print("src 없음")
            continue

        domain_removed = remove_domain(src)
        # masked = mask_file_name(domain_removed)

        masked_blocks.append({"id": b["id"], "src": domain_removed})

        src_labels.append(
            {
                "id": b["id"],
                "is_relevant": (
                    1
                    if domain_removed.find("upload") != -1
                    or domain_removed.find("admincenter") != -1
                    or domain_removed.find("cdn") != -1
                    or domain_removed.find("eDM") != -1
                    or domain_removed.find("attachment") != -1
                    or domain_removed.find("se2editor") != -1
                    or domain_removed.find("filegubun=poster") != -1
                    else 0
                ),
            }
        )

    # with open("data/train/train_image_src.json", "w") as f_masked:
    #     json.dump(masked_blocks, f_masked, ensure_ascii=False)

    with open("data/train/train_image_src_labels.json", "w") as f_labels:
        json.dump(src_labels, f_labels, ensure_ascii=False)
