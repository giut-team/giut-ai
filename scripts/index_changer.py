import json

campuspick_labels = []

ranges = [
    [23, 160],
    [181, 218],
    [239, 260],
    [279, 324],
    [343, 389],
    [1592, 1615],
    [1636, 1660],
    [1679, 1723],
]

mappings = [
    [23, 23],
    [181, 45],
    [239, 67],
    [279, 87],
    [343, 107],
    [1592, 1197],
    [1636, 1219],
    [1679, 1239],
]

# start_ids = [f"T{r[0]:06d}" for r in ranges]
start_ids = {f"T{m[0]:06d}": f"T{m[1]:06d}" for m in mappings}
end_ids = [f"T{r[1]:06d}" for r in ranges]

with open("data/test/test_blocks_br_removed.json", "r") as tb:
    blocks = (json.load(tb))["text"]

    check_range = False
    split_base = ""
    split_idx = 1

    for block in blocks:
        if block["id"] in start_ids.keys():
            check_range = True
            split_base = start_ids[block["id"]]
            split_idx = 1
        if block["id"] in end_ids:
            check_range = False

        if check_range:
            new_id = split_base + "-" + f"{split_idx:03d}"
            block["id"] = new_id
            campuspick_labels.append(block)
            split_idx += 1


with open("data/test/test_labels_br_split.json", "w") as tl:
    json.dump(campuspick_labels, tl, ensure_ascii=False)
