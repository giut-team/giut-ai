import json

relevant_blocks = []

with open("data/train/train_labels.json", "r") as f1:
    labels: list = json.load(f1)

with open("data/train/train_blocks.json", "r") as f2:
    blocks = json.load(f2)["text"]


for idx in range(len(blocks)):
    if labels[idx]["id"].startswith("T") and labels[idx]["is_relevant"] == "RELEVANT":
        relevant_blocks.append(blocks[idx])

print(idx, labels[idx]["id"])

with open("data/train_block_data_1.json", "w") as f3:
    json.dump(relevant_blocks, f3, ensure_ascii=False)
