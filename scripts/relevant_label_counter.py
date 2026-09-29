import json

with open("data/train/train_labels.json", "r") as f_train:
    train_labels = [
        lb["is_relevant"] for lb in json.load(f_train) if lb["id"].startswith("T")
    ]

rel_cnt = train_labels.count("RELEVANT")
irr_cnt = train_labels.count("IRRELEVANT")

print(len(train_labels))
print(irr_cnt, "/", rel_cnt)
print("weight =", irr_cnt / rel_cnt)

with open("data/test/test_labels.json", "r") as f_test:
    test_labels = [
        lb["is_relevant"] for lb in json.load(f_test) if lb["id"].startswith("T")
    ]

rel_cnt = test_labels.count("RELEVANT")
irr_cnt = test_labels.count("IRRELEVANT")

print(len(test_labels))
print(irr_cnt, "/", rel_cnt)
print("weight =", irr_cnt / rel_cnt)
