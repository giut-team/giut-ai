import json
import os
import random

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 64
MAX_LENGTH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACT_DIR = "app/models/artifacts"
SCALER_PATH = os.path.join(ARTIFACT_DIR, "num_scaler.joblib")
HEAD_PATH = os.path.join(ARTIFACT_DIR, "relevant_block_head.pt")

NUM_KEYS = ["depth", "link_density", "n_links", "rel_pos"]
LABEL2IDX = {"IRRELEVANT": 0, "RELEVANT": 1}

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
encoder = AutoModel.from_pretrained("klue/roberta-base")


def load_blocks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["text"]  # + data["images"] + data["links"]


def load_labels(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {
            row["id"]: row["is_relevant"]
            for row in json.load(f)
            if row["id"].startswith("T")
        }


def fit_num_scaler(blocks: list[dict]) -> StandardScaler:
    # 전체 학습셋 기준으로 수치 피처 스케일러를 한 번만 fit
    feats = [[b["num_features"][k] for k in NUM_KEYS] for b in blocks]
    return StandardScaler().fit(feats)


def _pair_text(block: dict) -> str:
    # "text [SEP] src/href" (src/href 없으면 text만)
    seg = block.get("text", "")
    src_ref = block.get("src") or block.get("href")
    if src_ref:
        seg = f"{seg} {tokenizer.sep_token} {src_ref}"
    return seg


def tokenize(blocks: list[dict], max_length: int = MAX_LENGTH):
    # class_id [SEP] text ([SEP] src/href) 순서로 토큰화, 패딩은 배치 내 최대 길이에 맞춤
    class_ids = [b.get("class_id", "") for b in blocks]
    pair_texts = [_pair_text(b) for b in blocks]

    return tokenizer(
        class_ids,
        pair_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def make_batch(
    blocks: list[dict],
    labels: dict[str, str],
    scaler: StandardScaler,
    device: torch.device = DEVICE,
):
    enc = tokenize(blocks)

    raw_num = [[b["num_features"][k] for k in NUM_KEYS] for b in blocks]
    num = torch.tensor(scaler.transform(raw_num), dtype=torch.float)

    target = torch.tensor([LABEL2IDX[labels[b["id"]]] for b in blocks])
    # target = idx.float()

    return (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        num.to(device),
        target.to(device),
    )


@torch.no_grad()
def evaluate(classifier, blocks, labels, scaler, batch_size=BATCH_SIZE):
    # RELEVANT(=1) 기준 accuracy / precision / recall / f1 + 혼동행렬
    classifier.eval()

    tp = fp = tn = fn = 0
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size]
        ids, mask, num, target = make_batch(batch, labels, scaler)

        pred = classifier.logits(ids, mask, num).argmax(dim=-1)
        gold = target

        tp += int(((pred == 1) & (gold == 1)).sum())
        fp += int(((pred == 1) & (gold == 0)).sum())
        tn += int(((pred == 0) & (gold == 0)).sum())
        fn += int(((pred == 0) & (gold == 1)).sum())

    classifier.train()

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


class RelevantBlockClassifier(nn.Module):
    def __init__(self, tokenizer, encoder, n_num=4):
        super().__init__()

        self.tok = tokenizer
        self.enc = encoder
        h = encoder.config.hidden_size

        self.head = nn.Linear(h + n_num, 2)
        # self.head = nn.Sequential(
        #     nn.Linear(h + n_num, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 2),
        # )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.head.parameters(), lr=0.001)

        # 인코더는 고정하고 head만 학습
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()

    def logits(self, ids, mask, num):
        # 인코더는 고정이므로 항상 no_grad.
        with torch.no_grad():
            cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([cls, num], dim=-1))

    def forward(self, ids, mask, num, target):
        pred = self.logits(ids, mask, num)
        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


EPOCHS = 1  # 임시

VAL_RATIO = 0.1
SEED = 42

if __name__ == "__main__":
    labels = load_labels("urls/train/train_labels.json")
    blocks = load_blocks("urls/train/train_blocks.json")
    blocks = [b for b in blocks if b["id"] in labels]

    # train / val 분리 (test는 마지막에 한 번만 사용)
    random.seed(SEED)
    random.shuffle(blocks)
    n_val = int(len(blocks) * VAL_RATIO)
    val_blocks, train_blocks = blocks[:n_val], blocks[n_val:]
    print(f"train={len(train_blocks)} val={len(val_blocks)}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    scaler = fit_num_scaler(train_blocks)  # 학습 셋으로 한 번만 fit
    joblib.dump(scaler, SCALER_PATH)
    print(f"[saved] {SCALER_PATH}")

    classifier = RelevantBlockClassifier(tokenizer, encoder).to(DEVICE)
    print(f"device: {DEVICE}")

    n_batches = (len(train_blocks) + BATCH_SIZE - 1) // BATCH_SIZE
    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        random.shuffle(train_blocks)
        running = 0.0

        for step, start in enumerate(range(0, len(train_blocks), BATCH_SIZE), start=1):
            batch = train_blocks[start : start + BATCH_SIZE]
            ids, mask, num, target = make_batch(batch, labels, scaler)

            running += classifier(ids, mask, num, target)

            if step % 50 == 0 or step == n_batches:
                print(
                    f"epoch {epoch}/{EPOCHS} "
                    f"batch {step}/{n_batches} "
                    f"loss {running / step:.4f}"
                )

        v = evaluate(classifier, val_blocks, labels, scaler)
        print(
            f"[val] epoch {epoch} acc={v['accuracy']:.4f} "
            f"P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f}"
        )

        if v["f1"] > best_f1:
            best_f1 = v["f1"]
            torch.save(classifier.head.state_dict(), HEAD_PATH)
            print(f"[saved] {HEAD_PATH} (best val F1={best_f1:.4f})")

    # === test ===============================================
    classifier.head.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))

    test_labels = load_labels("urls/test/test_labels.json")
    test_blocks = load_blocks("urls/test/test_blocks.json")
    test_blocks = [b for b in test_blocks if b["id"] in test_labels]

    m = evaluate(classifier, test_blocks, test_labels, scaler)
    print(
        f"[test] n={m['n']} acc={m['accuracy']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
    )
    print(f"[test] TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
