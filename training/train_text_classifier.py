import json
import os
import random
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler
import torch

from src.ml.text_classifier import TextClassifier, tokenize

TEXT_MODELS_DIR = "models/text_classifier"
TEXT_SCALER_PATH = os.path.join(TEXT_MODELS_DIR, "num_scaler.joblib")
TEXT_HEAD_PATH = os.path.join(TEXT_MODELS_DIR, "text_classifier_head.pt")
TRAIN_BLOCKS_PATH = Path("data/train/train_blocks.json")
TRAIN_LABELS_PATH = Path("data/train/train_labels.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_KEYS = ["link_density", "rel_pos"]

BATCH_SIZE = 64
EPOCHS = 20
VAL_RATIO = 0.1
SEED = 42
THRESHOLD = 0.3  # irr/rel 판정 기준값


def load_samples(
    blocks_path: str = TRAIN_BLOCKS_PATH,
    labels_path: str = TRAIN_LABELS_PATH,
):
    with open(blocks_path, encoding="utf-8") as f_blocks:
        blocks = json.load(f_blocks)["text"]

    with open(labels_path, encoding="utf-8") as f_labels:
        labels = {
            row["id"]: row["is_relevant"]
            for row in json.load(f_labels)
            if row["id"].startswith("T")
        }

    blocks = [b for b in blocks[:] if b["id"] in labels.keys()]

    return blocks, labels


def fit_num_scaler(blocks: list[dict]) -> StandardScaler:
    # 전체 학습셋 기준으로 수치 피처 스케일러를 한 번만 fit
    feats = [[b["num_features"][k] for k in NUM_KEYS] for b in blocks]
    return StandardScaler().fit(feats)


def make_batch(
    blocks: list[dict],
    labels: dict[str, str],
    scaler: StandardScaler,
    device: torch.device = DEVICE,
):
    enc = tokenize(blocks)

    raw_num = [[b["num_features"][k] for k in NUM_KEYS] for b in blocks]
    num = torch.tensor(scaler.transform(raw_num), dtype=torch.float)

    target = torch.tensor([int(labels[b["id"]]) for b in blocks])

    return (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        num.to(device),
        target.to(device),
    )


@torch.no_grad()
def evaluate(classifier, blocks, labels, scaler, batch_size=64, threshold=0.5):
    # RELEVANT(=1) 기준 accuracy / precision / recall / f1 + 혼동행렬
    # threshold: argmax 기본값 0.5 보다 작게 -> RELEVANT 판정 여유
    was_training = classifier.training
    classifier.eval()

    tp = fp = tn = fn = 0
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size]
        ids, mask, num, target = make_batch(batch, labels, scaler)

        prob = torch.softmax(classifier.logits(ids, mask, num), dim=-1)[:, 1]
        pred = (prob >= threshold).long()
        gold = target

        tp += int(((pred == 1) & (gold == 1)).sum())
        fp += int(((pred == 1) & (gold == 0)).sum())
        tn += int(((pred == 0) & (gold == 0)).sum())
        fn += int(((pred == 0) & (gold == 1)).sum())

    if was_training:
        classifier.train()

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if 4 * precision + recall
        else 0.0
    )
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    blocks, labels = load_samples()
    # train / val 분리
    random.seed(SEED)
    random.shuffle(blocks)
    n_val = int(len(blocks) * VAL_RATIO)
    val_blocks, train_blocks = blocks[:n_val], blocks[n_val:]
    print(f"train={len(train_blocks)} val={len(val_blocks)}")

    os.makedirs(TEXT_MODELS_DIR, exist_ok=True)

    scaler = fit_num_scaler(train_blocks)  # 학습 셋으로 한 번만 fit
    joblib.dump(scaler, TEXT_SCALER_PATH)
    print(f"[saved] {TEXT_SCALER_PATH}")

    classifier = TextClassifier(n_num=len(NUM_KEYS)).to(DEVICE)
    print(f"device: {DEVICE}\n")

    n_batches = (len(train_blocks) + BATCH_SIZE - 1) // BATCH_SIZE
    best_f2 = -1.0

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
            f"P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f} F2={v["f2"]:.4f}"
        )

        if v["f2"] > best_f2:
            best_f2 = v["f2"]
            torch.save(classifier.head.state_dict(), TEXT_HEAD_PATH)
            print(f"[saved] {TEXT_HEAD_PATH} (best val F2={best_f2:.4f})")
        print()


if __name__ == "__main__":
    main()
