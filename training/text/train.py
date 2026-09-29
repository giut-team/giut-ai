import os
import joblib
import random
import torch

from src.ml.text_classifier import TextClassifier
from training.common import evaluate, fit_num_scaler, make_batch
from training.text.dataset import load_text_blocks, load_text_labels
from training.config import *

BATCH_SIZE = 64
THRESHOLD = 0.3  # irr/rel 판정 기준값
EPOCHS = 20

VAL_RATIO = 0.1
SEED = 42


if __name__ == "__main__":
    labels = load_text_labels(TRAIN_LABELS_PATH)
    blocks = load_text_blocks(TRAIN_BLOCKS_PATH)
    blocks = [b for b in blocks[:] if b["id"] in labels]

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
    print(f"device: {DEVICE}")

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
