import torch
import joblib
from pathlib import Path

from src.ml.text_classifier import TextClassifier
from training.train_text_classifier import (
    DEVICE,
    NUM_KEYS,
    TEXT_HEAD_PATH,
    TEXT_SCALER_PATH,
    evaluate,
    load_samples,
)

TEXT_DIR = Path("data/text/test")


def main():
    scaler = joblib.load(TEXT_SCALER_PATH)
    classifier = TextClassifier(n_num=len(NUM_KEYS)).to(DEVICE)
    classifier.head.load_state_dict(torch.load(TEXT_HEAD_PATH, map_location=DEVICE))

    blocks, labels = load_samples(TEXT_DIR)

    pages: dict[int, list[dict]] = {}

    for b in blocks:
        pages.setdefault(b["page_id"], []).append(b)

    totals = {name: 0 for name in ("tp", "fp", "tn", "fn")}

    for page_id in sorted(pages):
        page_blocks = pages[page_id]

        metrics = evaluate(
            classifier, page_blocks, labels, scaler, batch_size=len(page_blocks)
        )
        for name in totals:
            totals[name] += metrics[name]

        print(
            f"[test] page={page_id} n={metrics['n']} acc={metrics['accuracy']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )
        print(
            f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}"
        )
        print(
            f"n_rel:n_irr={(metrics['tp']+metrics['fn'])/metrics['n']}:{(metrics['fp']+metrics['tn'])/metrics['n']} "
            f"pred_rel:pred_irr={(metrics['tp']+metrics['fp'])/metrics['n']}:{(metrics['tn']+metrics['fn'])/metrics['n']}"
        )
        print()

    if pages:
        tp, fp, tn, fn = (totals[name] for name in ("tp", "fp", "tn", "fn"))
        n = tp + fp + tn + fn
        accuracy = (tp + tn) / n if n else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        print(
            f"[test] overall pages={len(pages)} n={n} "
            f"acc={accuracy:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
        )
        print(f"TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    main()
