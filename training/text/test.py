import torch
import joblib

from src.ml.text_classifier import TextClassifier
from training.common import evaluate
from training.config import *
from training.text.dataset import load_text_blocks, load_text_labels

if __name__ == "main":
    scaler = joblib.load(TEXT_SCALER_PATH)
    classifier = TextClassifier(n_num=len(NUM_KEYS)).to(DEVICE)
    classifier.head.load_state_dict(torch.load(TEXT_HEAD_PATH, map_location=DEVICE))

    test_labels = load_text_labels(TEST_LABELS_PATH)
    test_blocks = load_text_blocks(TEST_BLOCKS_PATH)
    test_blocks = [b for b in test_blocks if b["id"] in test_labels]

    pages: dict[int, list[dict]] = {}

    for b in test_blocks:
        pages.setdefault(b["page_id"], []).append(b)

    for page_id in sorted(pages):
        page_blocks = pages[page_id]
        m = evaluate(
            classifier, page_blocks, test_labels, scaler, batch_size=len(page_blocks)
        )
        print(
            f"[test] page={page_id} n={m['n']} acc={m['accuracy']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
        )
        print(f"[test] TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        print(
            f"[test] n_rel:n_irr={(m['tp']+m['fn'])/m['n']}:{(m['fp']+m['tn'])/m['n']} "
            f"pred_rel:pred_irr={(m['tp']+m['fp'])/m['n']}:{(m['tn']+m['fn'])/m['n']}"
        )
