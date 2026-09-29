from sklearn.preprocessing import StandardScaler

from src.ml.text_classifier import tokenize
from training.config import *


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

    target = torch.tensor([LABEL2IDX[labels[b["id"]]] for b in blocks])

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
