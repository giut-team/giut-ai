import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.ml.image_classifier import ImageClassifier
from training.train_image_classifier import (
    BATCH_SIZE,
    DEVICE,
    IMAGE_EXTENSIONS,
    IMAGE_MODEL_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageDataset,
    evaluate,
)

IMAGES_DIR = Path("data/images/test")
IMAGE_LABELS_PATH = IMAGES_DIR / "image_cnn_labels.json"

NUM_WORKERS = 0


def load_samples():
    with IMAGE_LABELS_PATH.open(encoding="utf-8") as f:
        rows = json.load(f)

    labels = {row["id"]: int(row["is_poster"]) for row in rows}
    samples = []
    for image_path in sorted(IMAGES_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = labels.get(image_path.stem)
        if label is not None:
            samples.append((image_path, label))

    if not samples:
        raise ValueError(f"라벨과 매칭되는 이미지 없음: {IMAGES_DIR}")
    return samples


def main():
    if not IMAGE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"학습된 모델 없음: {IMAGE_MODEL_PATH} ")

    samples = load_samples()
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_loader = DataLoader(
        ImageDataset(samples, test_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    # 테스트 시 pretrained 가중치는 받지 않고 체크포인트 가중치 사용
    classifier = ImageClassifier(pretrained=False).to(DEVICE)
    state_dict = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
    classifier.model.load_state_dict(state_dict)

    metrics = evaluate(classifier, test_loader)
    print(f"device: {DEVICE}")
    print(f"test samples: {metrics['n']}")
    print(
        f"[test] acc={metrics['accuracy']:.4f} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} F2={metrics['f2']:.4f}"
    )
    print(
        f"confusion matrix: TP={metrics['tp']} FP={metrics['fp']} "
        f"TN={metrics['tn']} FN={metrics['fn']}"
    )
    print(
        f"[test] n_rel:n_irr={(metrics['tp']+metrics['fn'])/metrics['n']}:{(metrics['fp']+metrics['tn'])/metrics['n']} "
        f"pred_rel:pred_irr={(metrics['tp']+metrics['fp'])/metrics['n']}:{(metrics['tn']+metrics['fn'])/metrics['n']}"
    )
    print()


if __name__ == "__main__":
    main()
