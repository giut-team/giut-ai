import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.ml.image_classifier import ImageClassifier

IMAGES_DIR = Path("data/images/train")
IMAGE_LABELS_PATH = IMAGES_DIR / "image_cnn_labels.json"
IMAGE_MODELS_DIR = Path("models/image_classifier")
IMAGE_MODEL_PATH = IMAGE_MODELS_DIR / "image_classifier.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 20
VAL_RATIO = 0.1
SEED = 42
THRESHOLD = 0.5
NUM_WORKERS = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


@torch.no_grad()
def evaluate(classifier, data_loader, threshold=THRESHOLD):
    # 이미지 validation set의 accuracy, precision, recall, F1, F2, 혼동행렬 계산
    was_training = classifier.training
    classifier.eval()

    tp = fp = tn = fn = 0
    for images, target in data_loader:
        images = images.to(DEVICE)
        target = target.to(DEVICE).long()

        logits = classifier.model(images).squeeze(1)
        pred = (torch.sigmoid(logits) >= threshold).long()

        tp += int(((pred == 1) & (target == 1)).sum().item())
        fp += int(((pred == 1) & (target == 0)).sum().item())
        tn += int(((pred == 0) & (target == 0)).sum().item())
        fn += int(((pred == 0) & (target == 1)).sum().item())

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
        raise ValueError(f"라벨과 매칭되는 이미지가 없습니다: {IMAGES_DIR}")
    return samples


def split_samples(samples):
    # 정답 클래스별로 나눠 train/val의 클래스 비율 최대한 유지
    rng = random.Random(SEED)
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    train_samples, val_samples = [], []
    for class_samples in by_label.values():
        rng.shuffle(class_samples)
        n_val = int(len(class_samples) * VAL_RATIO)
        if len(class_samples) > 1 and VAL_RATIO > 0:
            n_val = max(1, min(n_val, len(class_samples) - 1))
        val_samples.extend(class_samples[:n_val])
        train_samples.extend(class_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    if not train_samples or not val_samples:
        raise ValueError(
            "train과 val 모두 이미지가 있어야 합니다. 이미지 수와 VAL_RATIO를 확인하세요."
        )
    return train_samples, val_samples


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    samples = load_samples()
    train_samples, val_samples = split_samples(samples)
    print(f"train={len(train_samples)} val={len(val_samples)}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_loader = DataLoader(
        ImageDataset(train_samples, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        ImageDataset(val_samples, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    classifier = ImageClassifier().to(DEVICE)
    print(f"device: {DEVICE}")

    IMAGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        running_loss = 0.0

        for step, (images, target) in enumerate(train_loader, start=1):
            images = images.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            running_loss += classifier(images, target)

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{EPOCHS} batch {step}/{len(train_loader)} "
                    f"loss {running_loss / step:.4f}"
                )

        metrics = evaluate(classifier, val_loader)
        print(
            f"[val] epoch {epoch} acc={metrics['accuracy']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} F2={metrics['f2']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(classifier.model.state_dict(), IMAGE_MODEL_PATH)
            print(f"[saved] {IMAGE_MODEL_PATH} (best val F1={best_f1:.4f})")
        print()


if __name__ == "__main__":
    main()
