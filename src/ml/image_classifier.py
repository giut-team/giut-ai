import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

WEIGHT = 5.37  # n_irr / n_rel
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageClassifier(nn.Module):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WEIGHT]))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    @classmethod
    def preprocess(cls, image: Image.Image) -> torch.Tensor:
        return cls.transform(image.convert("RGB"))

    def logits(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, (list, tuple)):
            images = torch.stack([self.preprocess(image) for image in images])
            images = images.to(next(self.model.parameters()).device)
        return self.model(images).squeeze(1)

    def forward(self, images, target):
        pred = self.logits(images)
        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
