import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

WEIGHT = 5.37  # n_irr / n_rel


class ImageClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WEIGHT]))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def forward(self, images, target):
        pred = self.model(images).squeeze(1)
        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
