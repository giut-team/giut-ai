import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

MAX_LENGTH = 256
WEIGHT = 3.5  # n_irr / n_rel

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
encoder = AutoModel.from_pretrained("klue/roberta-base")


def tokenize(blocks: list[dict], max_length: int = MAX_LENGTH):
    # class_id [SEP] text 순서로 토큰화, 패딩은 배치 내 최대 길이에 맞춤
    class_ids = [b.get("class_id", "") for b in blocks]
    texts = [b.get("text", "") for b in blocks]

    return tokenizer(
        class_ids,
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


class TextClassifier(nn.Module):
    def __init__(self, tokenizer=tokenizer, encoder=encoder, n_num=2):
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

        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, WEIGHT]))
        self.optimizer = optim.Adam(self.head.parameters(), lr=1e-3)

        # 인코더 가중치는 고정, head만 학습
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()

    def logits(self, ids, mask, num):
        # 텍스트 인코딩 결과와 수치 피처를 붙여 함께 학습
        cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([cls, num], dim=-1))

    def forward(self, ids, mask, num, target):
        pred = self.logits(ids, mask, num)
        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
