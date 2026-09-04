import torch
import torch.nn as nn
from transformers import AutoModel

encoder = AutoModel.from_pretrained("klue/roberta-base")


class RelevantBlockClassifier(nn.Module):
    def __init__(self, encoder, n_num=8):
        super().__init__()
        self.enc = encoder
        h = encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h + n_num, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, ids, mask, num):
        cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([cls, num], dim=-1))


if __name__ == "__main__":
    classifier = RelevantBlockClassifier(encoder)
    # "class_id [SEP] text ([SEP] src/href) [SEP] 수치 피처" 로 인코더에 넣고 학습
