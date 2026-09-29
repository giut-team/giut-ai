import os
import torch

TEXT_MODELS_DIR = "models/text_classifier"
TEXT_SCALER_PATH = os.path.join(TEXT_MODELS_DIR, "num_scaler.joblib")
TEXT_HEAD_PATH = os.path.join(TEXT_MODELS_DIR, "text_classifier_head.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2IDX = {"IRRELEVANT": 0, "RELEVANT": 1}
NUM_KEYS = ["link_density", "rel_pos"]

TRAIN_BLOCKS_PATH = "data/train/train_blocks.json"
TRAIN_LABELS_PATH = "data/train/train_labels.json"

TEST_BLOCKS_PATH = "data/test/test_blocks.json"
TEST_LABELS_PATH = "data/test/test_labels.json"
