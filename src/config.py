"""
Config file for reusable constants such as file paths
"""

CSV_DIR_TRAIN = "data/iivp-2026-challenge/train.csv"
IMG_DIR_TRAIN = "data/iivp-2026-challenge/train/train"
IMG_DIR_INFERENCE = "data/iivp-2026-challenge/test/test"
BEST_MODEL = "models/best_model.pth"
BATCH_SIZE = 16


## Hyperparameters
param_config = {"model": "EnhancedLeNet", "optimizer": "AdamW", "lr": 1e-3}
