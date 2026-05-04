"""
Config file for reusable constants such as file paths
"""

CSV_DIR_TRAIN = "data/iivp-2026-challenge/train.csv"
CSV_DIR_INFERENCE = "data/iivp-2026-challenge/test.csv"
IMG_DIR_TRAIN = "data/iivp-2026-challenge/train/train"
IMG_DIR_INFERENCE = "data/iivp-2026-challenge/test/test"
BEST_MODEL = "models/best_model.pth"
PATH_SUBMISSION = "outputs/submission.csv"
BATCH_SIZE = 32


## Hyperparameters
param_config = {
    "model": "resnet",
    "optimizer": "AdamW",
    "lr": 3e-4,  # OneCycle max_lr
    "weight_decay": 1e-4,
    "epochs": 15,  # must be > 1 for scheduler to matter
    "label_smoothing": 0.05,
    "scheduler": "onecycle",
    "pct_start": 0.3,
    "div_factor": 25.0,
    "final_div_factor": 1e4,
}
