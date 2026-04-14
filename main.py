from src.dataset import DigitDataset
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/iivp-2026-challenge/train.csv")

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

IMG_DIR_TRAIN = "iivp-2026-challenge/train/train"

train_dataset = DigitDataset(train_df, IMG_DIR_TRAIN)
val_dataset = DigitDataset(val_df, IMG_DIR_TRAIN)
test_dataset = DigitDataset(test_df, IMG_DIR_TRAIN)

print(len(train_dataset))
