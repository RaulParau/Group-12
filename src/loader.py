from torch.utils.data import DataLoader, dataloader
from config import IMG_DIR_TRAIN, CSV_DIR_TRAIN
from dataset import DigitDataset
from sklearn.model_selection import train_test_split
import pandas as pd


def get_dataloaders(batch_size=16):
    df = pd.read_csv(CSV_DIR_TRAIN)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dataset = DigitDataset(train_df, IMG_DIR_TRAIN)
    val_dataset = DigitDataset(val_df, IMG_DIR_TRAIN)
    test_dataset = DigitDataset(test_df, IMG_DIR_TRAIN)
    dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=16, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return (dataloader_train, dataloader_val, dataloader_test)
