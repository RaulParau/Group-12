from torch.utils.data import DataLoader, dataloader
from config import IMG_DIR_TRAIN, CSV_DIR_TRAIN, BATCH_SIZE
from dataset import DigitDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transforms import transform


def get_dataloaders(batch_size=BATCH_SIZE):
    df = pd.read_csv(CSV_DIR_TRAIN)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dataset = DigitDataset(
        train_df, IMG_DIR_TRAIN, transform(train=True, normalize=True, augment=True)
    )
    val_dataset = DigitDataset(val_df, IMG_DIR_TRAIN, transform(train=False))
    test_dataset = DigitDataset(test_df, IMG_DIR_TRAIN, transform(train=False))

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (dataloader_train, dataloader_val, dataloader_test)
