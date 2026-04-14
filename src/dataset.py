import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2


class DigitDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = f"{self.img_dir}/{row[1]}/{row[0]}.png"
        label = int(row[1])

        # read the image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # resize the image to 28x28
        image = cv2.resize(image, (28, 28))

        # normalize to 0-1
        image = image.astype("float32") / 255.0

        # convert to tensor (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0)

        return image, label
