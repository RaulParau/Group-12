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
        img_id = row["Id"]
        label = int(row["Category"])

        img_path = f"{self.img_dir}/{label}/{img_id}.png"

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(img_path)

        image = image.astype("float32") / 255.0

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
