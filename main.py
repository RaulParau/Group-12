from torch.utils.data import DataLoader, dataloader
from src.dataset import DigitDataset
from src.cnn import LeNet
from sklearn.model_selection import train_test_split
import pandas as pd
from src.config import IMG_DIR_TRAIN, CSV_DIR_TRAIN
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

df = pd.read_csv(CSV_DIR_TRAIN)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


train_dataset = DigitDataset(train_df, IMG_DIR_TRAIN)
val_dataset = DigitDataset(val_df, IMG_DIR_TRAIN)
test_dataset = DigitDataset(test_df, IMG_DIR_TRAIN)

dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=16, shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for images, labels in dataloader_train:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.8f}")
