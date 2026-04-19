from cnn import LeNet
from config import IMG_DIR_TRAIN, CSV_DIR_TRAIN
from loader import get_dataloaders
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataloader_train, dataloader_val, dataloader_test = get_dataloaders(batch_size=16)

for epoch in range(10):
    for images, labels in dataloader_train:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.8f}")
