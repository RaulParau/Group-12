from cProfile import label
from cnn import LeNet
from config import IMG_DIR_TRAIN, CSV_DIR_TRAIN
from loader import get_dataloaders
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train_losses = []
val_losses = []
val_accuracies = []


def train():
    dataloader_train, dataloader_val, _ = get_dataloaders(batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        train_loss = 0
        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_losses.append(train_loss / len(dataloader_train))
        val_losses.append(val_loss / len(dataloader_val))
        val_accuracies.append(correct / total)
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss / len(dataloader_train):.4f} | "
            f"Val Loss: {val_loss / len(dataloader_val):.4f} | "
            f"Val Acc: {correct / total:.4f}"
        )


if __name__ == "__main__":
    train()
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

