from cnn import LeNet, EnhancedLeNet
from config import IMG_DIR_TRAIN, CSV_DIR_TRAIN, BEST_MODEL, param_config
from loader import get_dataloaders
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_training(train_losses, val_losses, val_accuracies):
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


def train(config, verbose=False):
    dataloader_train, dataloader_val, _ = get_dataloaders(batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    if config["model"].lower() == "lenet":
        model = LeNet().to(device)
    else:
        model = EnhancedLeNet().to(device)

    if config["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_acc = 0
    best_model_state = None

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

        acc = correct / total
        if acc > best_acc:
            best_acc = correct / total
            best_model_state = model.state_dict()

        train_losses.append(train_loss / len(dataloader_train))
        val_losses.append(val_loss / len(dataloader_val))
        val_accuracies.append(correct / total)
        if verbose:
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss / len(dataloader_train):.4f} | "
                f"Val Loss: {val_loss / len(dataloader_val):.4f} | "
                f"Val Acc: {correct / total:.4f}"
            )
    return best_model_state, train_losses, val_losses, val_accuracies


if __name__ == "__main__":
    best_model_state, train_losses, val_losses, val_accuracies = train(
        param_config, verbose=True
    )

    save_path = Path(BEST_MODEL)
    if save_path.exists():
        checkpoint = torch.load(save_path)
        prev_best_acc = checkpoint.get("val_acc", 0)
    else:
        prev_best_acc = 0

    if best_model_state is not None and max(val_accuracies) > prev_best_acc:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_model_state,
                "val_acc": max(val_accuracies),
                "config": param_config,
            },
            save_path,
        )

    plot_training(train_losses, val_losses, val_accuracies)
