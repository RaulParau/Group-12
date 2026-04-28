import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import CSV_DIR_TRAIN, IMG_DIR_TRAIN, BEST_MODEL
from sklearn.metrics import accuracy_score, confusion_matrix
from loader import get_dataloaders
from cnn import LeNet, EnhancedLeNet


def test(model_path=BEST_MODEL):
    _, _, dataloader_test = get_dataloaders(batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})

    if config.get("model", "enhanced").lower() == "lenet":
        model = LeNet().to(device)
    else:
        model = EnhancedLeNet().to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()
    test_loss = 0

    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {test_loss / len(dataloader_test):.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()
    plt.xticks(range(cm.shape[0]))
    plt.yticks(range(cm.shape[0]))

    # annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


if __name__ == "__main__":
    test()
