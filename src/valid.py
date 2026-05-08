import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import CSV_DIR_TRAIN, IMG_DIR_TRAIN, BEST_MODEL, param_config, BATCH_SIZE
from sklearn.metrics import accuracy_score, confusion_matrix
from loader import get_dataloaders
from cnn import LeNet, EnhancedLeNet, CustomResNet


def test(model_path=BEST_MODEL):
    _, dataloader_val = get_dataloaders(batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})

    if config["model"].lower() == "lenet":
        model = LeNet().to(device)
    elif config["model"].lower() == "enhancedlenet":
        model = EnhancedLeNet().to(device)
    elif config["model"].lower() == "resnet":
        model = CustomResNet().to(device)
    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()
    test_loss = 0

    with torch.no_grad():
        for images, labels in dataloader_val:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {test_loss / len(dataloader_val):.4f}")
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
