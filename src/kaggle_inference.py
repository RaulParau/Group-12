from config import (
    BEST_MODEL,
    PATH_SUBMISSION,
    IMG_DIR_TRAIN,
    CSV_DIR_TRAIN,
    BATCH_SIZE,
    IMG_DIR_INFERENCE,
)
from loader import get_inference_dataloader
from cnn import LeNet, EnhancedLeNet, CustomResNet
import torch
from pathlib import Path
import pandas as pd


def run_inference(model_path=BEST_MODEL, output_path=PATH_SUBMISSION):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})

    if config.get("model").lower() == "lenet":
        model = LeNet().to(device)
    elif config.get("model").lower() == "enhancedlenet":
        model = EnhancedLeNet().to(device)
    else:
        model = CustomResNet().to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = get_inference_dataloader()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, img_ids in dataloader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(img_ids.cpu().numpy())

    submission = pd.DataFrame({"Id": all_ids, "Category": all_preds})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    run_inference()
