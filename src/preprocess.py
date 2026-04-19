import pandas as pd
import matplotlib.pyplot as plt
from config import CSV_DIR_TRAIN, IMG_DIR_TRAIN
from dataset import DigitDataset
import cv2

df = pd.read_csv(CSV_DIR_TRAIN)
dataset = DigitDataset(df, IMG_DIR_TRAIN)

img = cv2.imread(f"{IMG_DIR_TRAIN}/1/9673.png")
print(img.shape)  # returns (32, 32, 3)
# so we shouldn't resize the image down, just convert to grayscale in the dataset class

# for i in range(5):
#     img, label = dataset[i]
#     plt.imshow(img.squeeze(0), cmap="gray")
#     plt.title(f"label: {label}")
#     plt.show()
