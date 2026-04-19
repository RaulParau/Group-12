from cnn import LeNet
from config import IMG_DIR_TRAIN, CSV_DIR_TRAIN
from loader import get_dataloaders
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
