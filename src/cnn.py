import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EnhancedLeNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(EnhancedLeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Block 1: Conv -> BN -> ReLU -> MaxPool -> Dropout
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Dropout2d(p=0.25),
            # Block 2: Conv -> BN -> ReLU -> MaxPool -> Dropout
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Dropout2d(p=0.25),
            # Block 3: Conv -> BN -> ReLU (no pool, preserve spatial info)
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
        )

        # 128 channels * 8 * 8 spatial = 8192
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
