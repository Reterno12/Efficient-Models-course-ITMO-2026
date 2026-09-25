"""The exact FP32 inference network used throughout Homework 1."""
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []
        convs = (
            (3, 32, 7, 2),
            (32, 64, 5, 1),
            (64, 128, 3, 2),
            (128, 256, 1, 1),
            (256, 256, 3, 2),
            (256, 512, 1, 1),
        )
        for index, (in_channels, out_channels, kernel, stride) in enumerate(convs):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel, stride,
                          padding=kernel // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if index == 0:
                layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        return self.head(self.features(x))
