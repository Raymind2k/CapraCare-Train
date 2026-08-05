import torch
import torch.nn as nn
from torchvision import models

from lesion_attention import DLAB

class LABNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        self.features = base.features

        self.attention = DLAB(1280)

        self.pool = base.avgpool

        self.classifier = nn.Linear(
            1280,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)

        x = self.attention(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x