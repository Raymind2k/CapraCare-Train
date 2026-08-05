import torch
import torch.nn as nn


class DLAB(nn.Module):
    """
    Disease Lesion Attention Block v2

    Components:
    1. Multi-scale feature extraction
    2. Channel attention
    3. Spatial attention
    4. Residual feature enhancement
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        # -----------------------
        # Multi-scale branch
        # -----------------------
        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.conv5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            bias=False
        )

        self.bn = nn.BatchNorm2d(channels)

        # -----------------------
        # Channel Attention
        # -----------------------
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # -----------------------
        # Spatial Attention
        # -----------------------
        self.spatial = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=7,
                padding=3
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        residual = x

        # ======================
        # Multi-scale features
        # ======================
        f3 = self.conv3(x)

        f5 = self.conv5(x)

        x = f3 + f5

        x = self.bn(x)

        # ======================
        # Channel Attention
        # ======================
        b, c, _, _ = x.size()

        ca = self.avgpool(x).view(b, c)

        ca = self.channel_fc(ca)

        ca = ca.view(b, c, 1, 1)

        x = x * ca

        # ======================
        # Spatial Attention
        # ======================
        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        sa = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        sa = self.spatial(sa)

        x = x * sa

        # ======================
        # Residual Enhancement
        # ======================
        x = x + residual

        return x