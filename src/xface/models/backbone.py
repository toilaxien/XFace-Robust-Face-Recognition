from __future__ import annotations

import torch
from torch import nn


class SmallBackbone(nn.Module):
    """Compact fallback backbone used for tests and CPU smoke runs."""

    def __init__(self, out_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_features = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    if name == "small":
        return SmallBackbone()
    try:
        import timm

        model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        model.out_features = getattr(model, "num_features", 2048)
        return model
    except Exception:
        return SmallBackbone()
