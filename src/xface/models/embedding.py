from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xface.models.backbone import build_backbone


class FaceEmbeddingModel(nn.Module):
    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 512, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained)
        in_features = getattr(self.backbone, "out_features", getattr(self.backbone, "num_features", 512))
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.embedding(features)
        return F.normalize(embeddings) if normalize else embeddings
