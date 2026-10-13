from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xface.losses.adaptive_margin import RunningStandardizer
from xface.models.heads.arcface import ArcFaceHead


class AdaFaceHead(ArcFaceHead):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, base_margin: float = 0.40, h: float = 0.333):
        super().__init__(embedding_dim, num_classes, scale=scale, margin=base_margin)
        self.h = h
        self.standardizer = RunningStandardizer()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            return super().forward(embeddings, labels)
        norms = embeddings.norm(dim=1)
        z = self.standardizer(norms)
        self.margin = float((0.40 + self.h * 0.05 * torch.tanh(z).mean()).clamp(0.30, 0.50))
        return super().forward(embeddings, labels)
