from __future__ import annotations

import torch

from xface.models.heads.arcface import ArcFaceHead


class HAMFaceHead(ArcFaceHead):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, base_margin: float = 0.40, hardness_alpha: float = 0.05):
        super().__init__(embedding_dim, num_classes, scale=scale, margin=base_margin)
        self.base_margin = base_margin
        self.hardness_alpha = hardness_alpha

    def hardness_margin(self, target_cosine: torch.Tensor) -> torch.Tensor:
        return (self.base_margin + self.hardness_alpha * (1.0 - target_cosine)).clamp(0.30, 0.55)
