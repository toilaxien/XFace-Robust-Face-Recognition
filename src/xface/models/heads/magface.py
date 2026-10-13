from __future__ import annotations

import torch

from xface.models.heads.arcface import ArcFaceHead


class MagFaceHead(ArcFaceHead):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, min_margin: float = 0.45, max_margin: float = 0.80, min_norm: float = 10.0, max_norm: float = 110.0):
        super().__init__(embedding_dim, num_classes, scale=scale, margin=min_margin)
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.min_norm = min_norm
        self.max_norm = max_norm

    def margin_from_norm(self, norms: torch.Tensor) -> torch.Tensor:
        q = ((norms - self.min_norm) / (self.max_norm - self.min_norm)).clamp(0.0, 1.0)
        return self.min_margin + q * (self.max_margin - self.min_margin)
