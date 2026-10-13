from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CosFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.scale = scale
        self.margin = margin
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        one_hot = F.one_hot(labels, cosine.size(1)).to(dtype=cosine.dtype, device=cosine.device)
        return (cosine - one_hot * self.margin) * self.scale
