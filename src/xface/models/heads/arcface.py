from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.scale = scale
        self.margin = margin
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-7))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = F.one_hot(labels, cosine.size(1)).to(dtype=cosine.dtype, device=cosine.device)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
