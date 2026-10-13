from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xface.losses.adaptive_margin import RunningStandardizer, adaptive_angular_margin, ramp_factor
from xface.losses.reliability_weighting import reliability_weights


class XFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        base_margin: float = 0.40,
        min_margin: float = 0.34,
        max_margin: float = 0.46,
        quality_alpha: float = 0.025,
        gap_alpha: float = 0.025,
        ema_momentum: float = 0.99,
        z_clip: float = 2.5,
        quality_gate: float = 2.0,
        gap_gate: float = 2.5,
        max_down_weight: float = 0.10,
        margin_warmup: int = 8,
        reliability_warmup: int = 20,
        adaptation_ramp: int = 25,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.base_margin = base_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.quality_alpha = quality_alpha
        self.gap_alpha = gap_alpha
        self.quality_gate = quality_gate
        self.gap_gate = gap_gate
        self.max_down_weight = max_down_weight
        self.margin_warmup = margin_warmup
        self.reliability_warmup = reliability_warmup
        self.adaptation_ramp = adaptation_ramp
        self.quality_stats = RunningStandardizer(ema_momentum, clip=z_clip)
        self.gap_stats = RunningStandardizer(ema_momentum, clip=z_clip)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale, None

        batch = torch.arange(cosine.size(0), device=cosine.device)
        target = cosine[batch, labels]
        masked = cosine.masked_fill(F.one_hot(labels, cosine.size(1)).bool(), -2.0)
        gap = target - masked.max(dim=1).values
        quality = embeddings.norm(dim=1)

        z_quality = self.quality_stats(quality)
        z_gap = self.gap_stats(gap)
        margin_rho = ramp_factor(epoch, self.margin_warmup, self.adaptation_ramp)
        margin = adaptive_angular_margin(
            z_quality,
            z_gap,
            self.base_margin,
            self.quality_alpha,
            self.gap_alpha,
            self.min_margin,
            self.max_margin,
            margin_rho,
        )

        sine = torch.sqrt((1.0 - target.pow(2)).clamp_min(1e-7))
        phi_target = target * torch.cos(margin) - sine * torch.sin(margin)
        logits = cosine.clone()
        logits[batch, labels] = phi_target

        weight_rho = ramp_factor(epoch, self.reliability_warmup, self.adaptation_ramp)
        sample_weights = reliability_weights(
            z_quality,
            z_gap,
            self.quality_gate,
            self.gap_gate,
            self.max_down_weight,
            weight_rho,
        )
        return logits * self.scale, sample_weights
