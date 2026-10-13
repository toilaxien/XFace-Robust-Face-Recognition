from __future__ import annotations

import torch


def reliability_weights(
    z_quality: torch.Tensor,
    z_gap: torch.Tensor,
    quality_gate: float = 2.0,
    gap_gate: float = 2.5,
    max_down_weight: float = 0.10,
    rho: float = 1.0,
) -> torch.Tensor:
    """Down-weight samples whose quality and class-separation signals disagree."""

    conflict = torch.relu(-z_quality * z_gap)
    gated = torch.sigmoid(conflict / max(quality_gate * gap_gate, 1e-6))
    penalty = rho * max_down_weight * gated
    return (1.0 - penalty).clamp(1.0 - max_down_weight, 1.0)
