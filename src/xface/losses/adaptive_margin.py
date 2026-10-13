from __future__ import annotations

import torch


class RunningStandardizer(torch.nn.Module):
    """EMA standardizer used for quality and cosine-gap signals."""

    def __init__(self, momentum: float = 0.99, eps: float = 1e-6, clip: float = 2.5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        batch_mean = values.mean()
        batch_var = values.var(unbiased=False).clamp_min(self.eps)
        if not bool(self.initialized.item()):
            self.mean.copy_(batch_mean.reshape_as(self.mean))
            self.var.copy_(batch_var.reshape_as(self.var))
            self.initialized.fill_(True)
            return
        self.mean.mul_(self.momentum).add_(batch_mean * (1.0 - self.momentum))
        self.var.mul_(self.momentum).add_(batch_var * (1.0 - self.momentum))

    def forward(self, values: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.training and update:
            self.update(values)
        z = (values - self.mean) / torch.sqrt(self.var + self.eps)
        return z.clamp(-self.clip, self.clip)


def ramp_factor(epoch: int | None, warmup: int, ramp: int) -> float:
    if epoch is None:
        return 1.0
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return max(0.0, min(1.0, (epoch - warmup) / float(ramp)))


def adaptive_angular_margin(
    z_quality: torch.Tensor,
    z_gap: torch.Tensor,
    base_margin: float = 0.40,
    quality_alpha: float = 0.025,
    gap_alpha: float = 0.025,
    min_margin: float = 0.34,
    max_margin: float = 0.46,
    rho: float = 1.0,
) -> torch.Tensor:
    margin = base_margin + rho * (
        quality_alpha * torch.tanh(z_quality) + gap_alpha * torch.tanh(z_gap)
    )
    return margin.clamp(min_margin, max_margin)
