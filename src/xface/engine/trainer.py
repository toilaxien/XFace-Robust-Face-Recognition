from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None, label_smoothing: float = 0.0) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
    if weights is None:
        return losses.mean()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def train_one_epoch(model, head, loader, optimizer, device, epoch: int = 0, label_smoothing: float = 0.0):
    model.train()
    head.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(images)
        output = head(embeddings, labels, epoch=epoch) if head.__class__.__name__ == "XFaceHead" else head(embeddings, labels)
        logits, weights = output if isinstance(output, tuple) else (output, None)
        loss = weighted_cross_entropy(logits, labels, weights, label_smoothing)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return {"loss": total_loss / max(len(loader), 1)}
