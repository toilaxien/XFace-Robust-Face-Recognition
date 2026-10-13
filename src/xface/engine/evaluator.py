from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_embeddings(model, loader, device, normalize: bool = True):
    model.eval()
    embeddings = []
    labels = []
    for images, target in loader:
        images = images.to(device)
        emb = model(images, normalize=normalize)
        embeddings.append(emb.cpu())
        labels.append(target)
    return torch.cat(embeddings), torch.cat(labels)


@torch.no_grad()
def classification_accuracy(model, head, loader, device) -> float:
    model.eval()
    head.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        embeddings = model(images)
        output = head(embeddings)
        logits = output[0] if isinstance(output, tuple) else output
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
    return correct / max(total, 1)
