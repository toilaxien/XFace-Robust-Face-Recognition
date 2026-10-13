import torch

from xface.models.heads.xface import XFaceHead


def test_xface_head_outputs_logits_and_weights():
    head = XFaceHead(embedding_dim=16, num_classes=5)
    embeddings = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    logits, weights = head(embeddings, labels, epoch=40)
    assert logits.shape == (4, 5)
    assert weights.shape == (4,)
    assert torch.isfinite(logits).all()
