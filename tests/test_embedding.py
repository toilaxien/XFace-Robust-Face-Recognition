import torch

from xface.models.embedding import FaceEmbeddingModel


def test_embedding_model_small_backbone():
    model = FaceEmbeddingModel(backbone="small", embedding_dim=32, dropout=0.0, pretrained=False)
    model.eval()
    images = torch.randn(2, 3, 112, 112)
    embeddings = model(images, normalize=True)
    assert embeddings.shape == (2, 32)
    norms = embeddings.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
