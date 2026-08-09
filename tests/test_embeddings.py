import torch

from inspection.models.encoder import ResNet18Encoder


def test_resnet_embedding_dimension():
    encoder = ResNet18Encoder(device="cpu")

    batch = torch.randn(2, 3, 224, 224)

    embeddings = encoder.encode(batch)

    assert embeddings.shape == (2, 512)