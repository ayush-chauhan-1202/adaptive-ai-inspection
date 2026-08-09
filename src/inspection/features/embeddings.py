from pathlib import Path

import numpy as np
import torch

from inspection.models.encoder import ResNet18Encoder
from inspection.preprocessing.torch_image import load_tensor


def extract_embeddings(
    samples,
    encoder: ResNet18Encoder,
    batch_size: int = 32,
) -> np.ndarray:

    tensors = [
        load_tensor(Path(sample.image_path))
        for sample in samples
    ]

    embeddings = []

    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(
            tensors[start:start + batch_size]
        )

        batch_embeddings = encoder.encode(batch)

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

    return np.concatenate(embeddings, axis=0)