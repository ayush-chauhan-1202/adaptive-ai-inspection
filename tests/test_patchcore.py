import numpy as np
import torch

from inspection.localization.patchcore import (
    PatchMemoryBank,
    feature_map_to_patches,
)


def test_feature_map_to_patches():
    feature_map = torch.randn(2, 256, 14, 14)

    patches = feature_map_to_patches(feature_map)

    assert patches.shape == (2, 14, 14, 256)


def test_memory_bank():
    rng = np.random.default_rng(42)

    normal_features = rng.random((10, 8))
    test_features = rng.random((3, 8))

    memory_bank = PatchMemoryBank()
    memory_bank.fit(normal_features)

    scores = memory_bank.score(test_features)

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()