import numpy as np

from inspection.features.classical import extract_features


def test_extract_features():
    image = np.random.default_rng(42).random((64, 64))

    features = extract_features(image)

    assert features.ndim == 1
    assert len(features) == 11
    assert np.isfinite(features).all()