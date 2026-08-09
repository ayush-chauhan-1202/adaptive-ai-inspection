import numpy as np

from inspection.preprocessing.image import normalize_image


def test_normalize_image():
    image = np.array(
        [[0, 10], [20, 30]],
        dtype=np.uint8,
    )

    normalized = normalize_image(image)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0