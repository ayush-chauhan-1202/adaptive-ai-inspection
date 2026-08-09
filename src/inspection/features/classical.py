import cv2
import numpy as np


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract simple handcrafted features from an image."""

    image = image.astype(np.float32)

    features = [
        image.mean(),
        image.std(),
        np.percentile(image, 5),
        np.percentile(image, 25),
        np.percentile(image, 50),
        np.percentile(image, 75),
        np.percentile(image, 95),
    ]

    # Gradient magnitude statistics
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)

    features.extend(
        [
            magnitude.mean(),
            magnitude.std(),
            np.percentile(magnitude, 75),
            np.percentile(magnitude, 95),
        ]
    )

    return np.asarray(features, dtype=np.float32)