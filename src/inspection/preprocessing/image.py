import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    return image


def resize_image(
    image: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value == min_value:
        return np.zeros_like(image)

    return (image - min_value) / (max_value - min_value)