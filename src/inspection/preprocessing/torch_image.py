from pathlib import Path

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def get_transform():
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def load_tensor(
    path: str | Path,
):
    image = Image.open(path).convert("RGB")

    transform = get_transform()

    return transform(image)