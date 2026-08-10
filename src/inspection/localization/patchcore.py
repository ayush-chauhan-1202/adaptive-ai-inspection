from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)


class ResNetFeatureExtractor:
    """Extract spatial feature maps from a pretrained ResNet-18."""

    def __init__(self, device: str | None = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

        # Keep layers through layer3.
        self.model = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )

        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def extract(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)

        return self.model(batch)


def get_transform():
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")

    return get_transform()(image)


def feature_map_to_patches(
    feature_map: torch.Tensor,
) -> torch.Tensor:
    """
    Convert B x C x H x W feature maps into
    B x H x W x C patch embeddings.
    """

    return feature_map.permute(
        0,
        2,
        3,
        1,
    )


class PatchMemoryBank:

    def __init__(self):
        self.features = None

    def fit(
        self,
        embeddings: np.ndarray,
    ):
        self.features = embeddings.astype(
            np.float32
        )

    def score(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:

        if self.features is None:
            raise RuntimeError(
                "Memory bank has not been fitted."
            )

        # Euclidean distance.
        distances = np.linalg.norm(
            embeddings[:, None, :]
            - self.features[None, :, :],
            axis=2,
        )

        # Nearest normal patch.
        return distances.min(axis=1)


def build_memory_bank(
    samples,
    extractor: ResNetFeatureExtractor,
) -> PatchMemoryBank:

    all_embeddings = []

    for sample in samples:
        image = load_image(
            Path(sample.image_path)
        )

        feature_map = extractor.extract(
            image.unsqueeze(0)
        )

        patches = feature_map_to_patches(
            feature_map
        )

        patches = patches.squeeze(0)

        patches = patches.reshape(
            -1,
            patches.shape[-1],
        )

        all_embeddings.append(
            patches.cpu().numpy()
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    memory_bank = PatchMemoryBank()
    memory_bank.fit(embeddings)

    return memory_bank



def compute_anomaly_map(
    sample,
    extractor: ResNetFeatureExtractor,
    memory_bank: PatchMemoryBank,
):
    image = load_image(
        Path(sample.image_path)
    )

    feature_map = extractor.extract(
        image.unsqueeze(0)
    )

    patches = feature_map_to_patches(
        feature_map
    )

    _, height, width, channels = patches.shape

    patches = patches.squeeze(0)

    patches = patches.reshape(
        -1,
        channels,
    )

    patch_scores = memory_bank.score(
        patches.cpu().numpy()
    )

    anomaly_map = patch_scores.reshape(
        height,
        width,
    )

    # Resize feature-map anomaly map to image dimensions.
    anomaly_map = cv2.resize(
        anomaly_map,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    image_score = float(
        anomaly_map.max()
    )

    return image, anomaly_map, image_score