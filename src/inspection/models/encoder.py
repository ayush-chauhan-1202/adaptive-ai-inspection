import torch
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Encoder:
    """Pretrained ResNet-18 used as a fixed image feature extractor."""

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

        # Remove the classification layer.
        self.model = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

        self.model.eval()
        self.model.to(self.device)

        self.weights = weights

    @torch.inference_mode()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)

        embeddings = self.model(batch)

        return embeddings.flatten(start_dim=1)

    @property
    def embedding_dimension(self) -> int:
        return 512