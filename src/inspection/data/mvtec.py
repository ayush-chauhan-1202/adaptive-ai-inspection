from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class InspectionSample:
    image_path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None


class MVTecDataset:
    """Loader for a single MVTec AD category."""

    def __init__(self, root: str | Path, category: str):
        self.root = Path(root)
        self.category = category
        self.category_root = self.root / category

        if not self.category_root.exists():
            raise FileNotFoundError(
                f"Category directory does not exist: {self.category_root}"
            )

    def get_train_samples(self) -> list[InspectionSample]:
        train_root = self.category_root / "train"
        samples = []

        for defect_dir in sorted(train_root.iterdir()):
            if not defect_dir.is_dir():
                continue

            label = 0 if defect_dir.name == "good" else 1

            for image_path in sorted(defect_dir.glob("*.png")):
                samples.append(
                    InspectionSample(
                        image_path=image_path,
                        label=label,
                        defect_type=defect_dir.name,
                    )
                )

        return samples

    def get_test_samples(self) -> list[InspectionSample]:
        test_root = self.category_root / "test"
        ground_truth_root = self.category_root / "ground_truth"

        samples = []

        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue

            label = 0 if defect_dir.name == "good" else 1

            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = None

                if label == 1:
                    mask_candidate = (
                        ground_truth_root
                        / defect_dir.name
                        / f"{image_path.stem}_mask.png"
                    )

                    if mask_candidate.exists():
                        mask_path = mask_candidate

                samples.append(
                    InspectionSample(
                        image_path=image_path,
                        label=label,
                        defect_type=defect_dir.name,
                        mask_path=mask_path,
                    )
                )

        return samples

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")