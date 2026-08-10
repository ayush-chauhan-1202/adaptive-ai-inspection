from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from inspection.data.mvtec import MVTecDataset
from inspection.evaluation.metrics import evaluate
from inspection.localization.patchcore import (
    ResNetFeatureExtractor,
    build_memory_bank,
    compute_anomaly_map,
)


DATA_ROOT = Path(
    "data/raw/mvtec_anomaly_detection"
)


def main():

    dataset = MVTecDataset(
        DATA_ROOT,
        "bottle",
    )

    train_samples = dataset.get_train_samples()
    test_samples = dataset.get_test_samples()

    normal_samples = [
        sample
        for sample in train_samples
        if sample.label == 0
    ]

    train_samples, validation_samples = (
        train_test_split(
            normal_samples,
            test_size=0.20,
            random_state=42,
        )
    )

    extractor = ResNetFeatureExtractor()

    print(
        f"Using device: {extractor.device}"
    )

    print("\nBuilding patch memory bank...")

    memory_bank = build_memory_bank(
        train_samples,
        extractor,
    )

    print(
        f"Memory bank size: "
        f"{len(memory_bank.features)} patches"
    )

    print("\nScoring validation samples...")

    validation_scores = []

    for sample in validation_samples:

        _, _, score = compute_anomaly_map(
            sample,
            extractor,
            memory_bank,
        )

        validation_scores.append(score)

    validation_scores = np.asarray(
        validation_scores
    )

    threshold = np.percentile(
        validation_scores,
        95,
    )

    print(
        f"Selected threshold: "
        f"{threshold:.6f}"
    )

    print("\nScoring test samples...")

    test_scores = []

    example = None

    for sample in test_samples:

        image, anomaly_map, score = (
            compute_anomaly_map(
                sample,
                extractor,
                memory_bank,
            )
        )

        test_scores.append(score)

        if example is None and sample.label == 1:
            example = (
                image,
                anomaly_map,
                score,
                sample,
            )

    test_scores = np.asarray(
        test_scores
    )

    labels = np.asarray(
        [
            sample.label
            for sample in test_samples
        ],
        dtype=np.int32,
    )

    metrics = evaluate(
        labels,
        test_scores,
        threshold,
    )

    print("\nMilestone 3 Results")
    print("=" * 40)

    for name, value in metrics.items():
        print(
            f"{name:25s}: {value:.4f}"
        )

    if example is not None:

        image, anomaly_map, score, sample = (
            example
        )

        # Convert PyTorch CHW image to Matplotlib HWC format.

        image = image.permute(1, 2, 0).cpu().numpy()

        # Undo ImageNet normalization for visualization.

        mean = np.array(

            [0.485, 0.456, 0.406],

            dtype=np.float32,

        )

        std = np.array(

            [0.229, 0.224, 0.225],

            dtype=np.float32,

        )

        image = image * std + mean

        image = np.clip(image, 0.0, 1.0)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4),
        )

        axes[0].imshow(image)
        axes[0].set_title("Original")

        axes[1].imshow(
            anomaly_map,
            cmap="hot",
        )
        axes[1].set_title(
            f"Anomaly Map\nScore: {score:.3f}"
        )

        axes[2].imshow(image)
        axes[2].imshow(
            anomaly_map,
            cmap="hot",
            alpha=0.5,
        )
        axes[2].set_title(
            "Overlay"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        output_path = (
            "experiments/milestone_3/"
            "example_anomaly_map.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        print(
            f"\nSaved visualization: "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()