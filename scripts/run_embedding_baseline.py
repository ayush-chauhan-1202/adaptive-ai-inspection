from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from inspection.data.mvtec import MVTecDataset
from inspection.evaluation.metrics import evaluate
from inspection.features.embeddings import extract_embeddings
from inspection.models.anomaly import ClassicalAnomalyDetector
from inspection.models.encoder import ResNet18Encoder


DATA_ROOT = Path("data/raw/mvtec_anomaly_detection")


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

    train_samples, validation_samples = train_test_split(
        normal_samples,
        test_size=0.20,
        random_state=42,
    )

    print(f"Training samples:   {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")
    print(f"Test samples:       {len(test_samples)}")

    encoder = ResNet18Encoder()

    print(f"\nUsing device: {encoder.device}")
    print(
        f"Embedding dimension: "
        f"{encoder.embedding_dimension}"
    )

    print("\nExtracting training embeddings...")
    train_embeddings = extract_embeddings(
        train_samples,
        encoder,
    )

    print("Extracting validation embeddings...")
    validation_embeddings = extract_embeddings(
        validation_samples,
        encoder,
    )

    print("Extracting test embeddings...")
    test_embeddings = extract_embeddings(
        test_samples,
        encoder,
    )

    print(
        f"\nTraining embedding shape: "
        f"{train_embeddings.shape}"
    )

    detector = ClassicalAnomalyDetector(
        contamination=0.05,
    )

    print("\nFitting anomaly detector...")

    detector.fit(train_embeddings)

    validation_scores = detector.score(
        validation_embeddings
    )

    test_scores = detector.score(
        test_embeddings
    )

    # Threshold selected only from normal validation data.
    threshold = np.percentile(
        validation_scores,
        95,
    )

    labels = np.array(
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

    print(
        f"\nSelected threshold: "
        f"{threshold:.6f}"
    )

    print("\nMilestone 2 Results")
    print("=" * 40)

    for name, value in metrics.items():
        print(
            f"{name:25s}: {value:.4f}"
        )


if __name__ == "__main__":
    main()