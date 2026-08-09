from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from inspection.visualization.inspection import plot_inspection_result

from inspection.data.mvtec import MVTecDataset
from inspection.evaluation.metrics import evaluate
from inspection.features.classical import extract_features
from inspection.models.anomaly import ClassicalAnomalyDetector
from inspection.preprocessing.image import load_grayscale, normalize_image


DATA_ROOT = Path("data/raw/mvtec_anomaly_detection")


def build_features(samples):
    features = []

    for sample in samples:
        image = load_grayscale(str(sample.image_path))
        image = normalize_image(image)

        features.append(extract_features(image))

    return np.stack(features)


def main():
    dataset = MVTecDataset(DATA_ROOT, "bottle")

    train_samples = dataset.get_train_samples()
    test_samples = dataset.get_test_samples()

    # MVTec training images should be normal.
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

    print("\nBuilding training features...")
    train_features = build_features(train_samples)

    print("Building validation features...")
    validation_features = build_features(validation_samples)

    print("Building test features...")
    test_features = build_features(test_samples)

    detector = ClassicalAnomalyDetector(
        contamination=0.05,
    )

    print("\nFitting anomaly detector...")
    detector.fit(train_features)

    validation_scores = detector.score(validation_features)
    test_scores = detector.score(test_features)

    # Select threshold using only normal validation data.
    threshold = np.percentile(validation_scores, 95)

    print(f"\nSelected threshold: {threshold:.6f}")

    labels = np.array(
        [sample.label for sample in test_samples],
        dtype=np.int32,
    )

    metrics = evaluate(
        labels,
        test_scores,
        threshold,
    )

    # Save an example inspection result.
    example_index = int(np.argmax(test_scores))
    example_sample = test_samples[example_index]

    image = load_grayscale(str(example_sample.image_path))

    prediction = (
        "ANOMALY"
        if test_scores[example_index] >= threshold
        else "NORMAL"
    )

    ground_truth = (
        "ANOMALY"
        if example_sample.label == 1
        else "NORMAL"
    )

    plot_inspection_result(
        image=image,
        score=test_scores[example_index],
        threshold=threshold,
        prediction=prediction,
        ground_truth=ground_truth,
        output_path="experiments/milestone_1/example_result.png",
    )

    print("\nFinal Test Results")
    print("=" * 40)

    for name, value in metrics.items():
        print(f"{name:25s}: {value:.4f}")


if __name__ == "__main__":
    main()