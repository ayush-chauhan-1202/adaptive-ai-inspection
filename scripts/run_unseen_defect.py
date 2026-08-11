from pathlib import Path

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

UNSEEN_DEFECT = "broken_small"


def get_defect_type(sample):
    """
    MVTec test images are stored as:

    bottle/test/<defect_type>/<image>.png

    Therefore the parent directory identifies
    the defect type.
    """
    return Path(sample.image_path).parent.name


def main():

    dataset = MVTecDataset(
        DATA_ROOT,
        "bottle",
    )

    train_samples = dataset.get_train_samples()
    test_samples = dataset.get_test_samples()

    # Training remains NORMAL ONLY.
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

    print(
        f"Training samples:   {len(train_samples)}"
    )

    print(
        f"Validation samples: {len(validation_samples)}"
    )

    print(
        f"Test samples:       {len(test_samples)}"
    )

    print(
        f"\nHeld-out defect: {UNSEEN_DEFECT}"
    )

    # --------------------------------------------------
    # Feature extractor
    # --------------------------------------------------

    extractor = ResNetFeatureExtractor()

    print(
        f"\nUsing device: {extractor.device}"
    )

    # --------------------------------------------------
    # Build memory bank using NORMAL images only
    # --------------------------------------------------

    print(
        "\nBuilding normal patch memory bank..."
    )

    memory_bank = build_memory_bank(
        train_samples,
        extractor,
    )

    print(
        f"Memory bank size: "
        f"{len(memory_bank.features)} patches"
    )

    # --------------------------------------------------
    # Threshold selection
    # --------------------------------------------------

    print(
        "\nScoring normal validation samples..."
    )

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

    # --------------------------------------------------
    # Score ALL test images
    # --------------------------------------------------

    print(
        "\nScoring test samples..."
    )

    test_results = []

    for sample in test_samples:

        _, _, score = compute_anomaly_map(
            sample,
            extractor,
            memory_bank,
        )

        test_results.append(
            {
                "sample": sample,
                "score": score,
                "defect_type": get_defect_type(sample),
            }
        )

    # --------------------------------------------------
    # Overall results
    # --------------------------------------------------

    labels = np.asarray(
        [
            result["sample"].label
            for result in test_results
        ],
        dtype=np.int32,
    )

    scores = np.asarray(
        [
            result["score"]
            for result in test_results
        ]
    )

    overall_metrics = evaluate(
        labels,
        scores,
        threshold,
    )

    print(
        "\nOverall Test Results"
    )

    print("=" * 40)

    for name, value in overall_metrics.items():

        print(
            f"{name:25s}: {value:.4f}"
        )

    # --------------------------------------------------
    # Per-defect results
    # --------------------------------------------------

    defect_types = sorted(
        set(
            result["defect_type"]
            for result in test_results
            if result["sample"].label == 1
        )
    )

    print(
        "\nPer-Defect Detection"
    )

    print("=" * 60)

    for defect_type in defect_types:

        defect_results = [
            result
            for result in test_results
            if result["defect_type"] == defect_type
        ]

        defect_scores = np.asarray(
            [
                result["score"]
                for result in defect_results
            ]
        )

        detected = (
            defect_scores >= threshold
        )

        detection_rate = detected.mean()

        print(
            f"{defect_type:20s}: "
            f"{detected.sum():2d}/"
            f"{len(detected):2d} "
            f"({detection_rate:.4f})"
        )

    # --------------------------------------------------
    # Explicit unseen-defect result
    # --------------------------------------------------

    unseen_results = [
        result
        for result in test_results
        if result["defect_type"] == UNSEEN_DEFECT
    ]

    unseen_scores = np.asarray(
        [
            result["score"]
            for result in unseen_results
        ]
    )

    unseen_detected = (
        unseen_scores >= threshold
    )

    unseen_detection_rate = (
        unseen_detected.mean()
    )

    print(
        "\nUNSEEN DEFECT RESULT"
    )

    print("=" * 40)

    print(
        f"Defect type:        {UNSEEN_DEFECT}"
    )

    print(
        f"Samples:             "
        f"{len(unseen_scores)}"
    )

    print(
        f"Detected:            "
        f"{unseen_detected.sum()}"
    )

    print(
        f"Detection rate:      "
        f"{unseen_detection_rate:.4f}"
    )

    print(
        f"Minimum score:       "
        f"{unseen_scores.min():.4f}"
    )

    print(
        f"Maximum score:       "
        f"{unseen_scores.max():.4f}"
    )

    print(
        f"Threshold:            "
        f"{threshold:.4f}"
    )


if __name__ == "__main__":
    main()