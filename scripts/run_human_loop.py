from pathlib import Path
import csv

import numpy as np
from sklearn.model_selection import train_test_split

from inspection.data.mvtec import MVTecDataset
from inspection.localization.patchcore import (
    ResNetFeatureExtractor,
    build_memory_bank,
    compute_anomaly_map,
)
from inspection.localization.triage import (
    classify_score,
)


DATA_ROOT = Path(
    "data/raw/mvtec_anomaly_detection"
)


def get_defect_type(sample):
    return Path(sample.image_path).parent.name


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

    print(
        f"Training samples:   {len(train_samples)}"
    )

    print(
        f"Validation samples: {len(validation_samples)}"
    )

    print(
        f"Test samples:       {len(test_samples)}"
    )

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    extractor = ResNetFeatureExtractor()

    print(
        f"\nUsing device: {extractor.device}"
    )

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
    # Normal validation scores
    # --------------------------------------------------

    print(
        "\nScoring validation samples..."
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

    review_threshold = np.percentile(
        validation_scores,
        90,
    )

    defect_threshold = np.percentile(
        validation_scores,
        95,
    )

    print(
        f"\nReview threshold: "
        f"{review_threshold:.6f}"
    )

    print(
        f"Defect threshold: "
        f"{defect_threshold:.6f}"
    )

    # --------------------------------------------------
    # Test scoring
    # --------------------------------------------------

    print("\nScoring test samples...")

    results = []

    for sample in test_samples:

        _, _, score = compute_anomaly_map(
            sample,
            extractor,
            memory_bank,
        )

        triage = classify_score(
            score,
            review_threshold,
            defect_threshold,
        )

        results.append(
            {
                "image_path": str(
                    sample.image_path
                ),
                "ground_truth": int(
                    sample.label
                ),
                "defect_type": get_defect_type(
                    sample
                ),
                "score": score,
                "decision": triage.decision,
            }
        )

    # --------------------------------------------------
    # Decision counts
    # --------------------------------------------------

    auto_normal = [
        r for r in results
        if r["decision"] == "AUTO_NORMAL"
    ]

    human_review = [
        r for r in results
        if r["decision"] == "HUMAN_REVIEW"
    ]

    auto_defect = [
        r for r in results
        if r["decision"] == "AUTO_DEFECT"
    ]

    print("\nInspection Triage")
    print("=" * 40)

    print(
        f"AUTO_NORMAL:   "
        f"{len(auto_normal)}"
    )

    print(
        f"HUMAN_REVIEW:  "
        f"{len(human_review)}"
    )

    print(
        f"AUTO_DEFECT:   "
        f"{len(auto_defect)}"
    )

    review_rate = (
        len(human_review) / len(results)
    )

    print(
        f"\nHuman review rate: "
        f"{review_rate:.4f}"
    )

    # --------------------------------------------------
    # Safety analysis
    # --------------------------------------------------

    missed_defects = [
        r for r in results
        if (
            r["ground_truth"] == 1
            and r["decision"] == "AUTO_NORMAL"
        )
    ]

    auto_detected_defects = [
        r for r in results
        if (
            r["ground_truth"] == 1
            and r["decision"] == "AUTO_DEFECT"
        )
    ]

    defects_sent_to_human = [
        r for r in results
        if (
            r["ground_truth"] == 1
            and r["decision"] == "HUMAN_REVIEW"
        )
    ]

    print("\nDefect Handling")
    print("=" * 40)

    print(
        f"Defects auto-detected: "
        f"{len(auto_detected_defects)}"
    )

    print(
        f"Defects sent to human: "
        f"{len(defects_sent_to_human)}"
    )

    print(
        f"Defects incorrectly auto-cleared: "
        f"{len(missed_defects)}"
    )

    # --------------------------------------------------
    # Save review queue
    # --------------------------------------------------

    review_path = Path(
        "experiments/milestone_5/"
        "human_review_queue.csv"
    )

    with review_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "score",
                "decision",
                "ground_truth",
                "defect_type",
            ],
        )

        writer.writeheader()

        for result in human_review:

            writer.writerow(
                {
                    "image_path":
                        result["image_path"],
                    "score":
                        result["score"],
                    "decision":
                        result["decision"],
                    "ground_truth":
                        result["ground_truth"],
                    "defect_type":
                        result["defect_type"],
                }
            )

    print(
        f"\nHuman review queue saved to:"
        f"\n{review_path}"
    )


if __name__ == "__main__":
    main()