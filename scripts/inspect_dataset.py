from pathlib import Path

from inspection.data.mvtec import MVTecDataset


DATA_ROOT = Path("data/raw/mvtec_anomaly_detection")


def main():
    dataset = MVTecDataset(DATA_ROOT, "bottle")

    train_samples = dataset.get_train_samples()
    test_samples = dataset.get_test_samples()

    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples:     {len(test_samples)}")

    print("\nTraining distribution:")

    train_counts = {}

    for sample in train_samples:
        train_counts[sample.defect_type] = (
            train_counts.get(sample.defect_type, 0) + 1
        )

    for defect_type, count in train_counts.items():
        print(f"  {defect_type}: {count}")

    print("\nTest distribution:")

    test_counts = {}

    for sample in test_samples:
        test_counts[sample.defect_type] = (
            test_counts.get(sample.defect_type, 0) + 1
        )

    for defect_type, count in test_counts.items():
        print(f"  {defect_type}: {count}")


if __name__ == "__main__":
    main()