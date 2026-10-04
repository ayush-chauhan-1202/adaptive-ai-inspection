"""Train the PatchCore memory bank and register it in the MLflow Model Registry.

This reuses the same training logic as scripts/run_human_loop.py (build a
memory bank from normal training images, pick review/defect thresholds from
the validation-score percentiles) but adds the one thing that script never
needed: saving the result somewhere the API can find it again later.

Usage:
    # Real data (requires the MVTec AD "bottle" category under --data-root):
    python scripts/train_and_register.py --data-root data/raw/mvtec_anomaly_detection

    # Fast smoke test with synthetic data (see make_synthetic_dataset.py):
    python scripts/make_synthetic_dataset.py
    python scripts/train_and_register.py --data-root data/raw/synthetic_smoke_test

Promotion policy: the very first version of a model is promoted straight to
Production (there's nothing to compare it against, and the API needs *some*
model to serve). Every later run is left in Staging - scripts/retrain_and_
promote.py's job is to decide whether a new version actually beats the
current Production model before promoting it. Training and promotion being
separate steps is deliberate: it's what makes the champion/challenger pattern
in Milestone 10 possible.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.exceptions
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from inspection.data.mvtec import MVTecDataset
from inspection.evaluation.metrics import evaluate
from inspection.localization.patchcore import (
    ResNetFeatureExtractor,
    build_memory_bank,
    compute_anomaly_map,
)
from inspection.localization.triage import classify_score
from inspection.serving.publish import publish_production_bundle
from inspection.serving.pyfunc_model import InspectionModel, save_artifact_files

MODEL_NAME = "adaptive-ai-inspection"


def train(
    data_root: Path,
    category: str,
    review_pct: float,
    defect_pct: float,
    seed: int = 42,
    pretrained: bool = True,
):
    dataset = MVTecDataset(data_root, category)

    train_samples = dataset.get_train_samples()
    test_samples = dataset.get_test_samples()

    normal_samples = [s for s in train_samples if s.label == 0]

    train_samples, validation_samples = train_test_split(
        normal_samples, test_size=0.20, random_state=seed
    )

    extractor = ResNetFeatureExtractor(pretrained=pretrained)
    memory_bank = build_memory_bank(train_samples, extractor)

    validation_scores = np.asarray(
        [compute_anomaly_map(s, extractor, memory_bank)[2] for s in validation_samples]
    )

    review_threshold = float(np.percentile(validation_scores, review_pct))
    defect_threshold = float(np.percentile(validation_scores, defect_pct))

    test_scores = []
    labels = []
    decisions = []

    for sample in test_samples:
        _, _, score = compute_anomaly_map(sample, extractor, memory_bank)
        triage = classify_score(score, review_threshold, defect_threshold)

        test_scores.append(score)
        labels.append(sample.label)
        decisions.append(triage.decision)

    test_scores = np.asarray(test_scores)
    labels = np.asarray(labels, dtype=np.int32)

    metrics = evaluate(labels, test_scores, defect_threshold)

    human_review = sum(1 for d in decisions if d == "HUMAN_REVIEW")
    missed_defects = sum(
        1 for d, label in zip(decisions, labels) if d == "AUTO_NORMAL" and label == 1
    )

    metrics["human_review_rate"] = human_review / len(decisions) if decisions else 0.0
    metrics["missed_defect_count"] = float(missed_defects)

    params = {
        "category": category,
        "review_threshold_percentile": review_pct,
        "defect_threshold_percentile": defect_pct,
        "memory_bank_size": len(memory_bank.features),
        "n_train_samples": len(train_samples),
        "n_validation_samples": len(validation_samples),
        "n_test_samples": len(test_samples),
        "seed": seed,
    }

    metadata = {
        "category": category,
        "review_threshold": review_threshold,
        "defect_threshold": defect_threshold,
        "image_size": 224,
        "feature_layer": "resnet18.layer3",
        "pretrained_weights": pretrained,
    }

    return params, metrics, metadata, memory_bank.features


def register(
    params: dict,
    metrics: dict,
    metadata: dict,
    memory_bank_features: np.ndarray,
    publish_gcs_path: str | None = None,
) -> str:
    client = MlflowClient()

    with mlflow.start_run(run_name=f"train-{metadata['category']}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_paths = save_artifact_files(Path(tmp_dir), memory_bank_features, metadata)

            mlflow.pyfunc.log_model(
                name="model",
                python_model=InspectionModel(),
                artifacts=artifact_paths,
                pip_requirements=[
                    "numpy",
                    "opencv-python-headless",
                    "torch",
                    "torchvision",
                    "pillow",
                ],
                registered_model_name=MODEL_NAME,
            )

        run_id = run.info.run_id

    # Find the version that was just registered under this run.
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    new_version = next(v for v in versions if v.run_id == run_id)

    has_production_alias = any(
        alias == "production" for alias in (new_version.aliases or [])
    ) or _alias_exists(client, MODEL_NAME, "production")

    if not has_production_alias:
        client.set_registered_model_alias(MODEL_NAME, "production", new_version.version)
        print(f"No existing 'production' alias - promoted version {new_version.version} directly.")

        if publish_gcs_path:
            publish_production_bundle(MODEL_NAME, publish_gcs_path)
    else:
        client.set_registered_model_alias(MODEL_NAME, "staging", new_version.version)
        print(
            f"Registered version {new_version.version} with alias 'staging'. "
            "Run scripts/retrain_and_promote.py to compare it against the "
            "current 'production' model before promoting."
        )

    return new_version.version


def _alias_exists(client: MlflowClient, model_name: str, alias: str) -> bool:
    try:
        client.get_model_version_by_alias(model_name, alias)
        return True
    except mlflow.exceptions.MlflowException:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--review-percentile", type=float, default=90.0)
    parser.add_argument("--defect-percentile", type=float, default=95.0)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var, "
        "or a local ./mlruns directory if unset.",
    )
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help="Use randomly-initialized ResNet-18 weights instead of downloading "
        "ImageNet weights. Only for offline smoke tests (see "
        "make_synthetic_dataset.py) - never use this for a real training run.",
    )
    parser.add_argument(
        "--publish-gcs-path",
        default=None,
        help="If this run becomes the production model, also publish its "
        "artifacts to this gs:// path (see RUNBOOK.md) so the deployed API "
        "can load it without a live tracking server. Omit for local-only runs.",
    )
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    params, metrics, metadata, memory_bank_features = train(
        args.data_root,
        args.category,
        args.review_percentile,
        args.defect_percentile,
        pretrained=not args.no_pretrained_weights,
    )

    print("Training complete.")
    print(f"  review_threshold: {metadata['review_threshold']:.6f}")
    print(f"  defect_threshold: {metadata['defect_threshold']:.6f}")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    version = register(
        params, metrics, metadata, memory_bank_features, publish_gcs_path=args.publish_gcs_path
    )
    print(f"Registered as '{MODEL_NAME}' version {version}.")


if __name__ == "__main__":
    main()
