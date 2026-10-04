"""Champion/challenger evaluation: promote a new model only if it's better.

This is Milestone 10's core idea, deliberately split out from training
itself: train_and_register.py's job is to produce a candidate (always left
in 'staging' once a 'production' model already exists); this script's job is
to decide, on real held-out data, whether that candidate actually deserves to
replace the current champion.

This is intentionally a manual-trigger script, not a scheduled job. Fully
automatic retraining (triggered by accumulated human-review data or a drift
alert) is designed here in code but left manually invoked for now, because
automating it only makes sense once there's real production traffic feeding
real human-reviewed corrections back in - which this project, being a fresh
demo deployment, doesn't have yet. Wiring this to Milestone 9's drift alert
or a data-volume threshold later is a small follow-up, not a rewrite.

Usage:
    python scripts/retrain_and_promote.py \\
        --data-root data/raw/mvtec_anomaly_detection \\
        --challenger-version 3 \\
        --publish-gcs-path gs://my-bucket/production-model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from inspection.data.mvtec import MVTecDataset
from inspection.evaluation.metrics import evaluate
from inspection.serving.publish import publish_production_bundle

MODEL_NAME = "adaptive-ai-inspection"
PRIMARY_METRIC = "auprc"  # higher is better - matches the metric run_baseline.py and friends optimize for


def evaluate_model_version(model_uri: str, data_root: Path, category: str) -> dict[str, float]:
    """Score a registered model version against the real test set.

    Loaded as a generic pyfunc model (exactly how the API loads it), so this
    evaluates the same code path that will actually serve traffic - not a
    reconstructed copy of it.
    """

    model = mlflow.pyfunc.load_model(model_uri)

    dataset = MVTecDataset(data_root, category)
    test_samples = dataset.get_test_samples()

    scores = []
    labels = []

    for sample in test_samples:
        import pandas as pd

        result = model.predict(pd.DataFrame({"image_path": [str(sample.image_path)]}))[0]
        scores.append(result["score"])
        labels.append(sample.label)

    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.int32)

    # Use the median score as a threshold purely for computing precision/
    # recall/F1 in this comparison - the champion/challenger decision below
    # is made on auroc/auprc, which are threshold-independent.
    threshold = float(np.median(scores))

    return evaluate(labels, scores, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument(
        "--challenger-version",
        required=True,
        help="Model version currently aliased 'staging' (or any version) to evaluate.",
    )
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--publish-gcs-path",
        default=None,
        help="If the challenger wins and is promoted, also publish it here (see RUNBOOK.md).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Promote the challenger even if it doesn't beat the champion "
        "(e.g. for the first-ever manual promotion). Use sparingly.",
    )
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()

    challenger_uri = f"models:/{MODEL_NAME}/{args.challenger_version}"
    challenger_metrics = evaluate_model_version(challenger_uri, args.data_root, args.category)

    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        champion_uri = f"models:/{MODEL_NAME}/{champion_version.version}"
        champion_metrics = evaluate_model_version(champion_uri, args.data_root, args.category)
    except mlflow.exceptions.MlflowException:
        champion_metrics = None

    print(f"Challenger (v{args.challenger_version}) {PRIMARY_METRIC}: "
          f"{challenger_metrics[PRIMARY_METRIC]:.4f}")

    if champion_metrics is not None:
        print(f"Champion {PRIMARY_METRIC}: {champion_metrics[PRIMARY_METRIC]:.4f}")

    challenger_wins = (
        champion_metrics is None
        or challenger_metrics[PRIMARY_METRIC] > champion_metrics[PRIMARY_METRIC]
    )

    if challenger_wins or args.force:
        client.set_registered_model_alias(
            MODEL_NAME, "production", args.challenger_version
        )
        print(
            f"Promoted version {args.challenger_version} to 'production' "
            f"({'won on ' + PRIMARY_METRIC if challenger_wins else 'forced'})."
        )

        if args.publish_gcs_path:
            publish_production_bundle(MODEL_NAME, args.publish_gcs_path)
    else:
        print(
            f"Challenger did not beat the current champion on {PRIMARY_METRIC}; "
            "leaving 'production' unchanged."
        )


if __name__ == "__main__":
    main()
