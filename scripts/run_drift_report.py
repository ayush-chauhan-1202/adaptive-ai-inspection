"""Compare recent inspection scores against the training-time baseline (Milestone 9).

The question this answers: "does the anomaly-score distribution the model is
seeing in production still look like what it saw during training/validation?"
If not, that's the signal that the model may need re-evaluation or
retraining (Milestone 10), independent of whether its accuracy metrics have
been directly measured on new labeled data yet - this is the thing you can
check *before* you have new labels.

Inputs are plain CSVs with a `score` column, by design:
  --reference-csv  the validation scores from training time. train_and_
                    register.py doesn't currently dump these to a file, so
                    for now, save them yourself during a training run
                    (`validation_scores` in scripts/train_and_register.py's
                    `train()`), or recompute them with the memory bank.
  --current-csv     recent production scores. In a real deployment these
                    come from Cloud Logging (every /inspect call logs a
                    structured "inspect_request" line with a "score" field -
                    see RUNBOOK.md for the `gcloud logging read` + jq
                    one-liner that turns a day of those into this CSV).

This is intentionally a script you run (by hand or from a scheduled CI job),
not a background service - see Milestone 9's scope note in the project roadmap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_SHARE_ALERT_THRESHOLD = 0.5  # alert if >=50% of monitored columns show drift


def run_drift_report(reference_csv: Path, current_csv: Path, output_html: Path) -> bool:
    """Returns True if drift was detected above the alert threshold."""

    reference = pd.read_csv(reference_csv)
    current = pd.read_csv(current_csv)

    if "score" not in reference.columns or "score" not in current.columns:
        raise ValueError("Both CSVs must have a 'score' column.")

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference[["score"]], current_data=current[["score"]])

    output_html.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output_html))

    drifted_share = 0.0
    for metric in result.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            drifted_share = metric["value"]["share"]
            break

    return drifted_share >= DRIFT_SHARE_ALERT_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--current-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/drift_reports/latest_report.html"),
    )
    args = parser.parse_args()

    drift_detected = run_drift_report(args.reference_csv, args.current_csv, args.output)

    print(f"Drift report written to {args.output}")

    if drift_detected:
        print("DRIFT DETECTED: current score distribution differs from reference.")
        sys.exit(1)

    print("No significant drift detected.")


if __name__ == "__main__":
    main()
