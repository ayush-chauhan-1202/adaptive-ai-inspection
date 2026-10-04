"""Version a new batch of human-reviewed corrections with DVC.

This is the data-versioning half of Milestone 10. The triage layer
(src/inspection/localization/triage.py) already routes uncertain predictions
to HUMAN_REVIEW; scripts/run_human_loop.py already writes those out to a CSV
queue. What was missing is: once a person has looked at that queue and
corrected/confirmed labels, how do those corrected examples get folded back
into something retrainable, with a record of exactly which batch was used
for which model version?

This script is deliberately thin - it just wraps `dvc add` + tells you the
`git add`/`git commit` to run - because the actual review/correction step is
a human looking at images, which isn't something to automate away. Running
it after each reviewed batch is what gives Milestone 10's retraining
pipeline a reproducible data version to point at.

Usage:
    python scripts/add_human_review_batch.py --batch-dir data/human_review/2026-10-04
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Directory holding this batch's corrected images + labels "
        "(e.g. data/human_review/2026-10-04/).",
    )
    args = parser.parse_args()

    if not args.batch_dir.exists():
        raise SystemExit(f"{args.batch_dir} does not exist.")

    result = subprocess.run(
        ["dvc", "add", str(args.batch_dir)], capture_output=True, text=True, check=False
    )

    if result.returncode != 0:
        raise SystemExit(f"dvc add failed:\n{result.stdout}\n{result.stderr}")

    print(result.stdout)
    print(
        "Batch tracked. Next steps:\n"
        f"  git add {args.batch_dir}.dvc data/human_review/.gitignore\n"
        f'  git commit -m "Add human-review batch {args.batch_dir.name}"\n'
        "  dvc push\n"
        "\n"
        "Reference this batch's commit when you retrain, so it's clear which "
        "data version scripts/retrain_and_promote.py evaluated against."
    )


if __name__ == "__main__":
    main()
