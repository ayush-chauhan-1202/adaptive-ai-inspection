"""Publish the current 'production' model to a fixed GCS path.

Why this exists: the deployed API (Cloud Run) needs to load a model, but
pointing it at a live, always-reachable MLflow tracking server in production
means paying to keep that server up (or re-engineering it onto GCS-FUSE
volumes) just so a low-traffic demo service can ask "what's in production?".
That's complexity Milestone 7 doesn't actually need yet, per this project's
own "introduce complexity only when it's justified" principle.

The cheaper, equally real alternative: MLflow already knows how to read a
model directly from a `gs://` URI with no tracking server involved
(mlflow.pyfunc.load_model resolves cloud storage URIs natively via gcsfs).
So promotion - whether the very first registration in train_and_register.py,
or a champion/challenger win in retrain_and_promote.py - also copies the
newly-promoted model's artifacts to one well-known GCS path. The API's
INSPECTION_MODEL_PATH simply points at that path. MLflow's tracking server
and Model Registry are still used for everything they're good at - experiment
comparison, versioning, aliasing - just not as a runtime dependency of the
deployed service.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile

import mlflow.artifacts


def publish_production_bundle(model_name: str, gcs_path: str) -> None:
    """Copy the model currently aliased 'production' to `gcs_path`.

    Requires the `gsutil` CLI to be available and authenticated (it's
    preconfigured in this project's dev environment; in CI it comes from the
    `google-github-actions/auth` + `setup-gcloud` actions - see RUNBOOK.md).
    """

    if shutil.which("gsutil") is None:
        raise RuntimeError(
            "gsutil is not on PATH. Install the Google Cloud SDK, or skip "
            "--publish-gcs-path for a local-only run."
        )

    model_uri = f"models:/{model_name}@production"

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmp_dir)

        result = subprocess.run(
            ["gsutil", "-m", "rsync", "-r", "-d", local_path, gcs_path],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"gsutil rsync to {gcs_path} failed:\n{result.stdout}\n{result.stderr}"
            )

    print(f"Published '{model_name}@production' to {gcs_path}")
