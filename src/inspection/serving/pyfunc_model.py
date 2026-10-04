"""MLflow pyfunc wrapper around the Milestone 3-5 PatchCore + triage pipeline.

This is the one piece of serialization the project didn't need before Milestone 6:
every earlier milestone script (run_patchcore.py, run_human_loop.py, ...) rebuilds
the patch memory bank from the training images every time it runs. To serve the
model behind an API we need something we can save once, register in MLflow, and
load later without re-reading the training set. That's what this module adds.

The artifact format is deliberately simple:
  memory_bank.npy  - the fitted PatchMemoryBank's normal-patch embeddings
  metadata.json    - thresholds + everything needed to reproduce how the model
                      was trained (category, image size, patch feature layer)

Wrapping it as an mlflow.pyfunc.PythonModel (rather than just pickling the numpy
array) is what lets `mlflow.pyfunc.log_model(...)` register it in the Model
Registry and lets the API load "whatever is currently in Production" by name
and stage, instead of a hardcoded file path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from inspection.localization.patchcore import (
    PatchMemoryBank,
    ResNetFeatureExtractor,
    compute_anomaly_map,
)
from inspection.localization.triage import classify_score

ARTIFACT_MEMORY_BANK = "memory_bank"
ARTIFACT_METADATA = "metadata"


class InspectionModel(mlflow.pyfunc.PythonModel):
    """Serves PatchCore anomaly localization + 3-way triage as a single model.

    Input (model_input): a DataFrame with one column, "image_path", holding
    paths to images already on disk (the API writes the uploaded image to a
    temp file before calling predict - pyfunc models don't take raw bytes).

    Output: a list of dicts, one per row, each with:
        score           - max patch anomaly score for the image (float)
        decision        - "AUTO_NORMAL" | "HUMAN_REVIEW" | "AUTO_DEFECT"
        anomaly_map     - HxW list of per-pixel anomaly scores (same scale as
                           the Milestone 3 visualizations)
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        features = np.load(context.artifacts[ARTIFACT_MEMORY_BANK])

        self.memory_bank = PatchMemoryBank()
        self.memory_bank.fit(features)

        with open(context.artifacts[ARTIFACT_METADATA]) as handle:
            self.metadata = json.load(handle)

        self.review_threshold = float(self.metadata["review_threshold"])
        self.defect_threshold = float(self.metadata["defect_threshold"])

        # CPU is the right default for a serving container: no GPU to assume,
        # and PatchCore inference on a single image is fast enough on CPU.
        # `pretrained_weights` must match whatever the memory bank was built
        # with at training time, or the embeddings won't be comparable.
        self.extractor = ResNetFeatureExtractor(
            device="cpu",
            pretrained=self.metadata.get("pretrained_weights", True),
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = []

        for image_path in model_input["image_path"]:
            sample = SimpleNamespace(image_path=image_path)

            _, anomaly_map, score = compute_anomaly_map(
                sample,
                self.extractor,
                self.memory_bank,
            )

            triage = classify_score(
                score,
                self.review_threshold,
                self.defect_threshold,
            )

            results.append(
                {
                    "score": triage.score,
                    "decision": triage.decision,
                    "anomaly_map": anomaly_map.tolist(),
                }
            )

        return results


def save_artifact_files(
    output_dir: Path,
    memory_bank_features: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write the two files log_model needs, return their paths as a dict.

    Kept separate from the training script so tests can build a tiny synthetic
    artifact without going through full PatchCore training.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_bank_path = output_dir / "memory_bank.npy"
    metadata_path = output_dir / "metadata.json"

    np.save(memory_bank_path, memory_bank_features.astype(np.float32))

    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        ARTIFACT_MEMORY_BANK: str(memory_bank_path),
        ARTIFACT_METADATA: str(metadata_path),
    }
