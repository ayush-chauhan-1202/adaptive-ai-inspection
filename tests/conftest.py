"""Shared pytest fixtures.

The API and serving tests need *some* model to load, but they shouldn't need
a live MLflow tracking server, the real MVTec dataset, or internet access to
download ResNet-18 weights - none of that should be a precondition for
`pytest` to pass in CI. `local_model_dir` builds the smallest possible valid
model artifact (a 3-vector memory bank, untrained weights) and saves it with
`mlflow.pyfunc.save_model` to a plain local directory, which the registry
module's local-path fallback can load directly.
"""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pytest

from inspection.serving.pyfunc_model import InspectionModel, save_artifact_files
from inspection.serving.registry import (
    ENV_MODEL_LOCAL_PATH,
    ENV_TRACKING_URI,
    load_production_model,
)


@pytest.fixture
def local_model_dir(tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "artifacts"
    model_dir = tmp_path / "model"

    memory_bank_features = np.random.default_rng(0).normal(size=(8, 256)).astype(np.float32)
    metadata = {
        "category": "bottle",
        "review_threshold": -1.0,
        "defect_threshold": 100.0,
        "image_size": 224,
        "feature_layer": "resnet18.layer3",
        "pretrained_weights": False,
    }

    artifact_paths = save_artifact_files(artifacts_dir, memory_bank_features, metadata)

    mlflow.pyfunc.save_model(
        path=str(model_dir),
        python_model=InspectionModel(),
        artifacts=artifact_paths,
    )

    return model_dir


@pytest.fixture
def api_client(local_model_dir: Path, monkeypatch):
    """A FastAPI TestClient wired to the local synthetic model artifact."""

    monkeypatch.delenv(ENV_TRACKING_URI, raising=False)
    monkeypatch.setenv(ENV_MODEL_LOCAL_PATH, str(local_model_dir))
    load_production_model.cache_clear()

    from fastapi.testclient import TestClient

    from inspection.api.main import app

    with TestClient(app) as client:
        yield client

    load_production_model.cache_clear()
