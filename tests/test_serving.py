"""Unit tests for the serving layer added in Milestone 6/7, independent of the API."""

from __future__ import annotations

import pandas as pd
import pytest

from inspection.serving.registry import (
    ENV_MODEL_LOCAL_PATH,
    ENV_TRACKING_URI,
    ModelUnavailableError,
    load_production_model,
    model_source_description,
)


def test_pyfunc_model_roundtrip(local_model_dir):
    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(str(local_model_dir))

    # Use one of the dataset's own synthetic images rather than a bare path,
    # so this stays a real decode-and-score pass, not just a load check.
    import numpy as np
    from PIL import Image

    image_path = local_model_dir.parent / "probe.png"
    array = (np.random.default_rng(2).random((64, 64, 3)) * 255).astype("uint8")
    Image.fromarray(array, mode="RGB").save(image_path)

    result = model.predict(pd.DataFrame({"image_path": [str(image_path)]}))

    assert len(result) == 1
    assert result[0]["decision"] in {"AUTO_NORMAL", "HUMAN_REVIEW", "AUTO_DEFECT"}
    assert len(result[0]["anomaly_map"]) == 224


def test_load_production_model_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv(ENV_TRACKING_URI, raising=False)
    monkeypatch.delenv(ENV_MODEL_LOCAL_PATH, raising=False)
    load_production_model.cache_clear()

    with pytest.raises(ModelUnavailableError):
        load_production_model()

    load_production_model.cache_clear()


def test_load_production_model_uses_local_fallback(local_model_dir, monkeypatch):
    monkeypatch.delenv(ENV_TRACKING_URI, raising=False)
    monkeypatch.setenv(ENV_MODEL_LOCAL_PATH, str(local_model_dir))
    load_production_model.cache_clear()

    model = load_production_model()
    assert model is not None
    assert model_source_description().startswith("local-path:")

    load_production_model.cache_clear()
