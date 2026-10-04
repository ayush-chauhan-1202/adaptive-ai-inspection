"""Tests for the FastAPI service (Milestone 6/7).

These run against a tiny synthetic model artifact (see conftest.py), not the
real trained PatchCore model - they verify the API's behavior (routing,
validation, response shape, metrics), not detection quality. Detection
quality is what scripts/train_and_register.py's own printed metrics, and the
Milestone 1-5 evaluation work, are for.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _sample_png_bytes() -> bytes:
    array = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_reports_local_model(api_client):
    response = api_client.get("/version")
    assert response.status_code == 200

    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_source"].startswith("local-path:")


def test_inspect_returns_score_and_decision(api_client):
    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    response = api_client.post("/inspect", files=files)

    assert response.status_code == 200
    body = response.json()

    assert body["decision"] in {"AUTO_NORMAL", "HUMAN_REVIEW", "AUTO_DEFECT"}
    assert isinstance(body["score"], float)
    assert len(body["heatmap_preview"]) == body["heatmap_preview_size"]
    assert len(body["heatmap_preview"][0]) == body["heatmap_preview_size"]
    assert body["latency_ms"] > 0
    assert body["model_source"].startswith("local-path:")


def test_inspect_rejects_non_image_upload(api_client):
    files = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    response = api_client.post("/inspect", files=files)

    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]


def test_inspect_rejects_oversized_upload(api_client, monkeypatch):
    import inspection.api.main as main_module

    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 10)

    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    response = api_client.post("/inspect", files=files)

    assert response.status_code == 413


def test_metrics_endpoint_exposes_prometheus_text(api_client):
    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    api_client.post("/inspect", files=files)

    response = api_client.get("/metrics")

    assert response.status_code == 200
    assert "inspection_requests_total" in response.text
    assert "inspection_request_latency_seconds" in response.text
