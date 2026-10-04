"""REST API for the inspection pipeline (Milestone 6/7).

This is the first time the pipeline is reachable as a service instead of a
script. Design choices worth calling out:

- POST /inspect takes a real uploaded image, not a path on the server's disk
  - the whole point of an API is that the caller doesn't have filesystem
  access to wherever this is running.
- The heatmap in the response is downsampled (see `_downsample`). The raw
  PatchCore anomaly map is 224x224 floats; returning that as JSON on every
  request is ~50K numbers for a value nobody renders at full resolution.
  A 28x28 preview is enough to draw a heatmap overlay client-side.
- Model loading goes through `inspection.serving.registry`, so this file does
  not know or care whether the model came from the MLflow Registry or a local
  path - that decision point is Milestone 7's, not the API's.
"""

from __future__ import annotations

import io
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from inspection.serving.logging_config import configure_logging
from inspection.serving.registry import (
    ModelUnavailableError,
    load_production_model,
    model_source_description,
)

configure_logging()
logger = logging.getLogger("inspection.api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB - generous for a single inspection image
HEATMAP_PREVIEW_SIZE = 28

REQUEST_COUNT = Counter(
    "inspection_requests_total",
    "Total /inspect requests handled, by outcome",
    ["decision"],
)
REQUEST_ERRORS = Counter(
    "inspection_request_errors_total",
    "Total /inspect requests that failed, by reason",
    ["reason"],
)
REQUEST_LATENCY = Histogram(
    "inspection_request_latency_seconds",
    "End-to-end /inspect latency in seconds",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load (and cache) the model at startup rather than on the first request,
    # so a broken model config fails fast on boot instead of on someone's
    # first real inspection call.
    try:
        load_production_model()
        logger.info("startup_model_loaded", extra={"model_source": model_source_description()})
    except ModelUnavailableError as exc:
        # Deliberately not fatal: lets /health report "model_loaded: false"
        # instead of the container crash-looping when MLflow isn't reachable
        # yet (e.g. first boot, before a model has ever been registered).
        logger.error("startup_model_unavailable", extra={"error": str(exc)})

    yield


def _app_version() -> str:
    try:
        return pkg_version("adaptive-ai-inspection")
    except PackageNotFoundError:
        return "0.0.0+unknown"


app = FastAPI(
    title="Adaptive AI Inspection API",
    description="Serves the PatchCore anomaly-localization + triage pipeline.",
    version=_app_version(),
    lifespan=lifespan,
)


class InspectResponse(BaseModel):
    request_id: str
    score: float
    decision: str
    heatmap_preview: list[list[float]]
    heatmap_preview_size: int
    model_source: str
    latency_ms: float


class VersionResponse(BaseModel):
    app_version: str
    model_source: str
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str


def _downsample(anomaly_map: list[list[float]], size: int) -> list[list[float]]:
    """Average-pool a 2D list of floats down to size x size for compact JSON."""

    import numpy as np

    array = np.asarray(anomaly_map, dtype=np.float32)
    height, width = array.shape

    row_edges = np.linspace(0, height, size + 1).astype(int)
    col_edges = np.linspace(0, width, size + 1).astype(int)

    pooled = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            block = array[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            pooled[i, j] = block.mean() if block.size else 0.0

    return pooled.round(4).tolist()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def get_version() -> VersionResponse:
    model_loaded = True
    try:
        load_production_model()
    except ModelUnavailableError:
        model_loaded = False

    return VersionResponse(
        app_version=app.version,
        model_source=model_source_description(),
        model_loaded=model_loaded,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)) -> InspectResponse:  # noqa: B008 (FastAPI's documented DI pattern)
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        REQUEST_ERRORS.labels(reason="file_too_large").inc()
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds maximum size of {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            Image.open(io.BytesIO(contents)).convert("RGB").save(tmp_path, format="PNG")
    except UnidentifiedImageError as exc:
        REQUEST_ERRORS.labels(reason="invalid_image").inc()
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc

    try:
        try:
            model = load_production_model()
        except ModelUnavailableError as exc:
            REQUEST_ERRORS.labels(reason="model_unavailable").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            predictions: list[dict[str, Any]] = model.predict(
                pd.DataFrame({"image_path": [str(tmp_path)]})
            )
        except Exception as exc:
            REQUEST_ERRORS.labels(reason="inference_failed").inc()
            logger.error(
                "inspect_inference_failed",
                extra={"request_id": request_id, "error": str(exc)},
            )
            raise HTTPException(status_code=500, detail="Inference failed.") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    result = predictions[0]
    heatmap_preview = _downsample(result["anomaly_map"], HEATMAP_PREVIEW_SIZE)

    latency_ms = (time.perf_counter() - start_time) * 1000
    REQUEST_COUNT.labels(decision=result["decision"]).inc()
    REQUEST_LATENCY.observe(latency_ms / 1000)

    logger.info(
        "inspect_request",
        extra={
            "request_id": request_id,
            "score": result["score"],
            "decision": result["decision"],
            "latency_ms": round(latency_ms, 2),
        },
    )

    return InspectResponse(
        request_id=request_id,
        score=result["score"],
        decision=result["decision"],
        heatmap_preview=heatmap_preview,
        heatmap_preview_size=HEATMAP_PREVIEW_SIZE,
        model_source=model_source_description(),
        latency_ms=round(latency_ms, 2),
    )
