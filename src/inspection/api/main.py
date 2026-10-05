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
from fastapi.responses import HTMLResponse, Response
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

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Adaptive AI Inspection</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0b0d12;
    --panel: #151822;
    --border: #262b38;
    --text: #e6e8ee;
    --muted: #8a90a2;
    --accent: #5b8cff;
    --green: #35c46b;
    --amber: #e3b341;
    --red: #e5484d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    justify-content: center;
    padding: 32px 16px 64px;
  }
  main { width: 100%; max-width: 640px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 24px; font-size: 14px; }
  #drop {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s;
    background: var(--panel);
  }
  #drop.drag { border-color: var(--accent); }
  #drop p { margin: 0; color: var(--muted); font-size: 14px; }
  #fileInput { display: none; }
  #preview-wrap {
    position: relative;
    margin-top: 20px;
    display: none;
    line-height: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  #previewImg { width: 100%; display: block; }
  #heatmapCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  #runBtn {
    margin-top: 16px;
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 8px;
    background: var(--accent);
    color: white;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
  }
  #runBtn:disabled { opacity: 0.5; cursor: default; }
  #result { margin-top: 20px; display: none; }
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.02em;
  }
  .badge.normal { background: rgba(53,196,107,0.15); color: var(--green); }
  .badge.review { background: rgba(227,179,65,0.15); color: var(--amber); }
  .badge.defect { background: rgba(229,72,77,0.15); color: var(--red); }
  .stats { margin-top: 12px; display: flex; gap: 24px; color: var(--muted); font-size: 13px; }
  .stats b { color: var(--text); font-weight: 600; }
  #error {
    margin-top: 16px;
    padding: 12px 14px;
    border-radius: 8px;
    background: rgba(229,72,77,0.1);
    color: var(--red);
    font-size: 13px;
    display: none;
  }
</style>
</head>
<body>
<main>
  <h1>Adaptive AI Inspection</h1>
  <p class="sub">Upload a product image to run it through the PatchCore anomaly-localization model.</p>

  <div id="drop">
    <p>Click to choose an image, or drag one here</p>
    <input id="fileInput" type="file" accept="image/*" />
  </div>

  <div id="preview-wrap">
    <img id="previewImg" />
    <canvas id="heatmapCanvas"></canvas>
  </div>

  <button id="runBtn" disabled>Run inspection</button>

  <div id="error"></div>

  <div id="result">
    <span id="badge" class="badge"></span>
    <div class="stats">
      <div>Score <b id="scoreVal">-</b></div>
      <div>Latency <b id="latencyVal">-</b></div>
    </div>
  </div>
</main>

<script>
  const drop = document.getElementById("drop");
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("preview-wrap");
  const previewImg = document.getElementById("previewImg");
  const canvas = document.getElementById("heatmapCanvas");
  const runBtn = document.getElementById("runBtn");
  const resultEl = document.getElementById("result");
  const badgeEl = document.getElementById("badge");
  const scoreEl = document.getElementById("scoreVal");
  const latencyEl = document.getElementById("latencyVal");
  const errorEl = document.getElementById("error");

  let selectedFile = null;

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.onload = () => {
      canvas.width = previewImg.clientWidth;
      canvas.height = previewImg.clientHeight;
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    };
    previewImg.src = url;
    previewWrap.style.display = "block";
    runBtn.disabled = false;
    resultEl.style.display = "none";
    errorEl.style.display = "none";
  }

  function heatColor(t) {
    const hue = 240 - 240 * t;
    return `hsla(${hue}, 85%, 50%, ${0.15 + 0.45 * t})`;
  }

  function drawHeatmap(grid, size) {
    const ctx = canvas.getContext("2d");
    canvas.width = previewImg.clientWidth;
    canvas.height = previewImg.clientHeight;

    let min = Infinity, max = -Infinity;
    for (const row of grid) for (const v of row) { if (v < min) min = v; if (v > max) max = v; }
    const range = max - min || 1;

    const cellW = canvas.width / size;
    const cellH = canvas.height / size;

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const t = (grid[i][j] - min) / range;
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  function decisionClass(decision) {
    if (decision === "AUTO_NORMAL") return "normal";
    if (decision === "HUMAN_REVIEW") return "review";
    if (decision === "AUTO_DEFECT") return "defect";
    return "";
  }

  runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    runBtn.disabled = true;
    runBtn.textContent = "Running\u2026";
    errorEl.style.display = "none";

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch("/inspect", { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();

      drawHeatmap(data.heatmap_preview, data.heatmap_preview_size);

      badgeEl.textContent = data.decision;
      badgeEl.className = "badge " + decisionClass(data.decision);
      scoreEl.textContent = data.score.toFixed(3);
      latencyEl.textContent = Math.round(data.latency_ms) + " ms";
      resultEl.style.display = "block";
    } catch (err) {
      errorEl.textContent = err.message || "Something went wrong.";
      errorEl.style.display = "block";
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "Run inspection";
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


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
