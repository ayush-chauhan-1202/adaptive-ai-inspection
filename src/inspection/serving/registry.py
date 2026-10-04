"""Model loading for the API.

Milestone 7's whole point is that the API shouldn't know about a file path -
it should ask the MLflow Model Registry for "whatever is in Production" and
let promotion (scripts/retrain_and_promote.py) be the thing that changes what
gets served. That's `load_production_model`.

A local-path fallback (`INSPECTION_MODEL_PATH`) is kept alongside it on purpose:
it's what lets CI run API tests without a live MLflow tracking server, and
it's what a contributor without any MLflow setup yet can use for a quick local
smoke test. Precedence is registry-first, so in any environment that *does*
have MLflow configured, the registry wins.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import mlflow
import mlflow.pyfunc

logger = logging.getLogger("inspection.api")

DEFAULT_MODEL_NAME = "adaptive-ai-inspection"
# MLflow's "stage" field (Staging/Production) is deprecated in favor of
# aliases (mlflow.org/docs/latest/model-registry.html#migrating-from-stages):
# an alias is just a mutable name -> version pointer, which is exactly what
# "whatever is currently in production" means here, and it's the API that
# won't be removed out from under this project later.
DEFAULT_ALIAS = "production"

ENV_MODEL_NAME = "INSPECTION_MODEL_NAME"
ENV_MODEL_ALIAS = "INSPECTION_MODEL_ALIAS"
ENV_MODEL_LOCAL_PATH = "INSPECTION_MODEL_PATH"
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"


class ModelUnavailableError(RuntimeError):
    """Raised when no model can be loaded from the registry or a local path."""


def _load_from_registry(model_name: str, alias: str) -> mlflow.pyfunc.PyFuncModel | None:
    tracking_uri = os.environ.get(ENV_TRACKING_URI)

    if not tracking_uri:
        logger.info("%s not set; skipping MLflow Model Registry", ENV_TRACKING_URI)
        return None

    try:
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Loaded model %s from MLflow registry (%s)", model_uri, tracking_uri)
        return model
    except Exception as exc:  # noqa: BLE001 - we deliberately fall back on any failure
        logger.warning(
            "Could not load %s@%s from MLflow registry at %s: %s",
            model_name,
            alias,
            tracking_uri,
            exc,
        )
        return None


def _load_from_local_path(path: str) -> mlflow.pyfunc.PyFuncModel | None:
    try:
        model = mlflow.pyfunc.load_model(path)
        logger.info("Loaded model from local path %s", path)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load model from local path %s: %s", path, exc)
        return None


@lru_cache(maxsize=1)
def load_production_model() -> mlflow.pyfunc.PyFuncModel:
    """Load the model the API should serve, registry first, local path second.

    Cached with lru_cache so the (relatively expensive, since it loads a
    ResNet-18) model is only loaded once per process, not once per request.
    Call `load_production_model.cache_clear()` to force a reload, e.g. after
    a new model is promoted and the service is told to refresh without a
    restart.
    """

    model_name = os.environ.get(ENV_MODEL_NAME, DEFAULT_MODEL_NAME)
    alias = os.environ.get(ENV_MODEL_ALIAS, DEFAULT_ALIAS)

    model = _load_from_registry(model_name, alias)

    if model is None:
        local_path = os.environ.get(ENV_MODEL_LOCAL_PATH)

        if local_path:
            model = _load_from_local_path(local_path)

    if model is None:
        raise ModelUnavailableError(
            "No model could be loaded. Set MLFLOW_TRACKING_URI (and register/"
            "promote a model named "
            f"'{model_name}' with alias '{alias}'), or set {ENV_MODEL_LOCAL_PATH} "
            "to a local mlflow pyfunc model directory for local/dev use."
        )

    return model


def model_source_description() -> str:
    """Human-readable description of where the currently loaded model came from.

    Used by GET /version so a caller can tell at a glance whether they're
    hitting a registry-served model or a local fallback artifact.
    """

    if os.environ.get(ENV_TRACKING_URI):
        model_name = os.environ.get(ENV_MODEL_NAME, DEFAULT_MODEL_NAME)
        alias = os.environ.get(ENV_MODEL_ALIAS, DEFAULT_ALIAS)
        return f"mlflow-registry:{model_name}@{alias}"

    local_path = os.environ.get(ENV_MODEL_LOCAL_PATH)

    if local_path:
        return f"local-path:{local_path}"

    return "unconfigured"
