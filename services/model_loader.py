"""
Model loader — lazy, singleton, thread-safe, in-memory.

* Lazy: a model's ``.pkl`` is read the first time it is requested, never at
  import/startup. Flask boots in well under a second.
* Singleton: once loaded, the model lives in a module-level cache and every
  subsequent request reuses the same object — the file is read exactly once
  for the entire runtime.
* Thread-safe: a lock guards first-load so concurrent requests can't trigger a
  double read.
* No PySpark: inference uses plain scikit-learn / XGBoost objects; a Spark
  session is never created at runtime.

Active models:
    risk   -> best_xgb_model.pkl   (XGBoost pipeline — late-delivery risk)
    kmeans -> best_kmeans_model.pkl (+ optional best_kmeans_scaler.pkl)
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import joblib

from config import Config

logger = logging.getLogger(__name__)

# Module-level singleton cache: name -> loaded object. Loaded once per runtime.
_MODELS: dict[str, Any] = {}
_LOCK = threading.Lock()


def _load_pickle(filename: str) -> Any:
    """Load a pickle from the models directory, raising a clear error if absent."""
    path = Path(Config.MODELS_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run the training scripts in scripts/ first."
        )
    logger.info("Loading model from disk (first call): %s", path)
    model = joblib.load(path)
    logger.info("Loaded model %s (%s)", filename, type(model).__name__)
    return model


def _get(name: str, filename: str) -> Any:
    """Return a cached model, loading it once on first access (thread-safe)."""
    model = _MODELS.get(name)
    if model is not None:
        return model
    with _LOCK:
        # Re-check inside the lock in case another thread just loaded it.
        model = _MODELS.get(name)
        if model is None:
            model = _load_pickle(filename)
            _MODELS[name] = model
        return model


def get_risk_model() -> Any:
    """Lazily return the production risk pipeline (XGBoost, ``best_xgb_model.pkl``)."""
    return _get("risk", Config.RISK_MODEL_FILE)


def get_kmeans_model() -> Any:
    """Lazily return the K-Means model (``best_kmeans_model.pkl``)."""
    return _get("kmeans", Config.KMEANS_MODEL_FILE)


def get_kmeans_scaler() -> Any | None:
    """Lazily return the K-Means feature scaler, or ``None`` if not present.

    The scaler is optional: if it was bundled into the model pipeline there may
    be no standalone scaler file, in which case callers should skip scaling.
    """
    scaler = _MODELS.get("kmeans_scaler")
    if scaler is not None:
        return scaler
    path = Path(Config.MODELS_DIR) / Config.KMEANS_SCALER_FILE
    if not path.exists():
        logger.info("No standalone K-Means scaler at %s (skipping).", path)
        return None
    with _LOCK:
        if _MODELS.get("kmeans_scaler") is None:
            _MODELS["kmeans_scaler"] = _load_pickle(Config.KMEANS_SCALER_FILE)
        return _MODELS["kmeans_scaler"]


def warmup() -> None:
    """Optionally pre-load all active models (e.g. from a thread post-boot)."""
    try:
        get_risk_model()
        get_kmeans_model()
        get_kmeans_scaler()
    except FileNotFoundError as exc:
        logger.warning("Warmup skipped: %s", exc)


def reset() -> None:
    """Drop all cached models (used by tests / after re-training)."""
    with _LOCK:
        _MODELS.clear()


__all__ = [
    "get_risk_model",
    "get_kmeans_model",
    "get_kmeans_scaler",
    "warmup",
    "reset",
]
