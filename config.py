"""
Central configuration for the Supply Chain Intelligence Dashboard.

All tunables are read from environment variables (optionally from a `.env`
file) so the same code runs unchanged on a laptop or in the cloud.

Usage:
    from config import Config
    Config.PROCESSED_DIR  # -> absolute Path to processed_data/
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    # Optional: load a .env file if python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Project root = directory that contains this file.
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Application configuration. Override any value via environment variables."""

    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR: Path = BASE_DIR
    RAW_DIR: Path = Path(os.environ.get("RAW_DIR", BASE_DIR / "raw_data"))
    PROCESSED_DIR: Path = Path(os.environ.get("PROCESSED_DIR", BASE_DIR / "processed_data"))
    MODELS_DIR: Path = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
    FRONTEND_DIR: Path = Path(os.environ.get("FRONTEND_DIR", BASE_DIR / "frontend"))

    # ── Model files (active models only) ─────────────────────────────────────
    # Production risk model: XGBoost (selected by scripts/benchmark_models.py —
    # 0.35 MB, ~3 ms latency, AUC 0.74). Random Forest has been retired.
    RISK_MODEL_FILE: str = os.environ.get("RISK_MODEL_FILE", "best_xgb_model.pkl")
    KMEANS_MODEL_FILE: str = os.environ.get("KMEANS_MODEL_FILE", "best_kmeans_model.pkl")
    KMEANS_SCALER_FILE: str = os.environ.get("KMEANS_SCALER_FILE", "best_kmeans_scaler.pkl")

    # ── Flask / server ───────────────────────────────────────────────────────
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 5000)
    DEBUG: bool = _env_bool("FLASK_DEBUG", False)

    # ── Caching (flask-caching) ──────────────────────────────────────────────
    CACHE_TYPE: str = os.environ.get("CACHE_TYPE", "SimpleCache")
    CACHE_DEFAULT_TIMEOUT: int = _env_int("CACHE_DEFAULT_TIMEOUT", 300)  # seconds

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

    @classmethod
    def flask_cache_config(cls) -> dict:
        """Config dict consumed by flask-caching's Cache.init_app()."""
        return {
            "CACHE_TYPE": cls.CACHE_TYPE,
            "CACHE_DEFAULT_TIMEOUT": cls.CACHE_DEFAULT_TIMEOUT,
        }
