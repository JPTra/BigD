"""
Caching layer.

Exposes a single shared `flask-caching` instance plus a tiny helper for
mtime-aware memoisation. The Flask app initialises `cache` once at startup;
service modules import the same object so a cached response is shared across
all requests instead of being recomputed per request.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask_caching import Cache

logger = logging.getLogger(__name__)

# Shared instance. `init_app()` is called from app.py with Config.flask_cache_config().
cache: Cache = Cache()


def file_signature(path: Path) -> float:
    """Return a value that changes whenever the file changes (its mtime).

    Used as part of an lru_cache key so that a parquet file is re-read only
    when it is actually modified — unchanged files are served from memory.
    """
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def clear_all() -> None:
    """Flush every cache layer (flask-caching + lru_cache memoisers)."""
    try:
        cache.clear()
    except Exception as exc:  # pragma: no cover - cache may be uninitialised
        logger.warning("flask-caching clear failed: %s", exc)


__all__ = ["cache", "file_signature", "clear_all"]
