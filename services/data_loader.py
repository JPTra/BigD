"""
Data loader.

Reads the aggregated Parquet files produced by the PySpark ETL. Key properties:

* Auto-discovery — every `*.parquet` file in ``processed_data/`` is loaded; no
  filenames are hard-coded, so a new aggregation appears automatically.
* mtime-aware caching — a file is parsed only once and kept in memory; it is
  re-read only if the file on disk changes. Unchanged files never hit disk
  again, so ``/api/data`` is fast.
* No PySpark — uses pandas + pyarrow only, keeping Flask runtime lightweight.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config
from services.cache import file_signature

logger = logging.getLogger(__name__)


# Dashboard aggregations follow the ``agg_*.parquet`` naming convention. The
# loader only serves these — intermediate training tables that the ETL also
# writes to processed_data/ (e.g. training_rf.parquet, customer_features.parquet)
# are large and must NOT leak into the /api/data response.
AGG_GLOB = "agg_*.parquet"


def discover_parquet_files(processed_dir: Path | None = None) -> list[Path]:
    """Return every dashboard aggregation Parquet (``agg_*.parquet``), sorted.

    Only top-level single-file aggregations are returned; training-dataset
    directories are excluded. New ``agg_*.parquet`` files are picked up
    automatically (auto-discovery) without code changes or a restart.
    """
    directory = Path(processed_dir or Config.PROCESSED_DIR)
    if not directory.exists():
        logger.warning("Processed-data directory does not exist: %s", directory)
        return []
    files = sorted(p for p in directory.glob(AGG_GLOB) if p.is_file())
    logger.debug("Discovered %d aggregation parquet file(s) in %s", len(files), directory)
    return files


@lru_cache(maxsize=64)
def _read_parquet_cached(path_str: str, _signature: float) -> tuple[dict[str, Any], ...]:
    """Read one Parquet file into a tuple of row-dicts.

    The result is memoised on ``(path, mtime)``. ``_signature`` is the file's
    mtime: when the file changes the key changes and the file is re-read, which
    is what lets the loader pick up freshly-produced ETL output without a
    restart while still never re-reading an unchanged file.
    """
    logger.info("Reading parquet from disk: %s", path_str)
    df = pd.read_parquet(path_str)
    # Tuple of dicts is hashable/immutable -> safe to cache and cheap to JSON-ify.
    return tuple(df.to_dict(orient="records"))


def load_parquet(path: Path) -> list[dict[str, Any]]:
    """Load a single Parquet file as a list of row dicts (cached by mtime)."""
    path = Path(path)
    rows = _read_parquet_cached(str(path), file_signature(path))
    return list(rows)


def load_all() -> dict[str, list[dict[str, Any]]]:
    """Load every aggregation Parquet, keyed by filename stem.

    Example return::

        {
            "agg_market":   [{...}, ...],
            "agg_monthly":  [{...}, ...],
            "agg_shipping": [{...}, ...],
        }

    Reads are served from the per-file cache, so repeated calls are cheap.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for file in discover_parquet_files():
        try:
            result[file.stem] = load_parquet(file)
        except Exception as exc:
            logger.exception("Failed to read %s: %s", file, exc)
    return result


def get_aggregation(name: str) -> list[dict[str, Any]]:
    """Return a single aggregation by stem (e.g. ``"agg_market"``); [] if absent."""
    path = Path(Config.PROCESSED_DIR) / f"{name}.parquet"
    if not path.exists():
        logger.warning("Aggregation not found: %s", path)
        return []
    return load_parquet(path)


__all__ = ["discover_parquet_files", "load_parquet", "load_all", "get_aggregation"]
