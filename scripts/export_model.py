"""
Export / package the trained risk model for deployment.

Takes the model produced by train_xgb.py and:
  1. re-serialises it with maximum compression (smallest possible .pkl),
  2. writes a sidecar ``best_xgb_model.meta.json`` describing the contract
     (feature order, classes, library versions, size, smoke-test latency),
  3. runs a smoke prediction so a broken artifact never ships.

This decouples *training* from *packaging*: CI can re-export and validate the
artifact without retraining.

Run:
    python scripts/export_model.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn  # noqa: E402
import xgboost  # noqa: E402

from config import Config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | export | %(message)s"
)
logger = logging.getLogger("export")

CATEGORICAL = ["Shipping Mode", "Market", "Order Region"]
NUMERIC = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate",
    "order_month",
    "demand_score",
]
FEATURES = CATEGORICAL + NUMERIC

SAMPLE = {
    "Days for shipment (scheduled)": 4,
    "Shipping Mode": "Standard Class",
    "Market": "LATAM",
    "Order Region": "Southeast Asia",
    "Order Item Quantity": 1.0,
    "Sales": 150.0,
    "Order Item Discount Rate": 0.1,
    "order_month": 6,
    "demand_score": 25.0,
}


def main() -> None:
    src = Path(Config.MODELS_DIR) / Config.RISK_MODEL_FILE
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run scripts/train_xgb.py first.")

    logger.info("Loading model: %s", src)
    pipeline = joblib.load(src)

    # 1. Re-serialise with maximum compression in place.
    joblib.dump(pipeline, src, compress=("xz", 9))
    size_mb = src.stat().st_size / 1024 / 1024
    logger.info("Re-exported with max compression -> %.3f MB", size_mb)

    # 2. Smoke prediction + latency.
    df = pd.DataFrame([SAMPLE])[FEATURES]
    for _ in range(20):  # warmup
        pipeline.predict_proba(df)
    times = []
    for _ in range(200):
        t = time.perf_counter()
        pipeline.predict_proba(df)
        times.append((time.perf_counter() - t) * 1000)
    latency_ms = float(np.median(times))
    proba = pipeline.predict_proba(df)[0]
    classes = [int(c) for c in pipeline.classes_]
    logger.info("Smoke prediction proba=%s (classes=%s)  latency=%.3f ms", proba, classes, latency_ms)

    assert size_mb < 20, f"Model too large: {size_mb:.2f} MB"
    assert latency_ms < 20, f"Latency too high: {latency_ms:.2f} ms"

    # 3. Metadata sidecar (the API contract).
    meta = {
        "model_file": Config.RISK_MODEL_FILE,
        "model_type": "XGBoost (sklearn Pipeline: OneHotEncoder + XGBClassifier)",
        "task": "binary classification — late delivery risk",
        "feature_order": FEATURES,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "classes": classes,
        "positive_class": 1,
        "size_mb": round(size_mb, 3),
        "smoke_latency_ms": round(latency_ms, 3),
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    meta_path = src.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote metadata -> %s", meta_path)
    logger.info("Export OK: %.3f MB, %.3f ms — ready for deployment.", size_mb, latency_ms)


if __name__ == "__main__":
    main()
