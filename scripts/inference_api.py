"""
Standalone Flask inference API for the production risk model (XGBoost).

A minimal, self-contained reference showing the deployment pattern:
  * lazy + singleton model loading (loaded once, kept in memory),
  * no PySpark, no training code,
  * single-row prediction in a few milliseconds.

The full dashboard uses app.py (which adds /api/data, K-Means, caching, the
frontend). This file is the smallest possible production-inference service.

Run:
    python scripts/inference_api.py
    curl -X POST http://localhost:5001/predict_risk -H "Content-Type: application/json" \
         -d '{"shipping_mode":"Standard Class","market":"LATAM","sales":150,"discount_rate":0.1,"demand_score":25,"quantity":1,"order_month":6,"order_region":"Southeast Asia"}'
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("inference")

# Feature contract (must match training in train_xgb.py).
FEATURES = [
    "Shipping Mode", "Market", "Order Region",
    "Days for shipment (scheduled)", "Order Item Quantity", "Sales",
    "Order Item Discount Rate", "order_month", "demand_score",
]
SHIPPING_DAYS = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}

# ── Lazy singleton model ─────────────────────────────────────────────────────
_model: Any = None
_lock = threading.Lock()


def get_model() -> Any:
    """Load the XGBoost pipeline once, on first request, then reuse it."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                path = Path(Config.MODELS_DIR) / Config.RISK_MODEL_FILE
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found. Run scripts/train_xgb.py.")
                logger.info("Loading model (first request): %s", path)
                _model = joblib.load(path)
    return _model


def build_features(body: dict) -> pd.DataFrame:
    """Map the API payload to the model's expected feature frame."""
    shipping_mode = str(body.get("shipping_mode", "Standard Class"))
    row = {
        "Shipping Mode": shipping_mode,
        "Market": str(body.get("market", "LATAM")),
        "Order Region": str(body.get("order_region", "Southeast Asia")),
        "Days for shipment (scheduled)": int(SHIPPING_DAYS.get(shipping_mode, 4)),
        "Order Item Quantity": float(body.get("quantity", 1)),
        "Sales": float(body.get("sales", 150.0)),
        "Order Item Discount Rate": float(body.get("discount_rate", 0.1)),
        "order_month": int(body.get("order_month", 6)),
        "demand_score": float(body.get("demand_score", 25.0)),
    }
    return pd.DataFrame([row])[FEATURES]


app = Flask(__name__)


@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Empty JSON body."}), 400
    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    try:
        df = build_features(body)
        proba = model.predict_proba(df)[0]
        classes = list(model.classes_)
        late_idx = classes.index(1) if 1 in classes else len(classes) - 1
        late_prob = float(proba[late_idx])
        prediction = int(late_prob >= 0.5)
        return jsonify(
            {
                "prediction": prediction,
                "probability": round(late_prob, 4),
                "prediction_label": "Late Delivery Risk" if prediction else "On Time / Advance",
            }
        )
    except Exception as exc:
        logger.exception("prediction failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
