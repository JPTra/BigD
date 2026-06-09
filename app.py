"""
Supply Chain Intelligence Dashboard — Flask API.

This module is intentionally thin: it wires routes to the service layer and
nothing else. There is **no ETL and no model training here** (those live in
``scripts/``), and **no PySpark** is imported at runtime.

Endpoints:
    GET  /                      -> serve the dashboard
    GET  /api/health            -> liveness probe
    GET  /api/data              -> market + monthly + shipping aggregations (cached)
    POST /api/predict_risk      -> XGBoost late-delivery risk
    POST /api/predict_cluster   -> K-Means customer segment

Run:
    python app.py
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import Config
from services import data_loader, model_loader
from services.cache import cache

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

# ── App factory ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(Config.FRONTEND_DIR))
app.config.update(Config.flask_cache_config())
CORS(app)
cache.init_app(app)

# ── Domain constants (labels only — not business logic) ──────────────────────
SHIPPING_DAYS: dict[str, int] = {
    "Standard Class": 4,
    "Second Class": 2,
    "First Class": 1,
    "Same Day": 0,
}

CLUSTER_PROFILES: dict[int, dict[str, str]] = {
    0: {
        "name": "Cluster 0 — Premium Customer",
        "desc": "High transaction value with moderate digital engagement — prioritise "
                "premium programs and exclusive service.",
    },
    1: {
        "name": "Cluster 1 — Digital Explorer",
        "desc": "Highest digital interest but lowest transaction value — a wide "
                "conversion gap; target with retargeting ads and limited-time offers.",
    },
    2: {
        "name": "Cluster 2 — Selective Buyer",
        "desc": "Moderate transaction value with low digital demand — focus on "
                "retention via value-based loyalty programs.",
    },
}


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the dashboard frontend."""
    return send_from_directory(str(Config.FRONTEND_DIR), "dashboard.html")


@app.route("/api/health")
def health():
    """Lightweight liveness probe (does not touch models or disk-heavy work)."""
    return jsonify({"status": "ok"})


@app.route("/api/data")
@cache.cached(timeout=Config.CACHE_DEFAULT_TIMEOUT)
def get_data():
    """Return all aggregations. Cached so Parquet is not re-read every request."""
    aggregations = data_loader.load_all()

    market = aggregations.get("agg_market", [])
    monthly = aggregations.get("agg_monthly", [])
    shipping = aggregations.get("agg_shipping", [])

    total_orders = sum(int(r.get("Total_Orders", 0)) for r in shipping)
    active_risks = sum(
        int(r.get("Total_Orders", 0)) * float(r.get("Late_Rate", 0.0)) for r in shipping
    )

    return jsonify(
        {
            "stats": {
                "total_transactions": total_orders,
                "active_risks": int(active_risks),
            },
            "agg_market": market,
            "agg_monthly": monthly,
            "agg_shipping": shipping,
            # Pass through any extra aggregations discovered in processed_data/.
            **{k: v for k, v in aggregations.items()
               if k not in ("agg_market", "agg_monthly", "agg_shipping")},
        }
    )


@app.route("/api/predict_risk", methods=["POST"])
def predict_risk():
    """Predict late-delivery risk using the XGBoost pipeline (lazy-loaded)."""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Empty JSON request body."}), 400

    try:
        model = model_loader.get_risk_model()
    except FileNotFoundError as exc:
        logger.error("Risk model unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503

    try:
        shipping_mode = str(body.get("shipping_mode", "Standard Class"))
        features = {
            "Days for shipment (scheduled)": int(SHIPPING_DAYS.get(shipping_mode, 4)),
            "Shipping Mode": shipping_mode,
            "Market": str(body.get("market", "LATAM")),
            "Order Region": str(body.get("order_region", "Southeast Asia")),
            "Order Item Quantity": float(body.get("quantity", 1)),
            "Sales": float(body.get("sales", 150.0)),
            "Order Item Discount Rate": float(body.get("discount_rate", 0.1)),
            "order_month": int(body.get("order_month", 6)),
            "demand_score": float(body.get("demand_score", 25.0)),
        }
        df = pd.DataFrame([features])

        prediction = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0]
        classes = list(model.classes_)
        late_idx = classes.index(1) if 1 in classes else len(classes) - 1
        late_prob = float(proba[late_idx])

        label = "⚠️ Late Delivery Risk" if prediction == 1 else "✅ On Time / Advance"
        return jsonify(
            {
                "prediction": prediction,
                "probability": round(late_prob, 4),
                "prediction_label": label,
            }
        )
    except Exception as exc:
        logger.exception("predict_risk failed")
        return jsonify({"error": f"Prediction error: {exc}"}), 500


@app.route("/api/predict_cluster", methods=["POST"])
def predict_cluster():
    """Predict customer segment using the K-Means model (lazy-loaded)."""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Empty JSON request body."}), 400

    try:
        model = model_loader.get_kmeans_model()
        scaler = model_loader.get_kmeans_scaler()
    except FileNotFoundError as exc:
        logger.error("K-Means model unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503

    try:
        features = np.array(
            [[
                float(body.get("sales_per_customer", 180.0)),
                float(body.get("profit_ratio", 0.12)),
                float(body.get("demand_score", 15.0)),
            ]]
        )
        x = scaler.transform(features) if scaler is not None else features
        cluster_id = int(model.predict(x)[0])
        profile = CLUSTER_PROFILES.get(cluster_id, CLUSTER_PROFILES[0])

        return jsonify({"cluster": cluster_id, "profile": profile})
    except Exception as exc:
        logger.exception("predict_cluster failed")
        return jsonify({"error": f"Prediction error: {exc}"}), 500


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Supply Chain Intelligence — Flask API")
    logger.info("  Dashboard : http://localhost:%s", Config.PORT)
    logger.info("  Models    : lazy-loaded on first prediction request")
    logger.info("=" * 60)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
