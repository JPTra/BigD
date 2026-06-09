"""
Train the PRODUCTION late-delivery risk model — XGBoost.

Chosen by scripts/benchmark_models.py as the best trade-off:
    AUC 0.7365 | Accuracy 0.697 | size 0.42 MB | latency ~3 ms
All deployment targets met (latency < 20 ms, size < 20 MB, acc >= 0.68, AUC >= 0.69).

Input : processed_data/training_rf.parquet  (ETL output)
Output: models/best_xgb_model.pkl            (self-contained sklearn Pipeline)

The artifact is a full Pipeline (OneHotEncoder + XGBClassifier), so the Flask
runtime calls ``.predict_proba(df)`` on raw feature dicts — no manual encoding,
no PySpark.

Run:
    python scripts/train_xgb.py
    python scripts/train_xgb.py --grid     # small hyper-parameter search
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from config import Config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | train_xgb | %(message)s"
)
logger = logging.getLogger("train_xgb")

TARGET = "Late_delivery_risk"
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

# Winning hyper-parameters from the benchmark.
BEST_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss",
)


def load_training_data() -> pd.DataFrame:
    path = Path(Config.PROCESSED_DIR) / "training_rf.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/etl_pyspark.py first.")
    df = pd.read_parquet(path).dropna(subset=[TARGET])
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def build_pipeline(params: dict) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return Pipeline([("pre", pre), ("clf", XGBClassifier(**params))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the production XGBoost risk model.")
    parser.add_argument("--grid", action="store_true", help="Run a small grid search.")
    args = parser.parse_args()

    df = load_training_data()
    x = df[FEATURES]
    y = df[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(BEST_PARAMS)
    t0 = time.perf_counter()
    if args.grid:
        grid = {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [6, 8],
            "clf__learning_rate": [0.05, 0.1],
        }
        logger.info("Grid search: %s", grid)
        search = GridSearchCV(pipeline, grid, cv=3, scoring="roc_auc", n_jobs=-1)
        search.fit(x_train, y_train)
        pipeline = search.best_estimator_
        logger.info("Best params: %s (cv AUC=%.4f)", search.best_params_, search.best_score_)
    else:
        pipeline.fit(x_train, y_train)
    logger.info("Trained in %.2fs", time.perf_counter() - t0)

    proba = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(1)]
    pred = (proba >= 0.5).astype(int)
    logger.info("\n%s", classification_report(y_test, pred, digits=4))
    logger.info(
        "Accuracy=%.4f  AUC=%.4f  F1=%.4f",
        accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        f1_score(y_test, pred),
    )

    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(Config.MODELS_DIR) / Config.RISK_MODEL_FILE
    joblib.dump(pipeline, out, compress=3)
    size_mb = out.stat().st_size / 1024 / 1024
    logger.info("Saved XGBoost pipeline -> %s (%.2f MB)", out, size_mb)


if __name__ == "__main__":
    main()
