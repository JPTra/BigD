"""
Train the K-Means customer-segmentation model.

Input : processed_data/customer_features.parquet  (produced by etl_pyspark.py)
Output: models/best_kmeans_model.pkl
        models/best_kmeans_scaler.pkl

Features (in this exact order — the API sends them the same way):
    [sales_per_customer, profit_ratio, demand_score]

The scaler is exported separately so the runtime applies the identical
transform before ``model.predict``. No Spark is used here.

Run:
    python scripts/train_kmeans.py
    python scripts/train_kmeans.py --k 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import Config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | train_kmeans | %(message)s"
)
logger = logging.getLogger("train_kmeans")

FEATURES = ["sales_per_customer", "profit_ratio", "demand_score"]


def load_features() -> pd.DataFrame:
    path = Path(Config.PROCESSED_DIR) / "customer_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/etl_pyspark.py first.")
    df = pd.read_parquet(path)[FEATURES].dropna()
    logger.info("Loaded %d customer rows from %s", len(df), path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the K-Means segmentation model.")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters.")
    args = parser.parse_args()

    df = load_features()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[FEATURES])

    model = KMeans(n_clusters=args.k, n_init=10, random_state=42)
    labels = model.fit_predict(x_scaled)

    if len(set(labels)) > 1:
        score = silhouette_score(x_scaled, labels)
        logger.info("Silhouette score (k=%d): %.4f", args.k, score)
    logger.info("Cluster sizes: %s", pd.Series(labels).value_counts().to_dict())

    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_out = Path(Config.MODELS_DIR) / Config.KMEANS_MODEL_FILE
    scaler_out = Path(Config.MODELS_DIR) / Config.KMEANS_SCALER_FILE
    joblib.dump(model, model_out, compress=3)
    joblib.dump(scaler, scaler_out, compress=3)
    logger.info("Saved K-Means model  -> %s", model_out)
    logger.info("Saved K-Means scaler -> %s", scaler_out)


if __name__ == "__main__":
    main()
