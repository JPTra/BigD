"""
Benchmark 5 tree-based classifiers for the late-delivery risk task and pick the
best one for Flask production deployment.

Models compared:
    RandomForest, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting

Metrics measured (per model):
    1. Training time            6. F1 score
    2. Accuracy                 7. File size (MB)
    3. AUC                      8. RAM usage (MB, fresh-process load)
    4. Precision                9. Prediction latency (ms, single row)
    5. Recall

Targets for deployment:
    latency < 20 ms | size < 20 MB | accuracy >= 0.68 | AUC >= 0.69

Input : processed_data/training_rf.parquet  (ETL output)
Output: prints a comparison table + ranking + recommendation, and writes the
        benchmarked pipelines to models/benchmark/ for inspection.

Run:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --sample 80000   # faster trial run
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from config import Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("benchmark")

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
BENCH_DIR = Config.MODELS_DIR / "benchmark"


def build_models() -> dict:
    """Return the 5 candidate estimators, tuned for small size + fast inference."""
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=42,
            eval_metric="logloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1, random_state=42, verbose=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None, random_state=42
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [
            # Dense output: HistGradientBoosting requires dense X, and the
            # categorical cardinality here is low (~32 cols) so it's cheap.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def measure_load_ram_mb(model_path: Path) -> float:
    """Load the model in a *fresh* Python process and report the RSS delta (MB).

    A separate process gives a clean, deployment-realistic memory figure that
    isn't polluted by training artefacts left in this process.
    """
    code = (
        "import psutil,os,joblib,sys;"
        "p=psutil.Process(os.getpid());"
        "b=p.memory_info().rss;"
        "m=joblib.load(sys.argv[1]);"
        "a=p.memory_info().rss;"
        "print((a-b)/1024/1024)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning("RAM measure failed: %s", out.stderr[-300:])
        return float("nan")


def measure_latency_ms(pipeline: Pipeline, sample_row: pd.DataFrame, n: int = 300) -> float:
    """Median single-row predict_proba latency in ms (the Flask use case)."""
    # warmup
    for _ in range(20):
        pipeline.predict_proba(sample_row)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        pipeline.predict_proba(sample_row)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate models.")
    parser.add_argument("--sample", type=int, default=0, help="Subsample N rows (0 = all).")
    args = parser.parse_args()

    path = Config.PROCESSED_DIR / "training_rf.parquet"
    df = pd.read_parquet(path)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42)
    logger.info("Loaded %d rows from %s", len(df), path)

    x = df[CATEGORICAL + NUMERIC]
    y = df[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    sample_row = x_test.iloc[[0]].copy()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, estimator in build_models().items():
        logger.info("── Training %s ...", name)
        pipe = make_pipeline(estimator)

        t = time.perf_counter()
        pipe.fit(x_train, y_train)
        train_time = time.perf_counter() - t

        proba = pipe.predict_proba(x_test)[:, list(pipe.classes_).index(1)]
        pred = (proba >= 0.5).astype(int)

        model_path = BENCH_DIR / f"{name}.pkl"
        joblib.dump(pipe, model_path, compress=3)
        size_mb = model_path.stat().st_size / 1024 / 1024

        rows.append(
            {
                "Model": name,
                "TrainTime_s": round(train_time, 2),
                "Accuracy": round(accuracy_score(y_test, pred), 4),
                "AUC": round(roc_auc_score(y_test, proba), 4),
                "Precision": round(precision_score(y_test, pred), 4),
                "Recall": round(recall_score(y_test, pred), 4),
                "F1": round(f1_score(y_test, pred), 4),
                "Size_MB": round(size_mb, 2),
                "RAM_MB": round(measure_load_ram_mb(model_path), 1),
                "Latency_ms": round(measure_latency_ms(pipe, sample_row), 3),
            }
        )
        del pipe
        gc.collect()

    res = pd.DataFrame(rows)

    # ── Ranking: meets-targets first, then a weighted score ──────────────────
    def meets(r) -> bool:
        return (
            r["Latency_ms"] < 20
            and r["Size_MB"] < 20
            and r["Accuracy"] >= 0.68
            and r["AUC"] >= 0.69
        )

    res["MeetsTargets"] = res.apply(meets, axis=1)

    # Normalised score: reward AUC/Acc/F1, penalise size/latency.
    def norm(col, invert=False):
        v = res[col].astype(float)
        lo, hi = v.min(), v.max()
        if hi == lo:
            s = pd.Series(1.0, index=v.index)
        else:
            s = (v - lo) / (hi - lo)
        return 1 - s if invert else s

    res["Score"] = (
        0.30 * norm("AUC")
        + 0.25 * norm("Accuracy")
        + 0.15 * norm("F1")
        + 0.15 * norm("Latency_ms", invert=True)
        + 0.10 * norm("Size_MB", invert=True)
        + 0.05 * norm("TrainTime_s", invert=True)
    ).round(4)

    res = res.sort_values(["MeetsTargets", "Score"], ascending=[False, False]).reset_index(drop=True)
    res.insert(0, "Rank", res.index + 1)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print(f"BENCHMARK RESULTS — late-delivery risk (test set = 20 pct, n_test = {len(y_test)})")
    print("=" * 100)
    print(res.to_string(index=False))

    winner = res.iloc[0]
    print("\n" + "=" * 100)
    print(f"RECOMMENDED FOR DEPLOYMENT: {winner['Model']}")
    print("=" * 100)
    print(
        f"  Accuracy={winner['Accuracy']}  AUC={winner['AUC']}  F1={winner['F1']}  "
        f"Size={winner['Size_MB']}MB  Latency={winner['Latency_ms']}ms  "
        f"MeetsTargets={'YES' if winner['MeetsTargets'] else 'NO'}"
    )

    # Persist machine-readable results.
    (BENCH_DIR / "benchmark_results.json").write_text(
        json.dumps(res.to_dict(orient="records"), indent=2)
    )
    res.to_csv(BENCH_DIR / "benchmark_results.csv", index=False)
    logger.info("Saved results to %s", BENCH_DIR)


if __name__ == "__main__":
    main()
