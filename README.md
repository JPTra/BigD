# Supply Chain Intelligence Dashboard

A fast, deployable Flask + PySpark supply-chain analytics dashboard.

**Core design principle: PySpark is used _only_ offline (ETL & training). The
Flask runtime is pure pandas + scikit-learn — no Spark session is ever created
when a user makes a prediction.** This removes the Spark start-up bottleneck and
makes the dashboard responsive both locally and in the cloud.

---

## Folder structure

```
project/
│
├── raw_data/                 # INPUT: raw source CSV(s) — e.g. DataCo supply chain
│
├── processed_data/           # OUTPUT of ETL — Parquet only (never CSV)
│   ├── agg_market.parquet         # market sales & late rate
│   ├── agg_monthly.parquet        # monthly trend
│   ├── agg_shipping.parquet       # shipping-mode performance
│   ├── training_rf.parquet/       # per-order training table (Spark dir)
│   └── customer_features.parquet/ # per-customer training table (Spark dir)
│
├── models/                   # Trained, exported models (lazy-loaded by Flask)
│   ├── best_xgb_model.pkl         # sklearn Pipeline (OneHotEncoder + XGBoost)
│   ├── best_xgb_model.meta.json   # model contract (features, versions, metrics)
│   ├── best_kmeans_model.pkl      # sklearn KMeans
│   └── best_kmeans_scaler.pkl     # StandardScaler for KMeans
│
├── services/                 # Runtime service layer (imported by app.py)
│   ├── data_loader.py             # auto-discover + cache Parquet
│   ├── model_loader.py            # lazy, singleton model loading
│   └── cache.py                   # flask-caching + mtime helpers
│
├── scripts/                  # Offline batch jobs (PySpark lives here)
│   ├── etl_pyspark.py             # raw → cleansing → features → Parquet
│   ├── train_xgb.py               # train + export best_xgb_model.pkl (production)
│   ├── benchmark_models.py        # compare 5 models (dev only)
│   ├── export_model.py            # compress + validate + write meta.json
│   └── train_kmeans.py            # train + export best_kmeans_model.pkl
│
├── frontend/
│   └── dashboard.html             # HTML + Tailwind + Chart.js
│
├── config.py                 # configuration class (env-var driven)
├── app.py                    # Flask API — routing only (lightweight)
├── requirements.txt          # runtime only (pinned inference stack)
├── requirements-dev.txt      # + pyspark/lightgbm/psutil for ETL/training/benchmark
└── README.md
```

---

## Data flow

```
raw_data/ (CSV)
   │
   ▼  scripts/etl_pyspark.py        ← PySpark: cleansing, feature engineering, aggregation
processed_data/ (Parquet)
   │
   ├─► scripts/train_xgb.py    ──► models/best_xgb_model.pkl  (XGBoost, 0.35 MB)
   └─► scripts/train_kmeans.py ──► models/best_kmeans_model.pkl + best_kmeans_scaler.pkl
   │
   ▼  app.py  (Flask, pandas + sklearn only — NO Spark)
   │   • /api/data  reads Parquet via cached data_loader
   │   • /api/predict_* uses lazily-loaded models
   ▼
frontend/dashboard.html  (Tailwind + Chart.js)
```

The runtime path (`app.py → services → Parquet/models`) never imports PySpark.

---

## Quick start

### 1. Install runtime dependencies

```bash
cd project
pip install -r requirements.txt
```

> `requirements.txt` is the lightweight **runtime** set (no PySpark). The
> offline ETL/training/benchmark tools need extra libs — install those only on
> the build machine with `pip install -r requirements-dev.txt`.
>
> The model-inference libraries (numpy, scipy, scikit-learn, xgboost) are
> **version-pinned** to the training environment so the `.pkl` files unpickle
> reliably on any machine. Re-pin them if you re-train.

### 2. (Offline) Build data & models

```bash
pip install -r requirements-dev.txt    # adds pyspark/lightgbm/psutil
# 1. Put your raw CSV in raw_data/
python scripts/etl_pyspark.py          # → processed_data/*.parquet
python scripts/train_xgb.py            # → models/best_xgb_model.pkl
python scripts/export_model.py         # compress + validate the artifact
python scripts/train_kmeans.py         # → models/best_kmeans_model.pkl (+scaler)
```

### 3. Run the dashboard

```bash
python app.py
# open http://localhost:5000
```

Startup is < 3 s because **no model and no Spark session is loaded at boot** —
models load lazily on the first prediction request and stay in memory.

---

## API

| Method | Endpoint               | Description                                   |
|--------|------------------------|-----------------------------------------------|
| GET    | `/`                    | Serve the dashboard                           |
| GET    | `/api/health`          | Liveness probe                                |
| GET    | `/api/data`            | market + monthly + shipping aggregations (cached) |
| POST   | `/api/predict_risk`    | XGBoost late-delivery risk (0.35 MB, ~3 ms)   |
| POST   | `/api/predict_cluster` | K-Means customer segment                      |

### `POST /api/predict_risk`

```json
{
  "shipping_mode": "Standard Class",
  "market": "LATAM",
  "order_region": "Southeast Asia",
  "order_month": 6,
  "sales": 150.0,
  "quantity": 1,
  "discount_rate": 0.1,
  "demand_score": 25.0
}
```

### `POST /api/predict_cluster`

```json
{ "sales_per_customer": 180.0, "profit_ratio": 0.12, "demand_score": 15.0 }
```

---

## Performance design

| Requirement              | How it is met                                                        |
|--------------------------|---------------------------------------------------------------------|
| No Spark at runtime      | PySpark imported only in `scripts/`; runtime uses pandas + sklearn   |
| Prediction < 100 ms      | Models held in memory after first load; plain sklearn inference      |
| Startup < 3 s            | Lazy model loading — nothing heavy at import time                    |
| Fast `/api/data`         | `flask-caching` on the endpoint + mtime-keyed Parquet read cache     |
| Parquet, not CSV         | ETL writes Parquet; loader reads Parquet                             |
| Auto-detect new data     | `data_loader` globs `processed_data/`; cache re-reads on mtime change|

---

## Configuration (env vars)

All settings live in `config.py` (`Config` class) and can be overridden via
environment variables or a `.env` file:

| Variable                | Default          | Purpose                       |
|-------------------------|------------------|-------------------------------|
| `PORT`                  | `5000`           | Server port                   |
| `HOST`                  | `0.0.0.0`        | Bind address                  |
| `RAW_DIR`               | `raw_data`       | Raw input directory           |
| `PROCESSED_DIR`         | `processed_data` | Parquet output directory      |
| `MODELS_DIR`            | `models`         | Model directory               |
| `CACHE_TYPE`            | `SimpleCache`    | flask-caching backend         |
| `CACHE_DEFAULT_TIMEOUT` | `300`            | `/api/data` cache TTL (s)     |
| `LOG_LEVEL`             | `INFO`           | Logging verbosity             |
| `FLASK_DEBUG`           | `false`          | Flask debug mode              |
