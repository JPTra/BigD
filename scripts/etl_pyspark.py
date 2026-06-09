"""
PySpark ETL — the ONLY place PySpark runs.

Pipeline:
    raw_data/*.csv
        -> cleansing
        -> feature engineering (order_month, demand_score, late risk label)
        -> aggregation
        -> processed_data/*.parquet

Outputs (Parquet, never CSV):
    processed_data/agg_market.parquet      market-level sales & late rate
    processed_data/agg_monthly.parquet     monthly sales trend & late rate
    processed_data/agg_shipping.parquet    shipping-mode volume & late rate
    processed_data/training_rf.parquet     per-order rows for Random Forest training
    processed_data/customer_features.parquet  per-customer rows for K-Means training

Run:
    python scripts/etl_pyspark.py
    python scripts/etl_pyspark.py --raw raw_data --out processed_data

This script is the heavy, batch part of the system. It is run offline; the
Flask runtime never imports PySpark.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``config`` importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from config import Config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | etl | %(message)s",
)
logger = logging.getLogger("etl")


# ── Column resolution ────────────────────────────────────────────────────────
# DataCo-style supply chain headers vary slightly between exports; resolve the
# columns we need by trying a few common aliases so the ETL is robust.
COLUMN_ALIASES: dict[str, list[str]] = {
    "shipping_mode": ["Shipping Mode"],
    "scheduled_days": ["Days for shipment (scheduled)", "Days for shipping (scheduled)"],
    "late_risk": ["Late_delivery_risk", "late_delivery_risk"],
    "delivery_status": ["Delivery Status"],
    "market": ["Market"],
    "order_region": ["Order Region"],
    "category": ["Category Name"],
    "quantity": ["Order Item Quantity"],
    "sales": ["Sales"],
    "sales_per_customer": ["Sales per customer"],
    "discount_rate": ["Order Item Discount Rate"],
    "profit_ratio": ["Order Item Profit Ratio"],
    "customer_id": ["Customer Id", "Order Customer Id"],
    "customer_segment": ["Customer Segment"],
    "product_id": ["Product Card Id", "Order Item Cardprod Id"],
    "order_date": ["order date (DateOrders)", "order date (DateOrders) "],
}


def resolve(df: DataFrame, key: str) -> str | None:
    """Return the first aliased column that exists in ``df`` for ``key``."""
    for candidate in COLUMN_ALIASES[key]:
        if candidate in df.columns:
            return candidate
    return None


def create_spark(app_name: str = "SupplyChainETL") -> SparkSession:
    """Build a local Spark session tuned for single-machine batch ETL."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .master("local[*]")
        .getOrCreate()
    )


# ── Extract ──────────────────────────────────────────────────────────────────
def read_raw(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Read every CSV in ``raw_dir`` into one DataFrame."""
    pattern = str(raw_dir / "*.csv")
    logger.info("Reading raw CSV from: %s", pattern)
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("encoding", "latin1")  # DataCo dataset uses latin-1
        .csv(pattern)
    )
    n = df.count()
    if n == 0:
        raise ValueError(f"No rows read from {pattern}. Put the raw CSV in {raw_dir}.")
    logger.info("Loaded %d raw rows, %d columns.", n, len(df.columns))
    return df


# ── Transform: cleansing + feature engineering ───────────────────────────────
def transform(df: DataFrame) -> tuple[DataFrame, dict[str, str | None]]:
    """Cleanse and engineer features used by both aggregations and training.

    Returns the cleansed DataFrame plus the resolved column map (so downstream
    builders know which original columns were found).
    """
    cols = {key: resolve(df, key) for key in COLUMN_ALIASES}

    # --- late delivery label -------------------------------------------------
    if cols["late_risk"]:
        df = df.withColumn("late_label", F.col(cols["late_risk"]).cast("int"))
    elif cols["delivery_status"]:
        df = df.withColumn(
            "late_label",
            F.when(F.col(cols["delivery_status"]) == "Late delivery", 1).otherwise(0),
        )
    else:
        raise ValueError("Cannot derive late-delivery label (need Late_delivery_risk or Delivery Status).")

    # --- order_month ---------------------------------------------------------
    if cols["order_date"]:
        df = df.withColumn(
            "order_ts",
            F.coalesce(
                F.to_timestamp(cols["order_date"], "M/d/yyyy H:mm"),
                F.to_timestamp(cols["order_date"]),
            ),
        ).withColumn("order_month", F.month("order_ts"))
    else:
        df = df.withColumn("order_month", F.lit(1))

    # --- numeric casts + null cleansing -------------------------------------
    numeric_map = {
        "Sales": cols["sales"],
        "Order Item Quantity": cols["quantity"],
        "Order Item Discount Rate": cols["discount_rate"],
        "Order Item Profit Ratio": cols["profit_ratio"],
        "Sales per customer": cols["sales_per_customer"],
        "Days for shipment (scheduled)": cols["scheduled_days"],
    }
    for out_name, src in numeric_map.items():
        if src:
            df = df.withColumn(out_name, F.col(src).cast("double"))
        else:
            df = df.withColumn(out_name, F.lit(0.0))

    # --- categorical passthrough (renamed to the names the models expect) ----
    cat_map = {
        "Shipping Mode": cols["shipping_mode"],
        "Market": cols["market"],
        "Order Region": cols["order_region"],
        "Category Name": cols["category"],
        "Customer Segment": cols["customer_segment"],
    }
    for out_name, src in cat_map.items():
        df = df.withColumn(out_name, F.col(src).cast("string") if src else F.lit("Unknown"))

    # --- demand_score: per-product order frequency (a demand proxy) ----------
    if cols["product_id"]:
        demand = (
            df.groupBy(cols["product_id"])
            .agg(F.count(F.lit(1)).alias("demand_score_raw"))
        )
        df = df.join(demand, on=cols["product_id"], how="left").withColumn(
            "demand_score", F.col("demand_score_raw").cast("double")
        )
    else:
        df = df.withColumn("demand_score", F.lit(0.0))

    # Drop rows missing the essentials.
    df = df.dropna(subset=["Sales", "Shipping Mode", "Market", "late_label"])
    df = df.fillna({"Order Item Quantity": 1.0, "Order Item Discount Rate": 0.0})
    return df, cols


# ── Aggregations ─────────────────────────────────────────────────────────────
def agg_market(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("Market")
        .agg(
            F.count(F.lit(1)).alias("Total_Orders"),
            F.sum("Sales").alias("Total_Sales"),
            F.avg("late_label").alias("Late_Rate"),
        )
        .orderBy(F.desc("Total_Sales"))
    )


def agg_monthly(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("order_month")
        .agg(
            F.count(F.lit(1)).alias("Total_Orders"),
            F.sum("Sales").alias("Total_Sales"),
            F.avg("late_label").alias("Late_Rate"),
        )
        .orderBy("order_month")
    )


def agg_shipping(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("Shipping Mode")
        .agg(
            F.count(F.lit(1)).alias("Total_Orders"),
            F.sum("late_label").alias("Total_Late"),
            F.avg("late_label").alias("Late_Rate"),
            F.avg("Sales").alias("Avg_Sales"),
        )
        .orderBy(F.desc("Total_Orders"))
    )


# ── Training datasets ────────────────────────────────────────────────────────
def training_rf(df: DataFrame) -> DataFrame:
    """Per-order feature table consumed by train_rf.py."""
    return df.select(
        "Days for shipment (scheduled)",
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Item Quantity",
        "Sales",
        "Order Item Discount Rate",
        "order_month",
        "demand_score",
        F.col("late_label").alias("Late_delivery_risk"),
    )


def customer_features(df: DataFrame, cols: dict[str, str | None]) -> DataFrame:
    """Per-customer feature table consumed by train_kmeans.py."""
    if cols.get("customer_id"):
        grp = df.groupBy(cols["customer_id"])
    else:
        grp = df.groupBy("Customer Segment")
    return grp.agg(
        F.avg("Sales per customer").alias("sales_per_customer"),
        F.avg("Order Item Profit Ratio").alias("profit_ratio"),
        F.avg("demand_score").alias("demand_score"),
    ).dropna()


# ── Load (write Parquet) ─────────────────────────────────────────────────────
def write_parquet(df: DataFrame, out_dir: Path, name: str) -> None:
    """Write a (small) aggregation to a single Parquet file via pandas.

    Aggregations are tiny, so collecting to the driver and writing one clean
    ``name.parquet`` file (instead of a Spark part-directory) keeps the runtime
    loader simple — it just reads ``processed_data/name.parquet``.
    """
    pdf = df.toPandas()
    target = out_dir / f"{name}.parquet"
    pdf.to_parquet(target, index=False)
    logger.info("Wrote %s (%d rows) -> %s", name, len(pdf), target)


def write_parquet_spark(df: DataFrame, out_dir: Path, name: str) -> None:
    """Write a (potentially large) training table as a Spark Parquet dataset."""
    target = out_dir / name
    df.write.mode("overwrite").parquet(str(target))
    logger.info("Wrote training dataset %s -> %s", name, target)


# ── Orchestration ────────────────────────────────────────────────────────────
def run(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    spark = create_spark()
    try:
        raw = read_raw(spark, raw_dir)
        clean, cols = transform(raw)
        clean = clean.cache()
        logger.info("Cleansed rows: %d", clean.count())

        # Aggregations -> single-file Parquet (read by Flask at runtime).
        write_parquet(agg_market(clean), out_dir, "agg_market")
        write_parquet(agg_monthly(clean), out_dir, "agg_monthly")
        write_parquet(agg_shipping(clean), out_dir, "agg_shipping")

        # Training tables -> Spark Parquet datasets (read by training scripts).
        write_parquet_spark(training_rf(clean), out_dir, "training_rf.parquet")
        write_parquet_spark(customer_features(clean, cols), out_dir, "customer_features.parquet")

        logger.info("ETL complete.")
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="PySpark ETL for the dashboard.")
    parser.add_argument("--raw", default=str(Config.RAW_DIR), help="Raw data directory.")
    parser.add_argument("--out", default=str(Config.PROCESSED_DIR), help="Output directory.")
    args = parser.parse_args()
    run(Path(args.raw), Path(args.out))


if __name__ == "__main__":
    main()
