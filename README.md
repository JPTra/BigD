# Supply Chain Intelligence Dashboard

A fast, deployable Flask + PySpark supply-chain analytics dashboard.

**Core design principle: PySpark is used _only_ offline (ETL & training). The
Flask runtime is pure pandas + scikit-learn — no Spark session is ever created
when a user makes a prediction.** This removes the Spark start-up bottleneck and
makes the dashboard responsive both locally and in the cloud.

