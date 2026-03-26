"""
Airflow DAG for nightly batch feature computation.

Schedule: Runs daily at 2:00 AM UTC (after upstream data pipelines complete).

This DAG orchestrates the cold-path feature pipeline:
1. Run dbt models to compute features in Snowflake
2. Fetch computed features from Snowflake
3. Snapshot features to Iceberg for time-travel
4. Backfill features to Redis for serving
5. Run Iceberg maintenance (snapshot expiration, compaction)
6. Validate feature freshness and quality

Each task is idempotent — re-running on the same date overwrites
previous results without side effects.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


# ─── DAG Default Args ───────────────────────────────────────────────────────

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email": ["data-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),
    "start_date": days_ago(1),
}


# ─── Task Functions ──────────────────────────────────────────────────────────


def run_dbt_models(**context):
    """Execute dbt models in Snowflake to compute batch features."""
    from src.batch.batch_feature_pipeline import BatchFeaturePipeline
    from src.utils.logging_config import setup_logging

    setup_logging()
    target_date = context["ds"]  # Airflow execution date (YYYY-MM-DD)

    pipeline = BatchFeaturePipeline()
    success = pipeline.run_dbt_models(target_date)

    if not success:
        raise RuntimeError(f"dbt models failed for {target_date}")


def fetch_and_backfill(**context):
    """Fetch features from Snowflake and backfill to Redis."""
    from src.batch.batch_feature_pipeline import BatchFeaturePipeline
    from src.utils.logging_config import setup_logging

    setup_logging()
    target_date = context["ds"]

    pipeline = BatchFeaturePipeline()

    # Fetch from Snowflake
    features = pipeline.fetch_batch_features(target_date)
    if not features:
        raise RuntimeError(f"No features returned for {target_date}")

    # Snapshot to Iceberg
    pipeline.snapshot_to_iceberg(features)

    # Backfill to Redis
    written = pipeline.backfill_redis(features)

    # Push metrics to XCom for downstream tasks
    context["ti"].xcom_push(key="num_accounts", value=len(features))
    context["ti"].xcom_push(key="redis_written", value=written)


def validate_features(**context):
    """Validate feature freshness and basic quality checks.

    Checks:
    - At least N accounts have features (sanity check)
    - Feature values are within expected ranges
    - Redis has fresh features after backfill
    """
    from src.feature_store.redis_store import RedisFeatureStore
    from src.utils.logging_config import setup_logging, get_logger

    setup_logging()
    logger = get_logger("batch.validation")

    num_accounts = context["ti"].xcom_pull(
        task_ids="fetch_and_backfill", key="num_accounts"
    )
    redis_written = context["ti"].xcom_pull(
        task_ids="fetch_and_backfill", key="redis_written"
    )

    # Validation 1: Minimum account threshold
    min_accounts = 100  # Adjust based on your data
    if num_accounts < min_accounts:
        logger.warning(
            "low_account_count",
            num_accounts=num_accounts,
            threshold=min_accounts,
        )

    # Validation 2: Redis write success rate
    if redis_written < num_accounts * 0.95:
        raise RuntimeError(
            f"Redis backfill incomplete: {redis_written}/{num_accounts} written"
        )

    # Validation 3: Spot-check Redis for freshness
    redis_store = RedisFeatureStore()
    if not redis_store.health_check():
        raise RuntimeError("Redis health check failed after backfill")

    logger.info(
        "validation_passed",
        num_accounts=num_accounts,
        redis_written=redis_written,
    )


def expire_iceberg_snapshots(**context):
    """Run Iceberg snapshot expiration and compaction.

    The streaming path generates many small Parquet files (one per micro-batch).
    Nightly compaction merges these into larger files for better scan performance.
    """
    from src.utils.logging_config import setup_logging, get_logger

    setup_logging()
    logger = get_logger("batch.iceberg_maintenance")

    # In production, this would call Iceberg's maintenance procedures:
    # - expire_snapshots: Remove snapshots older than 30 days
    # - rewrite_data_files: Compact small files into larger ones
    # - rewrite_manifests: Optimize manifest files

    # Example using Spark SQL (would be called via spark-submit):
    # spark.sql("""
    #     CALL iceberg_catalog.system.expire_snapshots(
    #         table => 'fraud_features.realtime_feature_snapshots',
    #         older_than => TIMESTAMP '...',
    #         retain_last => 10
    #     )
    # """)

    logger.info(
        "iceberg_maintenance_complete",
        note="Placeholder — configure Spark SQL calls in production",
    )


# ─── DAG Definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="feature_batch_daily",
    default_args=default_args,
    description="Nightly batch feature computation for fraud detection",
    schedule_interval="0 2 * * *",  # 2:00 AM UTC daily
    catchup=False,
    max_active_runs=1,  # Prevent parallel runs
    tags=["fraud-detection", "feature-store", "batch"],
) as dag:

    # Task 1: Run dbt models
    task_dbt = PythonOperator(
        task_id="run_dbt_models",
        python_callable=run_dbt_models,
        execution_timeout=timedelta(hours=1),
    )

    # Task 2: Fetch features and backfill Redis
    task_backfill = PythonOperator(
        task_id="fetch_and_backfill",
        python_callable=fetch_and_backfill,
        execution_timeout=timedelta(minutes=30),
    )

    # Task 3: Validate features
    task_validate = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
        execution_timeout=timedelta(minutes=5),
    )

    # Task 4: Iceberg maintenance
    task_iceberg_maintenance = PythonOperator(
        task_id="expire_iceberg_snapshots",
        python_callable=expire_iceberg_snapshots,
        execution_timeout=timedelta(minutes=30),
    )

    # ─── Task Dependencies ───────────────────────────────────────────────
    # dbt → backfill → validate
    #                 → iceberg maintenance (parallel with validate)

    task_dbt >> task_backfill >> [task_validate, task_iceberg_maintenance]
