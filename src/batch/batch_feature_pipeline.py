"""
Batch feature pipeline — the cold path of the feature store.

This pipeline runs nightly via Airflow and computes expensive aggregate
features (30d/90d windows, percentiles) that are too costly for real-time
computation. Results are written to Snowflake, snapshotted in Iceberg,
and backfilled to Redis.

Pipeline stages:
1. Run dbt models in Snowflake (transforms raw → features)
2. Read computed features from Snowflake
3. Write snapshot to Iceberg (time-travel)
4. Backfill features to Redis (serving)

Processing 1M+ events/day for feature computation requires scanning
~30-90 days of history per account, which Snowflake handles efficiently
with its columnar storage and auto-scaling compute.
"""

import subprocess
import time
from datetime import datetime, date
from typing import Optional

from src.batch.snowflake_client import SnowflakeClient
from src.feature_store.redis_store import RedisFeatureStore
from src.iceberg.time_travel import IcebergSnapshotWriter
from src.models.schemas import BatchFeatures
from src.utils.config import get_snowflake_config
from src.utils.logging_config import get_logger

logger = get_logger("batch.pipeline")


class BatchFeaturePipeline:
    """Orchestrates the nightly batch feature computation."""

    def __init__(
        self,
        snowflake_client: Optional[SnowflakeClient] = None,
        redis_store: Optional[RedisFeatureStore] = None,
        iceberg_writer: Optional[IcebergSnapshotWriter] = None,
    ):
        self._sf = snowflake_client or SnowflakeClient()
        self._redis = redis_store or RedisFeatureStore()
        self._iceberg = iceberg_writer or IcebergSnapshotWriter()
        self._config = get_snowflake_config()

    def run_dbt_models(self, target_date: Optional[str] = None) -> bool:
        """Execute dbt models to transform raw data into features in Snowflake.

        dbt handles the SQL transformations (staging → feature tables).
        We shell out to the dbt CLI rather than using dbt's Python API
        because the CLI is more stable and better documented.

        Args:
            target_date: Date to compute features for (YYYY-MM-DD).
                        Defaults to yesterday.

        Returns:
            True if dbt run succeeded, False otherwise.
        """
        target_date = target_date or str(date.today())
        dbt_config = self._config.get("dbt", {})
        project_dir = dbt_config.get("project_dir", "./sql/dbt_models")
        profiles_dir = dbt_config.get("profiles_dir", "~/.dbt")
        target = dbt_config.get("target", "prod")

        cmd = [
            "dbt", "run",
            "--project-dir", project_dir,
            "--profiles-dir", profiles_dir,
            "--target", target,
            "--vars", f'{{"target_date": "{target_date}"}}',
        ]

        logger.info("dbt_run_starting", target_date=target_date, command=" ".join(cmd))
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )

            elapsed = time.monotonic() - start

            if result.returncode == 0:
                logger.info(
                    "dbt_run_succeeded",
                    target_date=target_date,
                    elapsed_seconds=round(elapsed, 2),
                )
                return True
            else:
                logger.error(
                    "dbt_run_failed",
                    target_date=target_date,
                    returncode=result.returncode,
                    stderr=result.stderr[-1000:],  # Last 1000 chars of error
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("dbt_run_timeout", target_date=target_date)
            return False
        except FileNotFoundError:
            logger.error("dbt_not_installed")
            return False

    def fetch_batch_features(self, target_date: Optional[str] = None) -> list[BatchFeatures]:
        """Read computed batch features from Snowflake.

        Queries the feature table populated by dbt and converts rows
        to BatchFeatures Pydantic models for type-safe downstream use.

        Args:
            target_date: Date to fetch features for (YYYY-MM-DD).

        Returns:
            List of BatchFeatures for all accounts.
        """
        target_date = target_date or str(date.today())

        sql = f"""
        SELECT
            account_id,
            computed_at,
            avg_txn_amount_30d,
            txn_count_30d,
            distinct_merchants_30d,
            distinct_merchants_7d,
            txn_count_7d,
            txn_amount_p50_90d,
            txn_amount_p95_90d,
            txn_amount_p99_90d,
            days_since_last_txn,
            avg_daily_txn_count_30d,
            most_common_merchant_category,
            pct_online_txns_30d
        FROM feature_store.account_features_daily
        WHERE computation_date = '{target_date}'
        """

        logger.info("fetching_batch_features", target_date=target_date)
        start = time.monotonic()

        features_list = []
        for batch in self._sf.execute_query_iter(sql, batch_size=10000):
            for row in batch:
                try:
                    features = BatchFeatures(
                        account_id=row["account_id"],
                        computed_at=row.get("computed_at", datetime.utcnow()),
                        avg_txn_amount_30d=float(row.get("avg_txn_amount_30d", 0)),
                        txn_count_30d=int(row.get("txn_count_30d", 0)),
                        distinct_merchants_30d=int(row.get("distinct_merchants_30d", 0)),
                        distinct_merchants_7d=int(row.get("distinct_merchants_7d", 0)),
                        txn_count_7d=int(row.get("txn_count_7d", 0)),
                        txn_amount_p50_90d=float(row.get("txn_amount_p50_90d", 0)),
                        txn_amount_p95_90d=float(row.get("txn_amount_p95_90d", 0)),
                        txn_amount_p99_90d=float(row.get("txn_amount_p99_90d", 0)),
                        days_since_last_txn=float(row.get("days_since_last_txn", 0)),
                        avg_daily_txn_count_30d=float(row.get("avg_daily_txn_count_30d", 0)),
                        most_common_merchant_category=row.get(
                            "most_common_merchant_category", "unknown"
                        ),
                        pct_online_txns_30d=float(row.get("pct_online_txns_30d", 0)),
                    )
                    features_list.append(features)
                except Exception as e:
                    logger.warning(
                        "batch_feature_parse_error",
                        account_id=row.get("account_id", "unknown"),
                        error=str(e),
                    )

        elapsed = time.monotonic() - start
        logger.info(
            "batch_features_fetched",
            target_date=target_date,
            num_accounts=len(features_list),
            elapsed_seconds=round(elapsed, 2),
        )
        return features_list

    def snapshot_to_iceberg(self, features: list[BatchFeatures]) -> bool:
        """Write batch features to Iceberg for time-travel.

        This creates a snapshot of today's batch features, enabling
        historical queries like "what were account X's batch features
        on January 15th?"
        """
        records = [f.model_dump(mode="json") for f in features]
        success = self._iceberg.write_snapshot(
            records=records,
            table_name="batch_feature_snapshots",
        )
        if success:
            logger.info("iceberg_batch_snapshot_written", num_records=len(records))
        return success

    def backfill_redis(self, features: list[BatchFeatures]) -> int:
        """Bulk-write batch features to Redis for serving.

        Uses Redis pipelines for efficient bulk writes. This is the
        final stage — after this, the ML model can access updated
        batch features via the Redis feature store.
        """
        logger.info("redis_backfill_starting", num_features=len(features))
        start = time.monotonic()

        written = self._redis.write_batch_features_bulk(features)

        elapsed = time.monotonic() - start
        logger.info(
            "redis_backfill_complete",
            num_written=written,
            elapsed_seconds=round(elapsed, 2),
        )
        return written

    def run(self, target_date: Optional[str] = None) -> dict:
        """Execute the full batch pipeline.

        Stages:
        1. dbt models (Snowflake transforms)
        2. Fetch features from Snowflake
        3. Snapshot to Iceberg
        4. Backfill to Redis

        Returns a summary dict with timing and counts.
        """
        target_date = target_date or str(date.today())
        pipeline_start = time.monotonic()
        summary = {
            "target_date": target_date,
            "status": "success",
            "stages": {},
        }

        # Stage 1: dbt
        logger.info("pipeline_stage_dbt", target_date=target_date)
        dbt_start = time.monotonic()
        dbt_success = self.run_dbt_models(target_date)
        summary["stages"]["dbt"] = {
            "success": dbt_success,
            "elapsed_seconds": round(time.monotonic() - dbt_start, 2),
        }
        if not dbt_success:
            summary["status"] = "failed"
            summary["failure_stage"] = "dbt"
            logger.error("pipeline_failed_at_dbt", target_date=target_date)
            return summary

        # Stage 2: Fetch from Snowflake
        logger.info("pipeline_stage_fetch", target_date=target_date)
        fetch_start = time.monotonic()
        features = self.fetch_batch_features(target_date)
        summary["stages"]["fetch"] = {
            "num_accounts": len(features),
            "elapsed_seconds": round(time.monotonic() - fetch_start, 2),
        }

        if not features:
            summary["status"] = "warning"
            summary["warning"] = "No features computed"
            logger.warning("pipeline_no_features", target_date=target_date)
            return summary

        # Stage 3: Iceberg snapshot
        logger.info("pipeline_stage_iceberg", target_date=target_date)
        iceberg_start = time.monotonic()
        iceberg_success = self.snapshot_to_iceberg(features)
        summary["stages"]["iceberg"] = {
            "success": iceberg_success,
            "elapsed_seconds": round(time.monotonic() - iceberg_start, 2),
        }

        # Stage 4: Redis backfill
        logger.info("pipeline_stage_redis", target_date=target_date)
        redis_start = time.monotonic()
        redis_written = self.backfill_redis(features)
        summary["stages"]["redis"] = {
            "num_written": redis_written,
            "elapsed_seconds": round(time.monotonic() - redis_start, 2),
        }

        summary["total_elapsed_seconds"] = round(
            time.monotonic() - pipeline_start, 2
        )
        logger.info("pipeline_complete", **summary)
        return summary
