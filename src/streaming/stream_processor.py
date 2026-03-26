"""
Stream feature processor — converts Spark micro-batch rows into
RealtimeFeatures and writes them to Redis and Iceberg.

This module bridges the gap between Spark's DataFrame world and our
Pydantic-based feature models. It's called by the foreachBatch sink
in the streaming job.

The processor:
1. Converts Spark Row objects to RealtimeFeatures (using shared definitions)
2. Writes features to Redis for serving
3. Writes features to Iceberg for time-travel snapshots
"""

from datetime import datetime
from typing import Optional

from pyspark.sql import Row

from src.feature_store.redis_store import RedisFeatureStore
from src.iceberg.time_travel import IcebergSnapshotWriter
from src.models.schemas import RealtimeFeatures
from src.utils.logging_config import get_logger

logger = get_logger("streaming.processor")


class StreamFeatureProcessor:
    """Processes streaming micro-batches into feature vectors."""

    def __init__(
        self,
        redis_store: Optional[RedisFeatureStore] = None,
        iceberg_writer: Optional[IcebergSnapshotWriter] = None,
    ):
        self._redis = redis_store or RedisFeatureStore()
        self._iceberg = iceberg_writer or IcebergSnapshotWriter()

    def _row_to_features(self, row: Row) -> RealtimeFeatures:
        """Convert a Spark Row (from windowed aggregation) to RealtimeFeatures.

        The Row contains pre-aggregated values from the Spark windowing
        operation. We map these directly to the feature model fields.

        Args:
            row: Spark Row with columns from compute_windowed_features().

        Returns:
            Populated RealtimeFeatures instance.
        """
        # Extract hour and weekend from the last event timestamp
        last_event_time = row["last_event_time"]
        hour_of_day = last_event_time.hour if last_event_time else 0
        is_weekend = last_event_time.weekday() >= 5 if last_event_time else False

        # Compute velocity: txn_count in the window / window_size_minutes
        # For a 1-hour window, this gives us txn/hour. We normalize to txn/min.
        txn_count = row["txn_count_1h"] or 0
        txn_velocity_5m = txn_count / 60.0  # Rough approximation from 1h window

        return RealtimeFeatures(
            account_id=row["account_id"],
            computed_at=datetime.utcnow(),
            txn_count_1h=txn_count,
            txn_amount_sum_1h=float(row["txn_amount_sum_1h"] or 0),
            txn_amount_max_1h=float(row["txn_amount_max_1h"] or 0),
            txn_velocity_5m=round(txn_velocity_5m, 4),
            distance_from_home=None,  # Requires account profile lookup (separate service)
            is_new_merchant=False,  # Requires merchant history state (separate state store)
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
            is_online=bool(row["last_is_online"]) if row["last_is_online"] is not None else False,
            card_present=bool(row["last_card_present"]) if row["last_card_present"] is not None else True,
            country_code=row["last_country_code"] or "US",
        )

    def process_batch(self, rows: list[Row], batch_id: int) -> int:
        """Process a micro-batch of aggregated rows.

        For each account in the batch:
        1. Convert to RealtimeFeatures
        2. Write to Redis (serving)
        3. Buffer for Iceberg snapshot (time-travel)

        Args:
            rows: List of Spark Row objects from the micro-batch.
            batch_id: Spark streaming batch identifier.

        Returns:
            Number of feature vectors successfully written.
        """
        features_written = 0
        snapshot_records = []

        for row in rows:
            try:
                features = self._row_to_features(row)

                # Write to Redis for real-time serving
                self._redis.write_realtime_features(features)

                # Buffer for Iceberg snapshot
                snapshot_records.append(features.model_dump(mode="json"))

                features_written += 1

            except Exception as e:
                logger.error(
                    "row_processing_failed",
                    account_id=row["account_id"] if "account_id" in row else "unknown",
                    batch_id=batch_id,
                    error=str(e),
                )
                # Continue processing other rows — don't let one bad row
                # block the entire batch

        # Write batch of snapshots to Iceberg for time-travel
        if snapshot_records:
            try:
                self._iceberg.write_snapshot(
                    records=snapshot_records,
                    table_name="realtime_feature_snapshots",
                    batch_id=batch_id,
                )
            except Exception as e:
                # Iceberg write failure is non-fatal — features are already
                # in Redis for serving. We log an error for investigation
                # but don't fail the batch.
                logger.error(
                    "iceberg_snapshot_write_failed",
                    batch_id=batch_id,
                    num_records=len(snapshot_records),
                    error=str(e),
                )

        return features_written
