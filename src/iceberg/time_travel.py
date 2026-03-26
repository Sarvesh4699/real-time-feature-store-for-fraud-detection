"""
Apache Iceberg time-travel queries and snapshot writer.

Iceberg's time-travel capability is the foundation of our training-serving
skew debugging story. Every feature computation (both streaming and batch)
writes a snapshot to Iceberg. When investigating why a model made a
particular prediction, we can query:

    SELECT * FROM features AS OF TIMESTAMP '2024-01-15 10:30:00'
    WHERE account_id = 'acct_12345'

This returns the EXACT feature vector the model saw at prediction time,
eliminating hours of manual feature reconstruction.

Design trade-offs:
- We write to Iceberg on every micro-batch (~10s), which generates many
  small files. The nightly batch DAG runs Iceberg's compaction to merge
  these into larger, more efficient files.
- Snapshots are retained for 30 days. Older snapshots are expired but
  the underlying Parquet files are retained per data retention policy.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from src.utils.config import get_iceberg_config
from src.utils.logging_config import get_logger

logger = get_logger("iceberg.time_travel")


class IcebergSnapshotWriter:
    """Writes feature snapshots to Iceberg tables for time-travel queries.

    In production, this uses PyIceberg or Spark SQL to write to an Iceberg
    catalog. This implementation abstracts the write interface so it can
    be swapped between PyIceberg (for lightweight writes) and Spark
    (for bulk writes in the streaming job).
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_iceberg_config()
        self._catalog = None  # Lazy-initialized
        self._namespace = self._config.get("namespace", "fraud_features")

    def _get_catalog(self):
        """Initialize the Iceberg catalog connection.

        Uses PyIceberg's REST catalog by default. In production, this
        might connect to AWS Glue, Hive Metastore, or Nessie.
        """
        if self._catalog is None:
            try:
                from pyiceberg.catalog import load_catalog

                catalog_config = self._config.get("catalog", {})
                self._catalog = load_catalog(
                    name="fraud_features",
                    **{
                        "type": catalog_config.get("type", "rest"),
                        "uri": catalog_config.get("uri", "http://localhost:8181"),
                        "warehouse": catalog_config.get("warehouse", ""),
                    },
                )
                logger.info(
                    "iceberg_catalog_connected",
                    catalog_type=catalog_config.get("type"),
                )
            except ImportError:
                logger.warning(
                    "pyiceberg_not_available",
                    msg="PyIceberg not installed — snapshot writes will be no-ops",
                )
            except Exception as e:
                logger.warning(
                    "iceberg_catalog_connection_failed",
                    error=str(e),
                    msg="Snapshot writes will be buffered locally",
                )
        return self._catalog

    def write_snapshot(
        self,
        records: list[dict],
        table_name: str,
        batch_id: Optional[int] = None,
    ) -> bool:
        """Write a batch of feature records as an Iceberg snapshot.

        Each call creates a new snapshot in the Iceberg table. The snapshot
        is automatically versioned by Iceberg's commit protocol, which
        enables time-travel queries against any past snapshot.

        Args:
            records: List of feature dicts to write.
            table_name: Target Iceberg table name.
            batch_id: Optional batch identifier for logging.

        Returns:
            True if the write succeeded, False otherwise.
        """
        if not records:
            return True

        catalog = self._get_catalog()

        if catalog is None:
            # Fallback: write to local Parquet for development/testing
            return self._write_local_fallback(records, table_name, batch_id)

        try:
            import pyarrow as pa

            table_identifier = f"{self._namespace}.{table_name}"

            # Convert records to PyArrow table
            arrow_table = pa.Table.from_pylist(records)

            # Load or create the Iceberg table
            try:
                iceberg_table = catalog.load_table(table_identifier)
            except Exception:
                # Table doesn't exist — create it from the Arrow schema
                iceberg_table = catalog.create_table(
                    identifier=table_identifier,
                    schema=arrow_table.schema,
                )
                logger.info("iceberg_table_created", table=table_identifier)

            # Append data — this creates a new snapshot
            iceberg_table.append(arrow_table)

            logger.debug(
                "iceberg_snapshot_written",
                table=table_identifier,
                num_records=len(records),
                batch_id=batch_id,
            )
            return True

        except Exception as e:
            logger.error(
                "iceberg_write_failed",
                table=table_name,
                num_records=len(records),
                batch_id=batch_id,
                error=str(e),
            )
            return False

    def _write_local_fallback(
        self,
        records: list[dict],
        table_name: str,
        batch_id: Optional[int],
    ) -> bool:
        """Write snapshots to local Parquet files when Iceberg is unavailable.

        This is a development fallback. In production, Iceberg writes are
        required for time-travel functionality.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            from pathlib import Path

            output_dir = Path("/tmp/iceberg-local-fallback") / table_name
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}_batch{batch_id or 0}.parquet"

            arrow_table = pa.Table.from_pylist(records)
            pq.write_table(arrow_table, output_dir / filename)

            logger.debug(
                "local_fallback_snapshot_written",
                path=str(output_dir / filename),
                num_records=len(records),
            )
            return True
        except Exception as e:
            logger.error("local_fallback_write_failed", error=str(e))
            return False


class IcebergTimeTravel:
    """Query historical feature snapshots using Iceberg time-travel.

    This is the key class for debugging training-serving skew. Given a
    prediction timestamp and account_id, it retrieves the exact feature
    vector the model would have seen.

    Usage:
        tt = IcebergTimeTravel()
        features = tt.get_features_at(
            account_id="acct_12345",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            table_name="realtime_feature_snapshots"
        )
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_iceberg_config()
        self._namespace = self._config.get("namespace", "fraud_features")
        self._catalog = None

    def _get_catalog(self):
        """Initialize Iceberg catalog for reads."""
        if self._catalog is None:
            try:
                from pyiceberg.catalog import load_catalog

                catalog_config = self._config.get("catalog", {})
                self._catalog = load_catalog(
                    name="fraud_features",
                    **{
                        "type": catalog_config.get("type", "rest"),
                        "uri": catalog_config.get("uri", "http://localhost:8181"),
                        "warehouse": catalog_config.get("warehouse", ""),
                    },
                )
            except Exception as e:
                logger.warning("iceberg_catalog_unavailable", error=str(e))
        return self._catalog

    def get_features_at(
        self,
        account_id: str,
        timestamp: datetime,
        table_name: str = "realtime_feature_snapshots",
    ) -> Optional[dict]:
        """Retrieve the feature vector for an account at a specific point in time.

        Uses Iceberg's snapshot-based time-travel to find the snapshot that
        was current at the given timestamp, then filters for the account.

        This replaces hours of manual feature reconstruction:
        Before: Query raw events → replay feature pipeline → compare
        After:  Single time-travel query → exact feature state

        Args:
            account_id: The account to look up.
            timestamp: The point in time to query.
            table_name: Which feature snapshot table to query.

        Returns:
            Feature dict as it existed at the given timestamp, or None.
        """
        catalog = self._get_catalog()
        if catalog is None:
            logger.warning("time_travel_unavailable_no_catalog")
            return None

        try:
            table_identifier = f"{self._namespace}.{table_name}"
            table = catalog.load_table(table_identifier)

            # Find the snapshot that was current at the requested timestamp
            target_snapshot = None
            for snapshot in table.metadata.snapshots:
                snapshot_time = datetime.fromtimestamp(
                    snapshot.timestamp_ms / 1000
                )
                if snapshot_time <= timestamp:
                    target_snapshot = snapshot

            if target_snapshot is None:
                logger.info(
                    "no_snapshot_found",
                    account_id=account_id,
                    timestamp=timestamp.isoformat(),
                )
                return None

            # Scan with the target snapshot
            scan = table.scan(
                snapshot_id=target_snapshot.snapshot_id,
                row_filter=f"account_id == '{account_id}'",
            )

            # Get the most recent record for this account
            arrow_table = scan.to_arrow()
            if len(arrow_table) == 0:
                return None

            # Convert last row to dict
            df = arrow_table.to_pandas()
            record = df.sort_values("computed_at", ascending=False).iloc[0].to_dict()

            logger.info(
                "time_travel_query_success",
                account_id=account_id,
                query_timestamp=timestamp.isoformat(),
                snapshot_id=target_snapshot.snapshot_id,
            )
            return record

        except Exception as e:
            logger.error(
                "time_travel_query_failed",
                account_id=account_id,
                timestamp=timestamp.isoformat(),
                error=str(e),
            )
            return None

    def get_feature_history(
        self,
        account_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        table_name: str = "realtime_feature_snapshots",
    ) -> list[dict]:
        """Retrieve all feature snapshots for an account within a time range.

        Useful for understanding how features evolved leading up to a
        fraud event. Returns snapshots in chronological order.

        Args:
            account_id: The account to query.
            start_time: Beginning of the time range.
            end_time: End of the time range (defaults to now).
            table_name: Which feature snapshot table to query.

        Returns:
            List of feature dicts ordered by computed_at.
        """
        end_time = end_time or datetime.utcnow()
        catalog = self._get_catalog()

        if catalog is None:
            return []

        try:
            table_identifier = f"{self._namespace}.{table_name}"
            table = catalog.load_table(table_identifier)

            scan = table.scan(
                row_filter=(
                    f"account_id == '{account_id}' "
                    f"AND computed_at >= '{start_time.isoformat()}' "
                    f"AND computed_at <= '{end_time.isoformat()}'"
                ),
            )

            arrow_table = scan.to_arrow()
            if len(arrow_table) == 0:
                return []

            df = arrow_table.to_pandas().sort_values("computed_at")
            return df.to_dict(orient="records")

        except Exception as e:
            logger.error(
                "feature_history_query_failed",
                account_id=account_id,
                error=str(e),
            )
            return []

    def compare_features(
        self,
        account_id: str,
        timestamp_a: datetime,
        timestamp_b: datetime,
        table_name: str = "realtime_feature_snapshots",
    ) -> Optional[dict]:
        """Compare feature vectors at two points in time.

        Useful for investigating what changed before a fraud event.
        Returns a diff showing which features changed and by how much.
        """
        features_a = self.get_features_at(account_id, timestamp_a, table_name)
        features_b = self.get_features_at(account_id, timestamp_b, table_name)

        if features_a is None or features_b is None:
            return None

        diff = {}
        all_keys = set(features_a.keys()) | set(features_b.keys())

        for key in all_keys:
            val_a = features_a.get(key)
            val_b = features_b.get(key)
            if val_a != val_b:
                diff[key] = {
                    "before": val_a,
                    "after": val_b,
                }
                # Add delta for numeric values
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    diff[key]["delta"] = val_b - val_a

        return {
            "account_id": account_id,
            "timestamp_a": timestamp_a.isoformat(),
            "timestamp_b": timestamp_b.isoformat(),
            "changes": diff,
            "num_changed_features": len(diff),
        }
