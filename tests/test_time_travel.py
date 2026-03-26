"""
Tests for Iceberg time-travel query functionality.

These tests validate the core debugging workflow:
1. Write feature snapshots at known timestamps
2. Query features AS OF a specific timestamp
3. Compare features between two points in time

Since Iceberg catalog may not be available in test environments,
these tests focus on the logic layer and mock the catalog interactions.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.iceberg.time_travel import IcebergSnapshotWriter, IcebergTimeTravel


class TestIcebergSnapshotWriter:
    """Tests for snapshot writing logic."""

    def test_empty_records_returns_true(self):
        """Writing empty records should succeed (no-op)."""
        writer = IcebergSnapshotWriter(config={
            "namespace": "test",
            "catalog": {"type": "rest", "uri": "http://localhost:8181", "warehouse": ""},
        })
        result = writer.write_snapshot(records=[], table_name="test_table")
        assert result is True

    def test_local_fallback_writes_parquet(self, tmp_path):
        """When Iceberg catalog is unavailable, should write local Parquet."""
        writer = IcebergSnapshotWriter(config={
            "namespace": "test",
            "catalog": {"type": "rest", "uri": "http://localhost:8181", "warehouse": ""},
        })
        # _catalog stays None → triggers local fallback

        records = [
            {
                "account_id": "acct_001",
                "computed_at": "2024-01-15T10:30:00",
                "txn_count_1h": 5,
                "txn_amount_sum_1h": 150.50,
            },
            {
                "account_id": "acct_002",
                "computed_at": "2024-01-15T10:30:00",
                "txn_count_1h": 3,
                "txn_amount_sum_1h": 80.00,
            },
        ]

        with patch.object(writer, '_write_local_fallback', return_value=True) as mock_fallback:
            result = writer.write_snapshot(
                records=records,
                table_name="test_snapshots",
                batch_id=42,
            )

            assert result is True
            mock_fallback.assert_called_once_with(records, "test_snapshots", 42)


class TestIcebergTimeTravel:
    """Tests for time-travel query logic."""

    def test_compare_features_identifies_changes(self):
        """compare_features should detect which features changed between timestamps."""
        tt = IcebergTimeTravel(config={
            "namespace": "test",
            "catalog": {"type": "rest", "uri": "http://localhost:8181", "warehouse": ""},
        })

        # Mock the get_features_at method to return controlled data
        features_before = {
            "account_id": "acct_001",
            "txn_count_1h": 3,
            "txn_amount_sum_1h": 80.0,
            "is_new_merchant": False,
        }
        features_after = {
            "account_id": "acct_001",
            "txn_count_1h": 8,
            "txn_amount_sum_1h": 350.0,
            "is_new_merchant": True,
        }

        with patch.object(tt, 'get_features_at', side_effect=[features_before, features_after]):
            diff = tt.compare_features(
                account_id="acct_001",
                timestamp_a=datetime(2024, 1, 15, 10, 0),
                timestamp_b=datetime(2024, 1, 15, 10, 30),
            )

        assert diff is not None
        assert diff["account_id"] == "acct_001"
        assert diff["num_changed_features"] == 3

        # Check specific changes
        changes = diff["changes"]
        assert changes["txn_count_1h"]["before"] == 3
        assert changes["txn_count_1h"]["after"] == 8
        assert changes["txn_count_1h"]["delta"] == 5

        assert changes["txn_amount_sum_1h"]["delta"] == 270.0
        assert changes["is_new_merchant"]["before"] is False
        assert changes["is_new_merchant"]["after"] is True

    def test_compare_features_no_changes(self):
        """compare_features should return empty changes when features are identical."""
        tt = IcebergTimeTravel(config={
            "namespace": "test",
            "catalog": {"type": "rest", "uri": "http://localhost:8181", "warehouse": ""},
        })

        features = {
            "account_id": "acct_001",
            "txn_count_1h": 5,
            "txn_amount_sum_1h": 100.0,
        }

        with patch.object(tt, 'get_features_at', return_value=features):
            diff = tt.compare_features(
                account_id="acct_001",
                timestamp_a=datetime(2024, 1, 15, 10, 0),
                timestamp_b=datetime(2024, 1, 15, 10, 30),
            )

        assert diff is not None
        assert diff["num_changed_features"] == 0

    def test_compare_features_missing_snapshot(self):
        """compare_features should return None if a snapshot is missing."""
        tt = IcebergTimeTravel(config={
            "namespace": "test",
            "catalog": {"type": "rest", "uri": "http://localhost:8181", "warehouse": ""},
        })

        with patch.object(tt, 'get_features_at', side_effect=[None, {"txn_count_1h": 5}]):
            diff = tt.compare_features(
                account_id="acct_001",
                timestamp_a=datetime(2024, 1, 15, 10, 0),
                timestamp_b=datetime(2024, 1, 15, 10, 30),
            )

        assert diff is None

    def test_get_features_at_no_catalog(self):
        """Should return None gracefully when catalog is unavailable."""
        tt = IcebergTimeTravel(config={
            "namespace": "test",
            "catalog": {"type": "rest", "uri": "http://localhost:8181", "warehouse": ""},
        })
        # _catalog stays None

        result = tt.get_features_at(
            account_id="acct_001",
            timestamp=datetime(2024, 1, 15, 10, 30),
        )
        assert result is None

    def test_get_feature_history_no_catalog(self):
        """Should return empty list when catalog is unavailable."""
        tt = IcebergTimeTravel(config={
            "namespace": "test",
            "catalog": {"type": "rest", "uri": "http://localhost:8181", "warehouse": ""},
        })

        result = tt.get_feature_history(
            account_id="acct_001",
            start_time=datetime(2024, 1, 14),
            end_time=datetime(2024, 1, 15),
        )
        assert result == []
