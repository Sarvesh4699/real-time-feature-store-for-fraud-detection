"""
Tests for Pydantic schema models (src/models/schemas.py).

Schema validation is our first line of defense against bad data.
These tests verify that malformed events are caught at ingestion
rather than silently corrupting downstream feature computations.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    BatchFeatures,
    FeatureVector,
    RealtimeFeatures,
    TransactionEvent,
    TransactionType,
)


class TestTransactionEvent:
    """Tests for TransactionEvent schema validation."""

    def test_valid_event(self):
        event = TransactionEvent(
            event_id="evt_abc123",
            account_id="acct_12345",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            amount=42.99,
            merchant_id="merch_xyz",
            transaction_type="purchase",
        )
        assert event.amount == 42.99
        assert event.transaction_type == TransactionType.PURCHASE

    def test_iso_timestamp_parsing(self):
        """Timestamps in ISO format with Z suffix should parse correctly."""
        event = TransactionEvent(
            event_id="evt_1",
            account_id="acct_1",
            timestamp="2024-01-15T10:30:00Z",
            amount=10.0,
            merchant_id="merch_1",
            transaction_type="purchase",
        )
        assert event.timestamp.hour == 10

    def test_negative_amount_rejected(self):
        """Negative transaction amounts should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            TransactionEvent(
                event_id="evt_1",
                account_id="acct_1",
                timestamp=datetime.now(),
                amount=-10.0,
                merchant_id="merch_1",
                transaction_type="purchase",
            )
        assert "amount" in str(exc_info.value)

    def test_invalid_transaction_type_rejected(self):
        with pytest.raises(ValidationError):
            TransactionEvent(
                event_id="evt_1",
                account_id="acct_1",
                timestamp=datetime.now(),
                amount=10.0,
                merchant_id="merch_1",
                transaction_type="invalid_type",
            )

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            TransactionEvent(
                event_id="evt_1",
                # missing account_id
                timestamp=datetime.now(),
                amount=10.0,
                merchant_id="merch_1",
                transaction_type="purchase",
            )

    def test_latitude_range_validation(self):
        """Latitude must be between -90 and 90."""
        with pytest.raises(ValidationError):
            TransactionEvent(
                event_id="evt_1",
                account_id="acct_1",
                timestamp=datetime.now(),
                amount=10.0,
                merchant_id="merch_1",
                transaction_type="purchase",
                latitude=91.0,
            )

    def test_defaults_applied(self):
        event = TransactionEvent(
            event_id="evt_1",
            account_id="acct_1",
            timestamp=datetime.now(),
            amount=10.0,
            merchant_id="merch_1",
            transaction_type="purchase",
        )
        assert event.merchant_category == "unknown"
        assert event.is_online is False
        assert event.card_present is True
        assert event.country_code == "US"

    def test_json_roundtrip(self):
        """Events should survive JSON serialization/deserialization."""
        event = TransactionEvent(
            event_id="evt_1",
            account_id="acct_1",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            amount=42.99,
            merchant_id="merch_1",
            transaction_type="purchase",
            latitude=40.7128,
            longitude=-74.0060,
        )
        json_str = event.model_dump_json()
        restored = TransactionEvent.model_validate_json(json_str)
        assert restored.event_id == event.event_id
        assert restored.amount == event.amount
        assert restored.latitude == event.latitude


class TestRealtimeFeatures:
    """Tests for RealtimeFeatures schema."""

    def test_valid_realtime_features(self):
        features = RealtimeFeatures(
            account_id="acct_1",
            txn_count_1h=5,
            txn_amount_sum_1h=150.50,
            txn_velocity_5m=1.5,
        )
        assert features.txn_count_1h == 5

    def test_defaults(self):
        features = RealtimeFeatures(account_id="acct_1")
        assert features.txn_count_1h == 0
        assert features.txn_amount_sum_1h == 0.0
        assert features.is_new_merchant is False
        assert features.is_online is False


class TestBatchFeatures:
    """Tests for BatchFeatures schema."""

    def test_valid_batch_features(self):
        features = BatchFeatures(
            account_id="acct_1",
            avg_txn_amount_30d=45.67,
            txn_count_30d=120,
            txn_amount_p95_90d=250.00,
        )
        assert features.txn_count_30d == 120

    def test_pct_online_range(self):
        """pct_online_txns_30d must be between 0 and 1."""
        with pytest.raises(ValidationError):
            BatchFeatures(
                account_id="acct_1",
                pct_online_txns_30d=1.5,  # > 1.0
            )


class TestFeatureVector:
    """Tests for the combined FeatureVector."""

    def test_to_model_input_flattens(self):
        """to_model_input() should produce a flat dict for ML model input."""
        fv = FeatureVector(
            account_id="acct_1",
            realtime=RealtimeFeatures(
                account_id="acct_1",
                txn_count_1h=5,
                txn_amount_sum_1h=100.0,
            ),
            batch=BatchFeatures(
                account_id="acct_1",
                avg_txn_amount_30d=45.0,
            ),
        )
        model_input = fv.to_model_input()

        # Should contain features from both paths
        assert model_input["txn_count_1h"] == 5
        assert model_input["avg_txn_amount_30d"] == 45.0
        assert model_input["account_id"] == "acct_1"

        # Should NOT contain metadata fields
        assert "computed_at" not in model_input
