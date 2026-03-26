"""
Tests for feature computation logic (src/models/features.py).

These tests validate the shared feature functions used by both streaming
and batch paths. Since both paths import from features.py, a failure
here indicates a potential training-serving skew risk.
"""

import math
from datetime import datetime

import pytest

from src.models.features import (
    check_is_new_merchant,
    compute_distance_km,
    compute_hour_features,
    compute_realtime_features,
)
from src.models.schemas import TransactionEvent


# ─── Distance Computation ───────────────────────────────────────────────────


class TestComputeDistanceKm:
    """Tests for Haversine distance calculation."""

    def test_same_point_returns_zero(self):
        """Distance from a point to itself should be zero."""
        assert compute_distance_km(40.7128, -74.0060, 40.7128, -74.0060) == 0.0

    def test_known_distance_nyc_to_la(self):
        """NYC to LA should be approximately 3,944 km."""
        distance = compute_distance_km(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < distance < 4000

    def test_short_distance(self):
        """Short distances (within a city) should be small."""
        # Two points in Manhattan, ~1.5 km apart
        distance = compute_distance_km(40.7580, -73.9855, 40.7484, -73.9856)
        assert 0.5 < distance < 2.0

    def test_none_latitude_returns_none(self):
        """Missing coordinates should return None, not crash."""
        assert compute_distance_km(None, -74.0, 40.7, -74.0) is None

    def test_none_longitude_returns_none(self):
        assert compute_distance_km(40.7, None, 40.7, -74.0) is None

    def test_both_none_returns_none(self):
        assert compute_distance_km(None, None, None, None) is None

    def test_symmetry(self):
        """Distance A→B should equal B→A."""
        d1 = compute_distance_km(40.7128, -74.0060, 34.0522, -118.2437)
        d2 = compute_distance_km(34.0522, -118.2437, 40.7128, -74.0060)
        assert abs(d1 - d2) < 0.001


# ─── Hour Features ───────────────────────────────────────────────────────────


class TestComputeHourFeatures:
    """Tests for hour-of-day and weekend extraction."""

    def test_weekday_morning(self):
        # Monday at 9 AM
        dt = datetime(2024, 1, 15, 9, 30, 0)
        hour, is_weekend = compute_hour_features(dt)
        assert hour == 9
        assert is_weekend is False

    def test_saturday(self):
        # Saturday at 2 PM
        dt = datetime(2024, 1, 13, 14, 0, 0)
        hour, is_weekend = compute_hour_features(dt)
        assert hour == 14
        assert is_weekend is True

    def test_sunday(self):
        dt = datetime(2024, 1, 14, 23, 59, 0)
        hour, is_weekend = compute_hour_features(dt)
        assert hour == 23
        assert is_weekend is True

    def test_midnight(self):
        dt = datetime(2024, 1, 15, 0, 0, 0)
        hour, _ = compute_hour_features(dt)
        assert hour == 0


# ─── New Merchant Check ─────────────────────────────────────────────────────


class TestCheckIsNewMerchant:
    """Tests for merchant novelty detection."""

    def test_known_merchant(self):
        known = {"merch_a", "merch_b", "merch_c"}
        assert check_is_new_merchant("merch_a", known) is False

    def test_new_merchant(self):
        known = {"merch_a", "merch_b"}
        assert check_is_new_merchant("merch_new", known) is True

    def test_empty_known_set(self):
        """First transaction ever should be a new merchant."""
        assert check_is_new_merchant("any_merchant", set()) is True


# ─── Realtime Feature Computation ────────────────────────────────────────────


class TestComputeRealtimeFeatures:
    """Integration tests for the full real-time feature computation."""

    @pytest.fixture
    def sample_event(self):
        return TransactionEvent(
            event_id="evt_test001",
            account_id="acct_00001",
            timestamp=datetime(2024, 1, 15, 14, 30, 0),
            amount=42.99,
            merchant_id="merch_starbucks",
            merchant_category="food_beverage",
            transaction_type="purchase",
            latitude=40.7128,
            longitude=-74.0060,
            is_online=False,
            card_present=True,
            country_code="US",
        )

    @pytest.fixture
    def window_events(self, sample_event):
        """Create a list of events simulating a 1-hour window."""
        events = [sample_event]
        for i in range(4):
            events.append(
                TransactionEvent(
                    event_id=f"evt_test{i+2:03d}",
                    account_id="acct_00001",
                    timestamp=datetime(2024, 1, 15, 14, i * 10, 0),
                    amount=10.0 + i * 5,
                    merchant_id=f"merch_{i}",
                    merchant_category="retail",
                    transaction_type="purchase",
                )
            )
        return events

    def test_basic_feature_computation(self, sample_event, window_events):
        features = compute_realtime_features(
            account_id="acct_00001",
            current_event=sample_event,
            window_events_1h=window_events,
            window_events_5m=window_events[:2],
            known_merchants={"merch_0", "merch_1"},
        )

        assert features.account_id == "acct_00001"
        assert features.txn_count_1h == 5
        assert features.txn_amount_sum_1h > 0
        assert features.hour_of_day == 14
        assert features.is_weekend is False
        assert features.is_online is False
        assert features.card_present is True

    def test_new_merchant_detection(self, sample_event, window_events):
        features = compute_realtime_features(
            account_id="acct_00001",
            current_event=sample_event,
            window_events_1h=window_events,
            window_events_5m=[],
            known_merchants={"merch_other"},
        )
        # merch_starbucks is not in known_merchants
        assert features.is_new_merchant is True

    def test_known_merchant(self, sample_event, window_events):
        features = compute_realtime_features(
            account_id="acct_00001",
            current_event=sample_event,
            window_events_1h=window_events,
            window_events_5m=[],
            known_merchants={"merch_starbucks"},
        )
        assert features.is_new_merchant is False

    def test_distance_computed_with_home(self, sample_event, window_events):
        features = compute_realtime_features(
            account_id="acct_00001",
            current_event=sample_event,
            window_events_1h=window_events,
            window_events_5m=[],
            known_merchants=set(),
            home_lat=40.7580,
            home_lon=-73.9855,
        )
        # NYC coordinates ~5km apart
        assert features.distance_from_home is not None
        assert features.distance_from_home > 0

    def test_distance_none_without_home(self, sample_event, window_events):
        features = compute_realtime_features(
            account_id="acct_00001",
            current_event=sample_event,
            window_events_1h=window_events,
            window_events_5m=[],
            known_merchants=set(),
        )
        assert features.distance_from_home is None

    def test_empty_windows(self, sample_event):
        """Edge case: no history, only the current event."""
        features = compute_realtime_features(
            account_id="acct_00001",
            current_event=sample_event,
            window_events_1h=[],
            window_events_5m=[],
            known_merchants=set(),
        )
        assert features.txn_count_1h == 0
        assert features.txn_amount_sum_1h == 0.0
