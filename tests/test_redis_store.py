"""
Tests for Redis feature store read/write operations.

Uses fakeredis for isolated, in-memory testing without requiring
a running Redis instance. Tests verify:
- Feature serialization/deserialization roundtrips
- TTL application on writes
- Pipeline bulk writes for batch backfill
- Feature vector merging (realtime + batch)
- Missing feature handling (partial vectors)
"""

from datetime import datetime
from unittest.mock import patch

import pytest

# fakeredis provides an in-memory Redis implementation for testing
try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from src.feature_store.redis_store import RedisFeatureStore
from src.models.schemas import BatchFeatures, RealtimeFeatures


@pytest.fixture
def redis_config():
    """Test Redis configuration."""
    return {
        "connection": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "decode_responses": True,
        },
        "key_prefix": "test:features",
        "ttl": {
            "realtime_features_seconds": 3600,
            "batch_features_seconds": 129600,
        },
        "pipeline": {
            "batch_size": 100,
        },
    }


@pytest.fixture
def fake_redis_store(redis_config):
    """Create a RedisFeatureStore backed by fakeredis."""
    if not HAS_FAKEREDIS:
        pytest.skip("fakeredis not installed")

    store = RedisFeatureStore(config=redis_config)
    # Replace the real Redis client with fakeredis
    store._client = fakeredis.FakeRedis(decode_responses=True)
    return store


@pytest.fixture
def sample_realtime():
    """Sample realtime features for testing."""
    return RealtimeFeatures(
        account_id="acct_test_001",
        computed_at=datetime(2024, 1, 15, 10, 30, 0),
        txn_count_1h=5,
        txn_amount_sum_1h=150.50,
        txn_amount_max_1h=75.00,
        txn_velocity_5m=1.5,
        distance_from_home=12.5,
        is_new_merchant=True,
        hour_of_day=10,
        is_weekend=False,
        is_online=False,
        card_present=True,
        country_code="US",
    )


@pytest.fixture
def sample_batch():
    """Sample batch features for testing."""
    return BatchFeatures(
        account_id="acct_test_001",
        computed_at=datetime(2024, 1, 15, 2, 0, 0),
        avg_txn_amount_30d=45.67,
        txn_count_30d=120,
        distinct_merchants_30d=15,
        distinct_merchants_7d=8,
        txn_count_7d=30,
        txn_amount_p50_90d=35.00,
        txn_amount_p95_90d=200.00,
        txn_amount_p99_90d=450.00,
        days_since_last_txn=0.5,
        avg_daily_txn_count_30d=4.0,
        most_common_merchant_category="grocery",
        pct_online_txns_30d=0.25,
    )


class TestRedisFeatureStoreWrite:
    """Tests for writing features to Redis."""

    def test_write_realtime_features(self, fake_redis_store, sample_realtime):
        """Writing realtime features should be retrievable."""
        fake_redis_store.write_realtime_features(sample_realtime)

        key = "test:features:acct_test_001:realtime"
        raw = fake_redis_store._client.get(key)
        assert raw is not None

        restored = RealtimeFeatures.model_validate_json(raw)
        assert restored.account_id == "acct_test_001"
        assert restored.txn_count_1h == 5
        assert restored.txn_amount_sum_1h == 150.50

    def test_write_batch_features(self, fake_redis_store, sample_batch):
        """Writing batch features should be retrievable."""
        fake_redis_store.write_batch_features(sample_batch)

        key = "test:features:acct_test_001:batch"
        raw = fake_redis_store._client.get(key)
        assert raw is not None

        restored = BatchFeatures.model_validate_json(raw)
        assert restored.avg_txn_amount_30d == 45.67
        assert restored.txn_count_30d == 120

    def test_realtime_ttl_set(self, fake_redis_store, sample_realtime):
        """Realtime features should have a TTL set."""
        fake_redis_store.write_realtime_features(sample_realtime)

        key = "test:features:acct_test_001:realtime"
        ttl = fake_redis_store._client.ttl(key)
        # TTL should be close to 3600 (±1 second for execution time)
        assert 3598 <= ttl <= 3600

    def test_batch_ttl_set(self, fake_redis_store, sample_batch):
        """Batch features should have a longer TTL."""
        fake_redis_store.write_batch_features(sample_batch)

        key = "test:features:acct_test_001:batch"
        ttl = fake_redis_store._client.ttl(key)
        assert 129598 <= ttl <= 129600

    def test_overwrite_updates_features(self, fake_redis_store, sample_realtime):
        """Writing features for the same account should overwrite."""
        fake_redis_store.write_realtime_features(sample_realtime)

        updated = RealtimeFeatures(
            account_id="acct_test_001",
            txn_count_1h=10,  # Updated value
            txn_amount_sum_1h=300.00,
        )
        fake_redis_store.write_realtime_features(updated)

        result = fake_redis_store.get_realtime_features("acct_test_001")
        assert result.txn_count_1h == 10
        assert result.txn_amount_sum_1h == 300.00


class TestRedisFeatureStoreRead:
    """Tests for reading features from Redis."""

    def test_get_realtime_features(self, fake_redis_store, sample_realtime):
        fake_redis_store.write_realtime_features(sample_realtime)

        result = fake_redis_store.get_realtime_features("acct_test_001")
        assert result is not None
        assert result.txn_count_1h == 5
        assert result.is_new_merchant is True

    def test_get_nonexistent_returns_none(self, fake_redis_store):
        """Looking up a missing account should return None, not crash."""
        result = fake_redis_store.get_realtime_features("acct_nonexistent")
        assert result is None

    def test_get_feature_vector_full(
        self, fake_redis_store, sample_realtime, sample_batch
    ):
        """Full feature vector should merge realtime and batch features."""
        fake_redis_store.write_realtime_features(sample_realtime)
        fake_redis_store.write_batch_features(sample_batch)

        fv = fake_redis_store.get_feature_vector("acct_test_001")
        assert fv is not None
        assert fv.account_id == "acct_test_001"
        assert fv.realtime.txn_count_1h == 5
        assert fv.batch.avg_txn_amount_30d == 45.67

    def test_get_feature_vector_realtime_only(
        self, fake_redis_store, sample_realtime
    ):
        """Feature vector with only realtime features should use batch defaults."""
        fake_redis_store.write_realtime_features(sample_realtime)

        fv = fake_redis_store.get_feature_vector("acct_test_001")
        assert fv is not None
        assert fv.realtime.txn_count_1h == 5
        assert fv.batch.avg_txn_amount_30d == 0.0  # Default

    def test_get_feature_vector_batch_only(self, fake_redis_store, sample_batch):
        """Feature vector with only batch features should use realtime defaults."""
        fake_redis_store.write_batch_features(sample_batch)

        fv = fake_redis_store.get_feature_vector("acct_test_001")
        assert fv is not None
        assert fv.realtime.txn_count_1h == 0  # Default
        assert fv.batch.avg_txn_amount_30d == 45.67

    def test_get_feature_vector_missing_returns_none(self, fake_redis_store):
        """Missing account should return None for feature vector."""
        fv = fake_redis_store.get_feature_vector("acct_missing")
        assert fv is None


class TestRedisFeatureStoreBulk:
    """Tests for bulk write operations (batch backfill)."""

    def test_bulk_write_batch_features(self, fake_redis_store):
        """Bulk write should store all features."""
        features = [
            BatchFeatures(
                account_id=f"acct_bulk_{i:03d}",
                avg_txn_amount_30d=float(i * 10),
            )
            for i in range(50)
        ]

        written = fake_redis_store.write_batch_features_bulk(features)
        assert written == 50

        # Spot check a few
        result = fake_redis_store.get_batch_features("acct_bulk_025")
        assert result is not None
        assert result.avg_txn_amount_30d == 250.0

    def test_bulk_write_empty_list(self, fake_redis_store):
        """Bulk write with empty list should return 0."""
        written = fake_redis_store.write_batch_features_bulk([])
        assert written == 0


class TestRedisFeatureStoreUtility:
    """Tests for utility operations."""

    def test_delete_account(self, fake_redis_store, sample_realtime, sample_batch):
        """Deleting an account should remove all feature keys."""
        fake_redis_store.write_realtime_features(sample_realtime)
        fake_redis_store.write_batch_features(sample_batch)

        fake_redis_store.delete_account("acct_test_001")

        assert fake_redis_store.get_realtime_features("acct_test_001") is None
        assert fake_redis_store.get_batch_features("acct_test_001") is None

    def test_key_format(self, fake_redis_store):
        """Key format should follow the documented convention."""
        key = fake_redis_store._key("acct_123", "realtime")
        assert key == "test:features:acct_123:realtime"

    def test_model_input_roundtrip(
        self, fake_redis_store, sample_realtime, sample_batch
    ):
        """Feature vector should produce a valid model input dict."""
        fake_redis_store.write_realtime_features(sample_realtime)
        fake_redis_store.write_batch_features(sample_batch)

        fv = fake_redis_store.get_feature_vector("acct_test_001")
        model_input = fv.to_model_input()

        # Should be a flat dict with no nested structures
        for key, value in model_input.items():
            assert not isinstance(value, dict), f"Nested dict found at key '{key}'"
            assert not isinstance(value, list), f"List found at key '{key}'"

        # Should contain features from both paths
        assert "txn_count_1h" in model_input
        assert "avg_txn_amount_30d" in model_input
