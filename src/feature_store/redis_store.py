"""
Redis-backed feature serving layer.

Redis is the unified serving layer for both hot and cold path features.
The ML inference service calls get_feature_vector(account_id) and receives
a merged FeatureVector containing both real-time and batch features from
a single Redis lookup.

Key schema:
    {prefix}:{account_id}:realtime  → JSON of RealtimeFeatures
    {prefix}:{account_id}:batch     → JSON of BatchFeatures

Design decisions:
- Hash maps per feature group: We store realtime and batch features as
  separate keys rather than a single hash. This lets us update them
  independently (streaming updates realtime every 10s, batch updates
  batch nightly) without read-modify-write races.
- TTL as staleness guard: Each key has a TTL. If the streaming pipeline
  goes down, stale features expire rather than silently serving outdated
  values. The model falls back to batch-only features.
- Pipeline bulk writes: Batch backfills use Redis pipelines to write
  thousands of feature vectors in a single round-trip.
"""

import json
import time
from datetime import datetime
from typing import Optional

import redis

from src.models.schemas import BatchFeatures, FeatureVector, RealtimeFeatures
from src.utils.config import get_redis_config
from src.utils.logging_config import get_logger

logger = get_logger("feature_store.redis")


class RedisFeatureStore:
    """Read/write feature vectors in Redis."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_redis_config()
        self._client: Optional[redis.Redis] = None
        self._prefix = self._config.get("key_prefix", "fraud:features")
        self._ttl = self._config.get("ttl", {})

    def _get_client(self) -> redis.Redis:
        """Lazy-initialize Redis connection with retry logic."""
        if self._client is None:
            conn_config = self._config.get("connection", {})
            self._client = redis.Redis(
                host=conn_config.get("host", "localhost"),
                port=int(conn_config.get("port", 6379)),
                db=int(conn_config.get("db", 0)),
                password=conn_config.get("password") or None,
                decode_responses=conn_config.get("decode_responses", True),
                socket_timeout=conn_config.get("socket_timeout", 5),
                socket_connect_timeout=conn_config.get("socket_connect_timeout", 5),
                retry_on_timeout=conn_config.get("retry_on_timeout", True),
            )
            # Verify connectivity
            self._client.ping()
            logger.info(
                "redis_connected",
                host=conn_config.get("host"),
                port=conn_config.get("port"),
            )
        return self._client

    def _key(self, account_id: str, feature_group: str) -> str:
        """Build a Redis key for a feature group.

        Format: fraud:features:{account_id}:{realtime|batch}
        """
        return f"{self._prefix}:{account_id}:{feature_group}"

    # ─── Write Operations ────────────────────────────────────────────────

    def write_realtime_features(self, features: RealtimeFeatures) -> None:
        """Write real-time features for an account.

        Called by the streaming processor on every micro-batch.
        TTL ensures stale features expire if streaming stops.
        """
        client = self._get_client()
        key = self._key(features.account_id, "realtime")
        ttl = int(self._ttl.get("realtime_features_seconds", 3600))

        start = time.monotonic()
        client.setex(
            name=key,
            time=ttl,
            value=features.model_dump_json(),
        )
        latency_ms = (time.monotonic() - start) * 1000

        logger.debug(
            "realtime_features_written",
            account_id=features.account_id,
            latency_ms=round(latency_ms, 2),
        )

    def write_batch_features(self, features: BatchFeatures) -> None:
        """Write batch features for an account.

        Called by the nightly batch pipeline during Redis backfill.
        Longer TTL (36h) gives headroom if the batch job runs late.
        """
        client = self._get_client()
        key = self._key(features.account_id, "batch")
        ttl = int(self._ttl.get("batch_features_seconds", 129600))

        client.setex(
            name=key,
            time=ttl,
            value=features.model_dump_json(),
        )

    def write_batch_features_bulk(self, features_list: list[BatchFeatures]) -> int:
        """Bulk-write batch features using Redis pipeline.

        Pipelines batch multiple commands into a single round-trip,
        reducing network overhead from O(n) to O(1) for large backfills.

        Returns the number of features written.
        """
        client = self._get_client()
        pipeline_config = self._config.get("pipeline", {})
        batch_size = pipeline_config.get("batch_size", 1000)
        ttl = int(self._ttl.get("batch_features_seconds", 129600))
        written = 0

        for i in range(0, len(features_list), batch_size):
            chunk = features_list[i : i + batch_size]
            pipe = client.pipeline(transaction=False)

            for features in chunk:
                key = self._key(features.account_id, "batch")
                pipe.setex(name=key, time=ttl, value=features.model_dump_json())

            pipe.execute()
            written += len(chunk)

            logger.debug(
                "batch_pipeline_chunk",
                chunk_size=len(chunk),
                total_written=written,
            )

        logger.info(
            "batch_features_bulk_written",
            total=written,
        )
        return written

    # ─── Read Operations ─────────────────────────────────────────────────

    def get_realtime_features(self, account_id: str) -> Optional[RealtimeFeatures]:
        """Retrieve real-time features for an account."""
        client = self._get_client()
        key = self._key(account_id, "realtime")

        raw = client.get(key)
        if raw is None:
            return None
        return RealtimeFeatures.model_validate_json(raw)

    def get_batch_features(self, account_id: str) -> Optional[BatchFeatures]:
        """Retrieve batch features for an account."""
        client = self._get_client()
        key = self._key(account_id, "batch")

        raw = client.get(key)
        if raw is None:
            return None
        return BatchFeatures.model_validate_json(raw)

    def get_feature_vector(self, account_id: str) -> Optional[FeatureVector]:
        """Retrieve the full feature vector (realtime + batch) for an account.

        This is the primary read path used by the ML inference service.
        Uses a pipeline to fetch both feature groups in a single round-trip.

        Returns None if neither feature group exists. If only one group
        exists, returns a FeatureVector with defaults for the missing group.
        """
        client = self._get_client()
        rt_key = self._key(account_id, "realtime")
        bt_key = self._key(account_id, "batch")

        start = time.monotonic()
        pipe = client.pipeline(transaction=False)
        pipe.get(rt_key)
        pipe.get(bt_key)
        rt_raw, bt_raw = pipe.execute()
        latency_ms = (time.monotonic() - start) * 1000

        if rt_raw is None and bt_raw is None:
            logger.debug(
                "feature_vector_miss",
                account_id=account_id,
                latency_ms=round(latency_ms, 2),
            )
            return None

        realtime = (
            RealtimeFeatures.model_validate_json(rt_raw)
            if rt_raw
            else RealtimeFeatures(account_id=account_id)
        )
        batch = (
            BatchFeatures.model_validate_json(bt_raw)
            if bt_raw
            else BatchFeatures(account_id=account_id)
        )

        logger.debug(
            "feature_vector_served",
            account_id=account_id,
            has_realtime=rt_raw is not None,
            has_batch=bt_raw is not None,
            latency_ms=round(latency_ms, 2),
        )

        return FeatureVector(
            account_id=account_id,
            realtime=realtime,
            batch=batch,
            served_at=datetime.utcnow(),
        )

    # ─── Utility ─────────────────────────────────────────────────────────

    def delete_account(self, account_id: str) -> None:
        """Remove all features for an account (e.g., for GDPR deletion)."""
        client = self._get_client()
        rt_key = self._key(account_id, "realtime")
        bt_key = self._key(account_id, "batch")
        client.delete(rt_key, bt_key)
        logger.info("account_features_deleted", account_id=account_id)

    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self._get_client().ping()
        except redis.ConnectionError:
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            self._client.close()
            logger.info("redis_connection_closed")
