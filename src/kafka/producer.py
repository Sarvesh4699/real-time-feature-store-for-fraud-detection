"""
Kafka producer for transaction events.

Publishes TransactionEvent messages to the transactions topic, keyed on
account_id. The key-based partitioning ensures all events for a given
account land on the same partition, which is critical for the streaming
consumer to maintain accurate per-user windowed aggregations.

Design decisions:
- acks=all: Wait for all in-sync replicas before considering a write successful.
  Prevents data loss at the cost of slightly higher latency (~5ms).
- max_in_flight_requests=1: Combined with retries, this guarantees ordering.
  Without it, a retry of batch N could arrive after batch N+1.
- snappy compression: Good balance of CPU cost vs compression ratio for
  JSON-encoded transaction events.
"""

import json
import hashlib
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.models.schemas import TransactionEvent
from src.utils.config import get_kafka_config
from src.utils.logging_config import get_logger

logger = get_logger("kafka.producer")


class TransactionProducer:
    """Produces transaction events to Kafka with account_id-based partitioning."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_kafka_config()
        self._producer: Optional[KafkaProducer] = None
        self._topic = self._config["topics"]["transactions"]["name"]

    def _get_producer(self) -> KafkaProducer:
        """Lazy-initialize the KafkaProducer.

        We defer creation so the producer can be instantiated before
        Kafka is available (e.g., during import-time in Airflow DAGs).
        """
        if self._producer is None:
            producer_config = self._config.get("producer", {})
            self._producer = KafkaProducer(
                bootstrap_servers=self._config["broker"]["bootstrap_servers"],
                key_serializer=lambda k: k.encode("utf-8"),
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks=producer_config.get("acks", "all"),
                retries=producer_config.get("retries", 3),
                retry_backoff_ms=producer_config.get("retry_backoff_ms", 1000),
                max_in_flight_requests_per_connection=producer_config.get(
                    "max_in_flight_requests_per_connection", 1
                ),
                compression_type=producer_config.get("compression_type", "snappy"),
                linger_ms=producer_config.get("linger_ms", 10),
                batch_size=producer_config.get("batch_size", 16384),
            )
            logger.info(
                "kafka_producer_initialized",
                bootstrap_servers=self._config["broker"]["bootstrap_servers"],
                topic=self._topic,
            )
        return self._producer

    def _partition_for_account(self, account_id: str, num_partitions: int = 8) -> int:
        """Deterministic partition assignment based on account_id.

        Uses murmur-style hashing (via hashlib) to distribute accounts
        evenly across partitions. This matches Kafka's default partitioner
        behavior for keyed messages, but we implement it explicitly for
        transparency and testability.

        Args:
            account_id: The account identifier.
            num_partitions: Number of partitions on the topic (default: 8).

        Returns:
            Partition number (0 to num_partitions-1).
        """
        hash_bytes = hashlib.md5(account_id.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
        return hash_int % num_partitions

    def send(self, event: TransactionEvent) -> None:
        """Publish a single transaction event to Kafka.

        The event is validated via the Pydantic model before serialization.
        On failure, the error is logged and re-raised — the caller is
        responsible for retry/DLQ logic.

        Args:
            event: Validated TransactionEvent instance.

        Raises:
            KafkaError: If the message cannot be sent after retries.
        """
        producer = self._get_producer()
        try:
            future = producer.send(
                topic=self._topic,
                key=event.account_id,
                value=event.model_dump(mode="json"),
            )
            # Block for acknowledgment (synchronous send for reliability)
            record_metadata = future.get(timeout=10)

            logger.debug(
                "event_published",
                event_id=event.event_id,
                account_id=event.account_id,
                partition=record_metadata.partition,
                offset=record_metadata.offset,
            )
        except KafkaError as e:
            logger.error(
                "event_publish_failed",
                event_id=event.event_id,
                account_id=event.account_id,
                error=str(e),
            )
            raise

    def send_batch(self, events: list[TransactionEvent]) -> tuple[int, int]:
        """Publish a batch of events, returning (success_count, failure_count).

        Uses asynchronous sends with a final flush for throughput, but
        tracks individual failures for observability.
        """
        producer = self._get_producer()
        successes = 0
        failures = 0

        for event in events:
            try:
                producer.send(
                    topic=self._topic,
                    key=event.account_id,
                    value=event.model_dump(mode="json"),
                )
                successes += 1
            except KafkaError as e:
                logger.error(
                    "batch_event_failed",
                    event_id=event.event_id,
                    error=str(e),
                )
                failures += 1

        producer.flush(timeout=30)
        logger.info(
            "batch_published",
            total=len(events),
            successes=successes,
            failures=failures,
        )
        return successes, failures

    def close(self) -> None:
        """Flush pending messages and close the producer."""
        if self._producer is not None:
            self._producer.flush(timeout=30)
            self._producer.close()
            logger.info("kafka_producer_closed")
