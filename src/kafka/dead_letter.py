"""
Dead-letter queue (DLQ) for messages that fail processing.

When a message fails validation or exhausts processing retries, it's
routed to a dedicated DLQ topic instead of being silently dropped.
The DLQ message includes the original payload plus metadata about
why it failed, which partition/offset it came from, and when it failed.

This is essential for zero-message-loss guarantees: every event is
either successfully processed OR preserved in the DLQ for investigation.
DLQ messages have a 30-day retention (vs 7 days for the main topic)
to allow time for root-cause analysis and reprocessing.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.utils.config import get_kafka_config
from src.utils.logging_config import get_logger

logger = get_logger("kafka.dlq")


class DeadLetterQueue:
    """Routes failed messages to a dead-letter Kafka topic."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_kafka_config()
        self._producer: Optional[KafkaProducer] = None
        self._dlq_topic = self._config["topics"]["dead_letter"]["name"]
        self._messages_routed = 0

    def _get_producer(self) -> KafkaProducer:
        """Lazy-initialize a dedicated producer for DLQ writes.

        We use a separate producer from the main transaction producer
        so DLQ writes don't interfere with normal message flow.
        """
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self._config["broker"]["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks="all",
                retries=5,  # More aggressive retries — losing a DLQ message is worse
                retry_backoff_ms=2000,
            )
            logger.info("dlq_producer_initialized", topic=self._dlq_topic)
        return self._producer

    def route(
        self,
        raw_message: Any,
        error_reason: str,
        source_partition: int,
        source_offset: int,
        extra_metadata: Optional[dict] = None,
    ) -> bool:
        """Route a failed message to the dead-letter queue.

        The DLQ message wraps the original payload with failure metadata.
        This envelope pattern makes it easy to reprocess DLQ messages —
        the original event is intact inside the 'original_message' field.

        Args:
            raw_message: The original message value (dict or string).
            error_reason: Human-readable description of why processing failed.
            source_partition: Partition the original message came from.
            source_offset: Offset of the original message.
            extra_metadata: Optional additional context (stack trace, etc.).

        Returns:
            True if the DLQ write succeeded, False otherwise.
        """
        dlq_envelope = {
            "original_message": raw_message,
            "error_reason": error_reason,
            "source_topic": self._config["topics"]["transactions"]["name"],
            "source_partition": source_partition,
            "source_offset": source_offset,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": extra_metadata or {},
        }

        try:
            producer = self._get_producer()
            future = producer.send(
                topic=self._dlq_topic,
                value=dlq_envelope,
            )
            future.get(timeout=10)
            self._messages_routed += 1

            logger.warning(
                "message_routed_to_dlq",
                source_partition=source_partition,
                source_offset=source_offset,
                error_reason=error_reason,
                total_dlq_messages=self._messages_routed,
            )
            return True

        except KafkaError as e:
            # If we can't even write to the DLQ, log critically.
            # This is the last resort — the message is effectively lost
            # unless it can be recovered from Kafka's main topic retention.
            logger.critical(
                "dlq_write_failed",
                source_partition=source_partition,
                source_offset=source_offset,
                error_reason=error_reason,
                dlq_error=str(e),
                original_message=str(raw_message)[:500],  # Truncate for logging
            )
            return False

    @property
    def messages_routed(self) -> int:
        """Total number of messages successfully routed to the DLQ."""
        return self._messages_routed

    def close(self) -> None:
        """Flush and close the DLQ producer."""
        if self._producer is not None:
            self._producer.flush(timeout=15)
            self._producer.close()
            logger.info(
                "dlq_producer_closed",
                total_messages_routed=self._messages_routed,
            )
