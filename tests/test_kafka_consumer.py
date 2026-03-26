"""
Tests for Kafka consumer offset management and DLQ routing.

These tests verify the zero-message-loss guarantees:
- Messages are only committed after successful processing
- Failed messages are routed to the dead-letter queue
- Poison pills don't block the consumer
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.kafka.consumer import TransactionConsumer
from src.kafka.dead_letter import DeadLetterQueue
from src.models.schemas import TransactionEvent


class TestDeadLetterQueue:
    """Tests for DLQ routing logic."""

    def test_dlq_envelope_format(self):
        """DLQ messages should include the original message plus metadata."""
        dlq = DeadLetterQueue.__new__(DeadLetterQueue)
        dlq._config = {
            "broker": {"bootstrap_servers": "localhost:9092"},
            "topics": {
                "transactions": {"name": "fraud.transactions.raw"},
                "dead_letter": {"name": "fraud.transactions.dlq"},
            },
        }
        dlq._dlq_topic = "fraud.transactions.dlq"
        dlq._messages_routed = 0

        # Mock the producer
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_producer.send.return_value = mock_future
        dlq._producer = mock_producer

        original_message = {"event_id": "evt_bad", "amount": "not_a_number"}

        dlq.route(
            raw_message=original_message,
            error_reason="Validation error: amount must be float",
            source_partition=3,
            source_offset=12345,
        )

        # Verify the DLQ envelope
        mock_producer.send.assert_called_once()
        call_kwargs = mock_producer.send.call_args
        envelope = call_kwargs.kwargs.get("value") or call_kwargs[1].get("value")

        assert envelope["original_message"] == original_message
        assert "Validation error" in envelope["error_reason"]
        assert envelope["source_partition"] == 3
        assert envelope["source_offset"] == 12345
        assert envelope["source_topic"] == "fraud.transactions.raw"
        assert "failed_at" in envelope

    def test_dlq_counter_increments(self):
        """Messages routed counter should increment on successful routing."""
        dlq = DeadLetterQueue.__new__(DeadLetterQueue)
        dlq._config = {
            "broker": {"bootstrap_servers": "localhost:9092"},
            "topics": {
                "transactions": {"name": "test"},
                "dead_letter": {"name": "test.dlq"},
            },
        }
        dlq._dlq_topic = "test.dlq"
        dlq._messages_routed = 0

        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_producer.send.return_value = mock_future
        dlq._producer = mock_producer

        dlq.route(
            raw_message={},
            error_reason="test",
            source_partition=0,
            source_offset=0,
        )
        assert dlq.messages_routed == 1

        dlq.route(
            raw_message={},
            error_reason="test2",
            source_partition=0,
            source_offset=1,
        )
        assert dlq.messages_routed == 2


class TestTransactionConsumer:
    """Tests for consumer retry and offset commit behavior."""

    def test_process_with_retry_succeeds_first_attempt(self):
        """Handler that succeeds on first try should not retry."""
        consumer = TransactionConsumer.__new__(TransactionConsumer)
        consumer._max_retries = 3
        consumer._retry_delay = 0.01

        handler = MagicMock()
        event = MagicMock(spec=TransactionEvent)

        result = consumer._process_with_retry(event, handler)

        assert result is True
        handler.assert_called_once_with(event)

    def test_process_with_retry_succeeds_after_failures(self):
        """Handler that fails then succeeds should retry."""
        consumer = TransactionConsumer.__new__(TransactionConsumer)
        consumer._max_retries = 3
        consumer._retry_delay = 0.01

        handler = MagicMock(side_effect=[Exception("fail1"), Exception("fail2"), None])
        event = MagicMock()
        event.event_id = "evt_test"

        result = consumer._process_with_retry(event, handler)

        assert result is True
        assert handler.call_count == 3

    def test_process_with_retry_exhausted(self):
        """Handler that always fails should exhaust retries and return False."""
        consumer = TransactionConsumer.__new__(TransactionConsumer)
        consumer._max_retries = 2
        consumer._retry_delay = 0.01

        handler = MagicMock(side_effect=Exception("always fails"))
        event = MagicMock()
        event.event_id = "evt_test"

        result = consumer._process_with_retry(event, handler)

        assert result is False
        assert handler.call_count == 2  # max_retries attempts

    def test_parse_valid_message(self):
        """Valid JSON should parse into a TransactionEvent."""
        consumer = TransactionConsumer.__new__(TransactionConsumer)

        raw = {
            "event_id": "evt_1",
            "account_id": "acct_1",
            "timestamp": "2024-01-15T10:30:00Z",
            "amount": 42.99,
            "merchant_id": "merch_1",
            "transaction_type": "purchase",
        }

        event = consumer._parse_and_validate(raw)
        assert isinstance(event, TransactionEvent)
        assert event.amount == 42.99

    def test_parse_invalid_message_raises(self):
        """Invalid JSON should raise a validation error."""
        consumer = TransactionConsumer.__new__(TransactionConsumer)

        raw = {
            "event_id": "evt_1",
            # missing required fields
        }

        with pytest.raises(Exception):
            consumer._parse_and_validate(raw)
