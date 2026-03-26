"""
Structured logging configuration using structlog.

All components in the feature store use this shared logging setup for
consistent, JSON-formatted log output. JSON logs are essential for
production — they're parseable by log aggregators (Datadog, Splunk, ELK)
and carry structured metadata (account_id, feature group, latency_ms)
that free-text logs can't reliably provide.
"""

import logging
import os
import sys

import structlog


def setup_logging(
    level: str | None = None,
    log_format: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var or INFO.
        log_format: Output format ('json' or 'console'). Defaults to LOG_FORMAT env var or 'json'.
    """
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    log_format = log_format or os.environ.get("LOG_FORMAT", "json")

    # Shared processors for both structlog and stdlib logging
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Quiet noisy third-party loggers
    for noisy_logger in ("kafka", "urllib3", "snowflake.connector", "botocore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound with the given component name.

    Usage:
        logger = get_logger("streaming.processor")
        logger.info("processing_batch", batch_size=500, latency_ms=45)
    """
    return structlog.get_logger(name)
