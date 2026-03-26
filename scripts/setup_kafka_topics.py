"""
Create Kafka topics with the correct partition and replication configuration.

This script creates the transactions topic (8 partitions, keyed on account_id)
and the dead-letter queue topic (2 partitions). Run once during initial setup.

The 8-partition count for the transactions topic is a deliberate choice:
- Matches the parallelism of our Spark streaming job (8 executor cores)
- Provides enough partitions for per-user ordering via account_id keys
- Not so many that we create excessive overhead for our throughput (~1M events/day)

Usage:
    python scripts/setup_kafka_topics.py
    python scripts/setup_kafka_topics.py --bootstrap-servers kafka1:9092,kafka2:9092
"""

import argparse
import sys

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from src.utils.config import get_kafka_config
from src.utils.logging_config import setup_logging, get_logger

setup_logging(log_format="console")
logger = get_logger("scripts.setup_kafka")


def create_topics(bootstrap_servers: str = None):
    """Create all required Kafka topics."""
    config = get_kafka_config()
    bootstrap = bootstrap_servers or config["broker"]["bootstrap_servers"]

    logger.info("connecting_to_kafka", bootstrap_servers=bootstrap)

    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap,
            client_id="topic-setup-script",
        )
    except Exception as e:
        logger.error("kafka_connection_failed", error=str(e))
        print(f"\nERROR: Cannot connect to Kafka at {bootstrap}")
        print("Make sure Kafka is running and accessible.")
        sys.exit(1)

    topics_to_create = []

    for topic_key in ["transactions", "dead_letter"]:
        topic_config = config["topics"][topic_key]
        topic_name = topic_config["name"]
        num_partitions = topic_config.get("num_partitions", 8)
        replication_factor = topic_config.get("replication_factor", 1)
        extra_config = topic_config.get("config", {})

        # Convert config values to strings (Kafka admin API requirement)
        topic_configs = {k: str(v) for k, v in extra_config.items()}

        topics_to_create.append(
            NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
                topic_configs=topic_configs,
            )
        )

        logger.info(
            "topic_configured",
            name=topic_name,
            partitions=num_partitions,
            replication=replication_factor,
            configs=topic_configs,
        )

    # Create topics
    for topic in topics_to_create:
        try:
            admin.create_topics(new_topics=[topic], validate_only=False)
            print(f"  ✓ Created topic: {topic.name} ({topic.num_partitions} partitions)")
        except TopicAlreadyExistsError:
            print(f"  ○ Topic already exists: {topic.name}")
        except Exception as e:
            print(f"  ✗ Failed to create topic {topic.name}: {e}")

    admin.close()
    print("\nTopic setup complete.")


def main():
    parser = argparse.ArgumentParser(description="Setup Kafka topics")
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=None,
        help="Kafka bootstrap servers (default: from config)",
    )
    args = parser.parse_args()
    create_topics(args.bootstrap_servers)


if __name__ == "__main__":
    main()
