# Real-Time Feature Store for Fraud Detection

A production-grade dual-path feature store that serves ML features for fraud detection
with sub-500ms latency on the hot path and nightly batch recomputation on the cold path.

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │               HOT PATH (Real-Time)              │
                    │                                                 │
  Transactions ───► │  Kafka (8 partitions, account_id key)           │
                    │    │                                            │
                    │    ▼                                            │
                    │  Spark Structured Streaming                     │
                    │    │                                            │
                    │    ▼                                            │
                    │  Redis (feature serving, <500ms)                │
                    │                                                 │
                    ├─────────────────────────────────────────────────┤
                    │               COLD PATH (Batch)                 │
                    │                                                 │
                    │  Airflow DAG (nightly)                          │
                    │    │                                            │
                    │    ▼                                            │
                    │  dbt models (feature transforms)                │
                    │    │                                            │
                    │    ▼                                            │
                    │  Snowflake (feature warehouse)                  │
                    │    │                                            │
                    │    ▼                                            │
                    │  Redis (backfill serving layer)                 │
                    │                                                 │
                    ├─────────────────────────────────────────────────┤
                    │            TIME-TRAVEL (Debugging)              │
                    │                                                 │
                    │  Apache Iceberg snapshots                       │
                    │    → Reproduce exact feature state at any       │
                    │      past prediction timestamp                  │
                    │    → Eliminates training-serving skew           │
                    │    → Debug time: hours → <5 min                 │
                    └─────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Dual-Path Architecture**: Hot path for real-time features (velocity, recent counts),
   cold path for expensive aggregates (90-day averages, percentiles). Both paths write
   to Redis so the serving layer has a single interface.

2. **Kafka Partitioning**: 8 partitions keyed on `account_id` ensures all events for a
   given user land on the same partition, preserving per-user ordering for accurate
   windowed aggregations. Manual offset commits + dead-letter queue routing guarantee
   zero message loss.

3. **Iceberg Time-Travel**: Every feature computation is snapshotted via Iceberg.
   When investigating a past prediction, we query `AS OF <timestamp>` to retrieve
   the exact feature vector the model saw, reducing skew debugging from hours to minutes.

4. **Feature Consistency**: The same feature definitions (in `src/models/features.py`)
   are used by both streaming and batch paths, preventing training-serving skew at the
   definition level.

## Project Structure

```
fraud-detection-feature-store/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── configs/
│   ├── kafka.yaml          # Kafka broker & topic configuration
│   ├── redis.yaml          # Redis connection settings
│   ├── snowflake.yaml      # Snowflake warehouse config
│   └── iceberg.yaml        # Iceberg catalog config
├── dags/
│   └── feature_batch_dag.py  # Airflow DAG for nightly batch
├── sql/
│   ├── snowflake_schema.sql  # Warehouse table DDL
│   └── dbt_models/
│       ├── staging_transactions.sql
│       ├── feature_account_daily.sql
│       └── feature_account_rolling.sql
├── src/
│   ├── __init__.py
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── spark_streaming_job.py   # Spark Structured Streaming consumer
│   │   └── stream_processor.py      # Feature computation on micro-batches
│   ├── batch/
│   │   ├── __init__.py
│   │   ├── batch_feature_pipeline.py  # Nightly batch feature builder
│   │   └── snowflake_client.py        # Snowflake connection manager
│   ├── kafka/
│   │   ├── __init__.py
│   │   ├── producer.py         # Transaction event producer
│   │   ├── consumer.py         # Low-level consumer with manual commits
│   │   └── dead_letter.py      # DLQ routing for failed messages
│   ├── feature_store/
│   │   ├── __init__.py
│   │   ├── redis_store.py      # Redis read/write for feature serving
│   │   └── feature_registry.py # Central feature metadata registry
│   ├── iceberg/
│   │   ├── __init__.py
│   │   └── time_travel.py      # Iceberg snapshot queries
│   ├── models/
│   │   ├── __init__.py
│   │   ├── features.py         # Shared feature definitions
│   │   └── schemas.py          # Pydantic models for events & features
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # YAML/env config loader
│       └── logging_config.py   # Structured logging setup
├── scripts/
│   ├── generate_sample_data.py  # Creates realistic test transactions
│   ├── setup_kafka_topics.py    # Topic creation with partition config
│   └── seed_redis.py            # Seed Redis with initial features
├── tests/
│   ├── __init__.py
│   ├── test_features.py         # Feature computation unit tests
│   ├── test_kafka_consumer.py   # Consumer offset/DLQ tests
│   ├── test_redis_store.py      # Feature store read/write tests
│   ├── test_schemas.py          # Schema validation tests
│   └── test_time_travel.py      # Iceberg snapshot query tests
└── data/
    └── sample/
        └── transactions.json    # Sample transaction events
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Apache Kafka 3.x (or Docker)
- Redis 7.x (or Docker)
- Apache Spark 3.4+ (for streaming job)
- Snowflake account (for batch path)
- Apache Airflow 2.x (for orchestration)

### 1. Clone and Install

```bash
cd fraud-detection-feature-store
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials (Snowflake, Redis, Kafka brokers)
```

### 3. Infrastructure Setup (Docker)

```bash
# Start Kafka + Zookeeper + Redis locally
docker-compose up -d  # if using docker-compose (not included, use standard images)

# Or manually:
# Kafka: https://kafka.apache.org/quickstart
# Redis: docker run -d -p 6379:6379 redis:7-alpine
```

### 4. Create Kafka Topics

```bash
python scripts/setup_kafka_topics.py
```

### 5. Generate Sample Data

```bash
python scripts/generate_sample_data.py --num-events 10000
```

### 6. Run Streaming Pipeline

```bash
# Submit Spark job
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
  src/streaming/spark_streaming_job.py
```

### 7. Run Batch Pipeline

```bash
# Trigger Airflow DAG manually or wait for nightly schedule
airflow dags trigger feature_batch_daily
```

### 8. Run Tests

```bash
pytest tests/ -v --tb=short
```

## Feature Definitions

| Feature Name | Path | Window | Description |
|---|---|---|---|
| `txn_count_1h` | Hot | 1 hour | Transaction count in last hour |
| `txn_amount_sum_1h` | Hot | 1 hour | Total amount in last hour |
| `txn_velocity_5m` | Hot | 5 min | Transactions per minute (5-min window) |
| `avg_txn_amount_30d` | Cold | 30 days | Average transaction amount |
| `distinct_merchants_7d` | Cold | 7 days | Unique merchants in last week |
| `txn_amount_p95_90d` | Cold | 90 days | 95th percentile amount (90 days) |
| `distance_from_home` | Hot | Point | Geo distance from home location |
| `is_new_merchant` | Hot | Point | First time at this merchant |
| `hour_of_day` | Hot | Point | Local hour of transaction |
| `days_since_last_txn` | Cold | Point | Days since previous transaction |

## Monitoring & Observability

- **Kafka lag**: Monitor consumer group lag via `kafka-consumer-groups.sh`
- **Feature freshness**: Redis TTLs + timestamp fields on every feature vector
- **DLQ depth**: Alert if dead-letter topic exceeds threshold
- **Iceberg snapshots**: Retained for 30 days for time-travel debugging

## License

Internal use only. Not for redistribution.
