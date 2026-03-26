"""
Spark Structured Streaming job — the hot path of the feature store.

This job continuously reads transaction events from Kafka, computes
real-time features using windowed aggregations, and writes the results
to both Redis (for serving) and Iceberg (for time-travel snapshots).

Pipeline: Kafka → Spark Structured Streaming → Redis + Iceberg

Performance target: sub-500ms end-to-end feature latency from event
arrival in Kafka to feature availability in Redis.

Key design decisions:
- Micro-batch trigger of 10 seconds: Balances freshness vs throughput.
  Continuous processing mode would give lower latency but at much higher
  resource cost for marginal benefit.
- Watermark of 30 seconds: Allows for late-arriving events (network
  delays, mobile apps) while keeping state store size manageable.
- foreachBatch sink: Gives us control to write to both Redis and Iceberg
  in a single micro-batch, ensuring consistency.

Usage:
    spark-submit \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \\
        src/streaming/spark_streaming_job.py
"""

import os
import sys
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.streaming.stream_processor import StreamFeatureProcessor
from src.utils.config import get_kafka_config
from src.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("streaming.spark_job")

# ─── Schema for Kafka message value ─────────────────────────────────────────
# Must match TransactionEvent Pydantic model exactly.

TRANSACTION_SCHEMA = StructType([
    StructField("event_id", StringType(), nullable=False),
    StructField("account_id", StringType(), nullable=False),
    StructField("timestamp", TimestampType(), nullable=False),
    StructField("amount", DoubleType(), nullable=False),
    StructField("merchant_id", StringType(), nullable=False),
    StructField("merchant_category", StringType(), nullable=True),
    StructField("transaction_type", StringType(), nullable=False),
    StructField("latitude", DoubleType(), nullable=True),
    StructField("longitude", DoubleType(), nullable=True),
    StructField("is_online", BooleanType(), nullable=True),
    StructField("card_present", BooleanType(), nullable=True),
    StructField("country_code", StringType(), nullable=True),
])


def create_spark_session() -> SparkSession:
    """Create a SparkSession configured for streaming with Kafka and Iceberg."""
    spark_master = os.environ.get("SPARK_MASTER", "local[4]")
    app_name = os.environ.get("SPARK_APP_NAME", "fraud-feature-streaming")

    spark = (
        SparkSession.builder
        .master(spark_master)
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")  # Match Kafka partitions
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.stateStore.stateSchemaCheck", "false")
        # Iceberg configs
        .config("spark.sql.catalog.iceberg_catalog", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg_catalog.type", "rest")
        .config(
            "spark.sql.catalog.iceberg_catalog.uri",
            os.environ.get("ICEBERG_CATALOG_URI", "http://localhost:8181"),
        )
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info("spark_session_created", master=spark_master, app_name=app_name)
    return spark


def read_kafka_stream(spark: SparkSession) -> DataFrame:
    """Read a streaming DataFrame from the Kafka transactions topic.

    Returns a DataFrame with parsed transaction columns (not raw Kafka
    key/value bytes). The Kafka offset is preserved for checkpointing.
    """
    kafka_config = get_kafka_config()
    bootstrap_servers = kafka_config["broker"]["bootstrap_servers"]
    topic = kafka_config["topics"]["transactions"]["name"]

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 100000)  # Backpressure: max events per micro-batch
        .load()
    )

    # Parse JSON value into structured columns
    parsed = (
        raw_stream
        .selectExpr("CAST(key AS STRING) AS kafka_key", "CAST(value AS STRING) AS json_value")
        .select(
            F.col("kafka_key"),
            F.from_json(F.col("json_value"), TRANSACTION_SCHEMA).alias("data"),
        )
        .select("kafka_key", "data.*")
    )

    logger.info(
        "kafka_stream_initialized",
        bootstrap_servers=bootstrap_servers,
        topic=topic,
    )
    return parsed


def compute_windowed_features(stream_df: DataFrame) -> DataFrame:
    """Apply windowed aggregations for real-time features.

    Uses Spark's built-in windowing with watermarks for stateful
    per-account aggregations. The watermark of 30 seconds tells Spark
    how late data can arrive before being dropped.

    Two window sizes:
    - 1 hour: For count, sum, max features
    - 5 minutes: For velocity calculation
    """
    watermark_delay = os.environ.get("SPARK_WATERMARK_DELAY", "30 seconds")

    watermarked = stream_df.withWatermark("timestamp", watermark_delay)

    # 1-hour window aggregates grouped by account
    features_1h = (
        watermarked
        .groupBy(
            F.col("account_id"),
            F.window(F.col("timestamp"), "1 hour", "10 seconds"),
        )
        .agg(
            F.count("*").alias("txn_count_1h"),
            F.sum("amount").alias("txn_amount_sum_1h"),
            F.max("amount").alias("txn_amount_max_1h"),
            F.last("merchant_id").alias("last_merchant_id"),
            F.last("latitude").alias("last_latitude"),
            F.last("longitude").alias("last_longitude"),
            F.last("is_online").alias("last_is_online"),
            F.last("card_present").alias("last_card_present"),
            F.last("country_code").alias("last_country_code"),
            F.last("timestamp").alias("last_event_time"),
        )
        .select(
            "account_id",
            "txn_count_1h",
            F.round("txn_amount_sum_1h", 2).alias("txn_amount_sum_1h"),
            F.round("txn_amount_max_1h", 2).alias("txn_amount_max_1h"),
            "last_merchant_id",
            "last_latitude",
            "last_longitude",
            "last_is_online",
            "last_card_present",
            "last_country_code",
            "last_event_time",
            F.col("window.end").alias("window_end"),
        )
    )

    return features_1h


def process_micro_batch(batch_df: DataFrame, batch_id: int) -> None:
    """Process a single micro-batch: compute features and write to Redis + Iceberg.

    This function is called by foreachBatch for every micro-batch.
    It uses the StreamFeatureProcessor to handle the actual feature
    computation and storage writes.
    """
    if batch_df.isEmpty():
        return

    start_time = datetime.utcnow()
    processor = StreamFeatureProcessor()

    try:
        # Collect to driver for Redis writes (features are small per-account structs)
        rows = batch_df.collect()
        features_written = processor.process_batch(rows, batch_id)

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "micro_batch_processed",
            batch_id=batch_id,
            rows_in=len(rows),
            features_written=features_written,
            latency_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(
            "micro_batch_failed",
            batch_id=batch_id,
            error=str(e),
            exc_info=True,
        )
        raise


def main():
    """Entry point for the streaming job."""
    logger.info("starting_streaming_job")

    spark = create_spark_session()
    stream_df = read_kafka_stream(spark)
    features_df = compute_windowed_features(stream_df)

    checkpoint_dir = os.environ.get(
        "SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/fraud-features"
    )
    trigger_interval = os.environ.get("SPARK_TRIGGER_INTERVAL", "10 seconds")

    query = (
        features_df.writeStream
        .foreachBatch(process_micro_batch)
        .outputMode("update")
        .trigger(processingTime=trigger_interval)
        .option("checkpointLocation", checkpoint_dir)
        .start()
    )

    logger.info(
        "streaming_query_started",
        checkpoint_dir=checkpoint_dir,
        trigger_interval=trigger_interval,
    )

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("streaming_job_stopping")
        query.stop()
        spark.stop()
        logger.info("streaming_job_stopped")


if __name__ == "__main__":
    main()
