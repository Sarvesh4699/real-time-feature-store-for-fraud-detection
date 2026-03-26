"""
Feature registry — central metadata catalog for all features.

The registry tracks feature definitions, data types, computation paths,
and ownership. It serves as documentation-as-code and is used by:
- The streaming processor to know which features to compute
- The batch pipeline to validate output schemas
- Monitoring to check feature freshness
- Data scientists to discover available features

This is intentionally a code-based registry (not a database) because
feature definitions change infrequently and should be version-controlled
alongside the computation logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FeaturePath(str, Enum):
    """Which computation path produces this feature."""
    HOT = "hot"      # Streaming (Kafka → Spark → Redis)
    COLD = "cold"    # Batch (Airflow → dbt → Snowflake → Redis)
    BOTH = "both"    # Computed on both paths (for consistency validation)


class FeatureDataType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"


@dataclass
class FeatureDefinition:
    """Metadata for a single feature."""
    name: str
    path: FeaturePath
    data_type: FeatureDataType
    description: str
    window: Optional[str] = None          # e.g., "1h", "30d", "point"
    owner: str = "fraud-detection-team"
    is_deprecated: bool = False
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)


# ─── Feature Registry ───────────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, FeatureDefinition] = {
    # --- Hot Path (Real-Time) Features ---
    "txn_count_1h": FeatureDefinition(
        name="txn_count_1h",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.INT,
        description="Number of transactions in the last hour",
        window="1h",
        tags=["velocity", "fraud-signal"],
    ),
    "txn_amount_sum_1h": FeatureDefinition(
        name="txn_amount_sum_1h",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.FLOAT,
        description="Sum of transaction amounts in the last hour",
        window="1h",
        tags=["velocity", "amount"],
    ),
    "txn_amount_max_1h": FeatureDefinition(
        name="txn_amount_max_1h",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.FLOAT,
        description="Maximum single transaction amount in the last hour",
        window="1h",
        tags=["velocity", "amount", "anomaly"],
    ),
    "txn_velocity_5m": FeatureDefinition(
        name="txn_velocity_5m",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.FLOAT,
        description="Transactions per minute over a 5-minute sliding window",
        window="5m",
        tags=["velocity", "fraud-signal"],
    ),
    "distance_from_home": FeatureDefinition(
        name="distance_from_home",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.FLOAT,
        description="Haversine distance from account home location in km",
        window="point",
        tags=["geo", "fraud-signal"],
    ),
    "is_new_merchant": FeatureDefinition(
        name="is_new_merchant",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.BOOL,
        description="Whether this is the first transaction with this merchant",
        window="point",
        tags=["behavioral", "fraud-signal"],
    ),
    "hour_of_day": FeatureDefinition(
        name="hour_of_day",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.INT,
        description="Local hour of day (0-23) of the transaction",
        window="point",
        tags=["temporal"],
    ),
    "is_weekend": FeatureDefinition(
        name="is_weekend",
        path=FeaturePath.HOT,
        data_type=FeatureDataType.BOOL,
        description="Whether the transaction occurred on a weekend",
        window="point",
        tags=["temporal"],
    ),

    # --- Cold Path (Batch) Features ---
    "avg_txn_amount_30d": FeatureDefinition(
        name="avg_txn_amount_30d",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.FLOAT,
        description="Average transaction amount over the last 30 days",
        window="30d",
        tags=["baseline", "amount"],
    ),
    "txn_count_30d": FeatureDefinition(
        name="txn_count_30d",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.INT,
        description="Total number of transactions in the last 30 days",
        window="30d",
        tags=["baseline", "velocity"],
    ),
    "distinct_merchants_7d": FeatureDefinition(
        name="distinct_merchants_7d",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.INT,
        description="Number of unique merchants in the last 7 days",
        window="7d",
        tags=["behavioral", "diversity"],
    ),
    "distinct_merchants_30d": FeatureDefinition(
        name="distinct_merchants_30d",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.INT,
        description="Number of unique merchants in the last 30 days",
        window="30d",
        tags=["behavioral", "diversity"],
    ),
    "txn_amount_p95_90d": FeatureDefinition(
        name="txn_amount_p95_90d",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.FLOAT,
        description="95th percentile of transaction amounts over 90 days",
        window="90d",
        tags=["percentile", "anomaly", "fraud-signal"],
    ),
    "txn_amount_p99_90d": FeatureDefinition(
        name="txn_amount_p99_90d",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.FLOAT,
        description="99th percentile of transaction amounts over 90 days",
        window="90d",
        tags=["percentile", "anomaly"],
    ),
    "days_since_last_txn": FeatureDefinition(
        name="days_since_last_txn",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.FLOAT,
        description="Days since the account's last transaction",
        window="point",
        tags=["behavioral", "dormancy"],
    ),
    "pct_online_txns_30d": FeatureDefinition(
        name="pct_online_txns_30d",
        path=FeaturePath.COLD,
        data_type=FeatureDataType.FLOAT,
        description="Percentage of online transactions in the last 30 days",
        window="30d",
        tags=["channel", "behavioral"],
    ),
}


def get_features_by_path(path: FeaturePath) -> list[FeatureDefinition]:
    """Get all features computed on a given path."""
    return [f for f in FEATURE_REGISTRY.values() if f.path == path and not f.is_deprecated]


def get_features_by_tag(tag: str) -> list[FeatureDefinition]:
    """Get all features with a given tag."""
    return [f for f in FEATURE_REGISTRY.values() if tag in f.tags and not f.is_deprecated]


def list_all_features() -> list[FeatureDefinition]:
    """List all active (non-deprecated) features."""
    return [f for f in FEATURE_REGISTRY.values() if not f.is_deprecated]
