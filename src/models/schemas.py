"""
Pydantic models for transaction events and feature vectors.

These schemas are the contract between all components of the feature store:
- Kafka producers serialize TransactionEvent to JSON
- Streaming consumers deserialize and validate incoming events
- Feature computations output FeatureVector instances
- Redis stores/retrieves serialized FeatureVector JSON

Using Pydantic ensures type safety and catches schema mismatches early
rather than surfacing as silent data quality issues downstream.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TransactionType(str, Enum):
    """Transaction categories. Extensible as new payment types emerge."""
    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    REFUND = "refund"


class TransactionEvent(BaseModel):
    """Raw transaction event as produced to Kafka.

    This is the source-of-truth schema for the transactions topic.
    The Kafka producer validates every event against this model before
    publishing. account_id is used as the partition key.
    """
    event_id: str = Field(..., description="Globally unique event identifier (UUID)")
    account_id: str = Field(..., description="Account identifier, used as Kafka partition key")
    timestamp: datetime = Field(..., description="Event timestamp in UTC")
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(default="unknown", description="MCC category code")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    is_online: bool = Field(default=False, description="Whether transaction was online")
    card_present: bool = Field(default=True, description="Whether physical card was used")
    country_code: str = Field(default="US", max_length=3)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_abc123",
                "account_id": "acct_12345",
                "timestamp": "2024-01-15T10:30:00Z",
                "amount": 42.99,
                "merchant_id": "merch_xyz",
                "merchant_category": "grocery",
                "transaction_type": "purchase",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "is_online": False,
                "card_present": True,
                "country_code": "US",
            }
        }


class RealtimeFeatures(BaseModel):
    """Features computed on the hot path (Spark Structured Streaming).

    These features have sub-second freshness and capture recent behavioral
    patterns. They refresh every micro-batch interval (~10 seconds).
    """
    account_id: str
    computed_at: datetime = Field(default_factory=datetime.utcnow)

    # Velocity features (1-hour window)
    txn_count_1h: int = Field(default=0, description="Transaction count in last hour")
    txn_amount_sum_1h: float = Field(default=0.0, description="Total amount in last hour")
    txn_amount_max_1h: float = Field(default=0.0, description="Max single amount in last hour")

    # Short-window velocity (5-minute)
    txn_velocity_5m: float = Field(
        default=0.0,
        description="Transactions per minute over 5-min window"
    )

    # Point-in-time features
    distance_from_home: Optional[float] = Field(
        default=None,
        description="Haversine distance from home location in km"
    )
    is_new_merchant: bool = Field(
        default=False,
        description="First time transacting with this merchant"
    )
    hour_of_day: int = Field(default=0, ge=0, le=23)
    is_weekend: bool = Field(default=False)

    # Cross-channel signals
    is_online: bool = Field(default=False)
    card_present: bool = Field(default=True)
    country_code: str = Field(default="US")


class BatchFeatures(BaseModel):
    """Features computed on the cold path (Airflow → dbt → Snowflake).

    These features are expensive aggregates over longer windows. They're
    recomputed nightly and backfilled to Redis. Examples: 90-day percentiles,
    30-day averages — computations that require scanning large amounts of
    historical data and aren't feasible in real-time.
    """
    account_id: str
    computed_at: datetime = Field(default_factory=datetime.utcnow)

    # 30-day aggregates
    avg_txn_amount_30d: float = Field(default=0.0)
    txn_count_30d: int = Field(default=0)
    distinct_merchants_30d: int = Field(default=0)

    # 7-day aggregates
    distinct_merchants_7d: int = Field(default=0)
    txn_count_7d: int = Field(default=0)

    # 90-day percentiles (expensive to compute, high signal for anomaly detection)
    txn_amount_p50_90d: float = Field(default=0.0)
    txn_amount_p95_90d: float = Field(default=0.0)
    txn_amount_p99_90d: float = Field(default=0.0)

    # Behavioral baselines
    days_since_last_txn: float = Field(default=0.0)
    avg_daily_txn_count_30d: float = Field(default=0.0)
    most_common_merchant_category: str = Field(default="unknown")
    pct_online_txns_30d: float = Field(default=0.0, ge=0.0, le=1.0)


class FeatureVector(BaseModel):
    """Combined feature vector served to the ML model at inference time.

    This is the union of realtime + batch features. The feature serving layer
    (Redis) stores both halves and merges them into this single struct when
    the model requests features for an account_id.
    """
    account_id: str
    realtime: RealtimeFeatures
    batch: BatchFeatures
    served_at: datetime = Field(default_factory=datetime.utcnow)
    feature_store_version: str = Field(default="1.0.0")

    def to_model_input(self) -> dict:
        """Flatten to a dict suitable for ML model input.

        Returns a single-level dict with all numeric and categorical features
        that the fraud detection model expects.
        """
        rt = self.realtime.model_dump(exclude={"account_id", "computed_at"})
        bt = self.batch.model_dump(exclude={"account_id", "computed_at"})
        return {**rt, **bt, "account_id": self.account_id}
