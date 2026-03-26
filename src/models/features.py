"""
Feature computation logic shared between streaming and batch paths.

CRITICAL DESIGN DECISION: Both the streaming and batch paths import from
this module. This prevents training-serving skew at the definition level.
If you change how a feature is computed here, both paths pick up the change.

Never duplicate feature logic in spark_streaming_job.py or batch_feature_pipeline.py.
Always define the computation here and call it from both paths.
"""

import math
from datetime import datetime, timedelta
from typing import Optional

from src.models.schemas import RealtimeFeatures, TransactionEvent


# ─── Constants ───────────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

# Window sizes for streaming features
WINDOW_1H = timedelta(hours=1)
WINDOW_5M = timedelta(minutes=5)


# ─── Point-in-time Feature Functions ─────────────────────────────────────────


def compute_distance_km(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> Optional[float]:
    """Haversine distance between two lat/lon points in kilometers.

    Returns None if either point is missing coordinates.
    Used for distance_from_home feature — large distances from the
    account's registered address are a strong fraud signal.
    """
    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return None

    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def compute_hour_features(timestamp: datetime) -> tuple[int, bool]:
    """Extract hour-of-day and weekend flag from a timestamp.

    Late-night transactions (1-5 AM) and weekend patterns differ
    significantly from weekday daytime behavior.
    """
    hour = timestamp.hour
    is_weekend = timestamp.weekday() >= 5  # Saturday=5, Sunday=6
    return hour, is_weekend


def check_is_new_merchant(
    merchant_id: str, known_merchants: set[str]
) -> bool:
    """Check if this is the first transaction with a given merchant.

    New-merchant transactions have ~3x higher fraud rate in our training data.
    The known_merchants set is maintained per-account in the streaming state.
    """
    return merchant_id not in known_merchants


# ─── Windowed Feature Functions ──────────────────────────────────────────────


def compute_realtime_features(
    account_id: str,
    current_event: TransactionEvent,
    window_events_1h: list[TransactionEvent],
    window_events_5m: list[TransactionEvent],
    known_merchants: set[str],
    home_lat: Optional[float] = None,
    home_lon: Optional[float] = None,
) -> RealtimeFeatures:
    """Compute all real-time features for a single account.

    This function is called per micro-batch per account by the streaming
    processor. It receives the current event plus all events in the
    1-hour and 5-minute windows from the streaming state store.

    Args:
        account_id: The account being scored.
        current_event: The transaction that triggered this computation.
        window_events_1h: All events for this account in the last hour.
        window_events_5m: All events for this account in the last 5 minutes.
        known_merchants: Set of merchant_ids this account has seen before.
        home_lat: Account's registered home latitude.
        home_lon: Account's registered home longitude.

    Returns:
        RealtimeFeatures with all hot-path features populated.
    """
    # 1-hour window aggregates
    txn_count_1h = len(window_events_1h)
    txn_amount_sum_1h = sum(e.amount for e in window_events_1h)
    txn_amount_max_1h = max((e.amount for e in window_events_1h), default=0.0)

    # 5-minute velocity: transactions per minute
    minutes_in_window = max(len(window_events_5m), 1)
    txn_velocity_5m = len(window_events_5m) / min(5.0, minutes_in_window)

    # Point-in-time features
    hour_of_day, is_weekend = compute_hour_features(current_event.timestamp)
    distance = compute_distance_km(
        current_event.latitude, current_event.longitude, home_lat, home_lon
    )
    is_new = check_is_new_merchant(current_event.merchant_id, known_merchants)

    return RealtimeFeatures(
        account_id=account_id,
        computed_at=datetime.utcnow(),
        txn_count_1h=txn_count_1h,
        txn_amount_sum_1h=round(txn_amount_sum_1h, 2),
        txn_amount_max_1h=round(txn_amount_max_1h, 2),
        txn_velocity_5m=round(txn_velocity_5m, 4),
        distance_from_home=round(distance, 2) if distance is not None else None,
        is_new_merchant=is_new,
        hour_of_day=hour_of_day,
        is_weekend=is_weekend,
        is_online=current_event.is_online,
        card_present=current_event.card_present,
        country_code=current_event.country_code,
    )


# ─── Batch Feature SQL Generators ────────────────────────────────────────────
# These generate the SQL that dbt models execute in Snowflake.
# Defined here (not in .sql files alone) so we can unit-test the logic
# and ensure batch/streaming use identical definitions for shared concepts.


def get_batch_feature_sql(target_date: str) -> str:
    """Generate the SQL for batch feature computation.

    This SQL runs in Snowflake via dbt. It computes expensive aggregates
    over 7d/30d/90d windows that would be too costly for real-time.

    Args:
        target_date: The date to compute features for (YYYY-MM-DD).

    Returns:
        SQL string ready for execution in Snowflake.
    """
    return f"""
    WITH daily_txns AS (
        SELECT
            account_id,
            transaction_date,
            amount,
            merchant_id,
            merchant_category,
            is_online
        FROM {{{{ ref('staging_transactions') }}}}
        WHERE transaction_date BETWEEN DATEADD(day, -90, '{target_date}')
              AND '{target_date}'
    ),
    account_features AS (
        SELECT
            account_id,

            -- 30-day aggregates
            AVG(CASE WHEN transaction_date >= DATEADD(day, -30, '{target_date}')
                THEN amount END) AS avg_txn_amount_30d,
            COUNT(CASE WHEN transaction_date >= DATEADD(day, -30, '{target_date}')
                THEN 1 END) AS txn_count_30d,
            COUNT(DISTINCT CASE WHEN transaction_date >= DATEADD(day, -30, '{target_date}')
                THEN merchant_id END) AS distinct_merchants_30d,

            -- 7-day aggregates
            COUNT(DISTINCT CASE WHEN transaction_date >= DATEADD(day, -7, '{target_date}')
                THEN merchant_id END) AS distinct_merchants_7d,
            COUNT(CASE WHEN transaction_date >= DATEADD(day, -7, '{target_date}')
                THEN 1 END) AS txn_count_7d,

            -- 90-day percentiles
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount) AS txn_amount_p50_90d,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS txn_amount_p95_90d,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) AS txn_amount_p99_90d,

            -- Behavioral baselines
            DATEDIFF(day, MAX(transaction_date), '{target_date}') AS days_since_last_txn,
            COUNT(*) / 30.0 AS avg_daily_txn_count_30d,
            MODE(merchant_category) AS most_common_merchant_category,
            AVG(CASE WHEN is_online THEN 1.0 ELSE 0.0 END) AS pct_online_txns_30d

        FROM daily_txns
        GROUP BY account_id
    )
    SELECT
        account_id,
        CURRENT_TIMESTAMP() AS computed_at,
        COALESCE(avg_txn_amount_30d, 0) AS avg_txn_amount_30d,
        COALESCE(txn_count_30d, 0) AS txn_count_30d,
        COALESCE(distinct_merchants_30d, 0) AS distinct_merchants_30d,
        COALESCE(distinct_merchants_7d, 0) AS distinct_merchants_7d,
        COALESCE(txn_count_7d, 0) AS txn_count_7d,
        COALESCE(txn_amount_p50_90d, 0) AS txn_amount_p50_90d,
        COALESCE(txn_amount_p95_90d, 0) AS txn_amount_p95_90d,
        COALESCE(txn_amount_p99_90d, 0) AS txn_amount_p99_90d,
        COALESCE(days_since_last_txn, 0) AS days_since_last_txn,
        COALESCE(avg_daily_txn_count_30d, 0) AS avg_daily_txn_count_30d,
        COALESCE(most_common_merchant_category, 'unknown') AS most_common_merchant_category,
        COALESCE(pct_online_txns_30d, 0) AS pct_online_txns_30d
    FROM account_features
    """
