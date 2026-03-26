"""
Seed Redis with initial feature vectors from sample data.

Reads sample transaction events, computes both realtime and batch-style
features, and writes them to Redis. Useful for local development and
integration testing without running the full streaming/batch pipelines.

Usage:
    python scripts/seed_redis.py
    python scripts/seed_redis.py --data-file data/sample/transactions.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from src.feature_store.redis_store import RedisFeatureStore
from src.models.schemas import BatchFeatures, RealtimeFeatures, TransactionEvent
from src.utils.logging_config import setup_logging, get_logger

setup_logging(log_format="console")
logger = get_logger("scripts.seed_redis")


def load_events(data_file: str) -> list[TransactionEvent]:
    """Load and validate transaction events from a JSON file."""
    path = Path(data_file)
    if not path.exists():
        print(f"Data file not found: {path}")
        print("Run: python scripts/generate_sample_data.py first")
        return []

    with open(path) as f:
        raw_events = json.load(f)

    events = []
    for raw in raw_events:
        try:
            events.append(TransactionEvent.model_validate(raw))
        except Exception as e:
            logger.warning("event_parse_error", error=str(e))

    return events


def compute_seed_features(
    events: list[TransactionEvent],
) -> tuple[dict[str, RealtimeFeatures], dict[str, BatchFeatures]]:
    """Compute initial features from historical events.

    Groups events by account_id and computes both realtime and batch
    features. This is a simplified version of the actual pipeline
    computations — sufficient for seeding development data.
    """
    # Group by account
    account_events: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        account_events[event.account_id].append(event)

    realtime_features = {}
    batch_features = {}

    for account_id, acct_events in account_events.items():
        acct_events.sort(key=lambda e: e.timestamp)
        latest = acct_events[-1]
        amounts = [e.amount for e in acct_events]
        merchants = set(e.merchant_id for e in acct_events)

        # Realtime features (based on recent events)
        recent_1h = acct_events[-min(10, len(acct_events)):]
        realtime_features[account_id] = RealtimeFeatures(
            account_id=account_id,
            computed_at=datetime.utcnow(),
            txn_count_1h=len(recent_1h),
            txn_amount_sum_1h=round(sum(e.amount for e in recent_1h), 2),
            txn_amount_max_1h=round(max(e.amount for e in recent_1h), 2),
            txn_velocity_5m=round(len(recent_1h) / 60.0, 4),
            is_new_merchant=False,
            hour_of_day=latest.timestamp.hour,
            is_weekend=latest.timestamp.weekday() >= 5,
            is_online=latest.is_online,
            card_present=latest.card_present,
            country_code=latest.country_code,
        )

        # Batch features (based on full history)
        sorted_amounts = sorted(amounts)
        p50_idx = int(len(sorted_amounts) * 0.50)
        p95_idx = int(len(sorted_amounts) * 0.95)
        p99_idx = int(len(sorted_amounts) * 0.99)

        batch_features[account_id] = BatchFeatures(
            account_id=account_id,
            computed_at=datetime.utcnow(),
            avg_txn_amount_30d=round(mean(amounts), 2),
            txn_count_30d=len(acct_events),
            distinct_merchants_30d=len(merchants),
            distinct_merchants_7d=min(len(merchants), len(merchants)),
            txn_count_7d=min(len(acct_events), len(acct_events)),
            txn_amount_p50_90d=round(sorted_amounts[p50_idx], 2),
            txn_amount_p95_90d=round(sorted_amounts[min(p95_idx, len(sorted_amounts) - 1)], 2),
            txn_amount_p99_90d=round(sorted_amounts[min(p99_idx, len(sorted_amounts) - 1)], 2),
            days_since_last_txn=0.0,
            avg_daily_txn_count_30d=round(len(acct_events) / 30.0, 2),
            most_common_merchant_category=max(
                set(e.merchant_category for e in acct_events),
                key=lambda c: sum(1 for e in acct_events if e.merchant_category == c),
            ),
            pct_online_txns_30d=round(
                sum(1 for e in acct_events if e.is_online) / len(acct_events), 2
            ),
        )

    return realtime_features, batch_features


def main():
    parser = argparse.ArgumentParser(description="Seed Redis with initial features")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/sample/transactions.json",
        help="Path to sample transaction data",
    )
    args = parser.parse_args()

    print("Loading sample events...")
    events = load_events(args.data_file)
    if not events:
        return

    print(f"  Loaded {len(events)} events")

    print("Computing features...")
    realtime, batch = compute_seed_features(events)
    print(f"  Computed features for {len(realtime)} accounts")

    print("Writing to Redis...")
    store = RedisFeatureStore()

    try:
        # Write realtime features
        for features in realtime.values():
            store.write_realtime_features(features)

        # Write batch features (bulk)
        store.write_batch_features_bulk(list(batch.values()))

        print(f"  ✓ Written {len(realtime)} realtime feature vectors")
        print(f"  ✓ Written {len(batch)} batch feature vectors")

        # Verify with a spot check
        sample_account = list(realtime.keys())[0]
        fv = store.get_feature_vector(sample_account)
        if fv:
            print(f"\n  Spot check ({sample_account}):")
            print(f"    txn_count_1h: {fv.realtime.txn_count_1h}")
            print(f"    avg_txn_amount_30d: ${fv.batch.avg_txn_amount_30d:.2f}")
            print(f"    distinct_merchants_30d: {fv.batch.distinct_merchants_30d}")

    except Exception as e:
        print(f"\n  ✗ Redis connection failed: {e}")
        print("  Make sure Redis is running on localhost:6379")

    finally:
        store.close()

    print("\nSeeding complete.")


if __name__ == "__main__":
    main()
