"""
Generate realistic sample transaction data for testing.

Creates a JSON file of TransactionEvent records with realistic patterns:
- Multiple accounts with varying activity levels
- Mix of merchants and categories
- Geographic distribution
- Temporal patterns (higher volume during business hours)
- Some anomalous patterns for fraud detection testing

Usage:
    python scripts/generate_sample_data.py --num-events 10000
    python scripts/generate_sample_data.py --num-events 1000 --num-accounts 50
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# ─── Data Generators ─────────────────────────────────────────────────────────

MERCHANTS = [
    ("merch_walmart", "grocery"),
    ("merch_amazon", "online_retail"),
    ("merch_starbucks", "food_beverage"),
    ("merch_shell", "gas_station"),
    ("merch_target", "retail"),
    ("merch_netflix", "subscription"),
    ("merch_uber", "transportation"),
    ("merch_wholefds", "grocery"),
    ("merch_bestbuy", "electronics"),
    ("merch_cvs", "pharmacy"),
    ("merch_homedepot", "home_improvement"),
    ("merch_mcdonalds", "fast_food"),
    ("merch_apple", "electronics"),
    ("merch_costco", "wholesale"),
    ("merch_unknown_001", "unknown"),
    ("merch_unknown_002", "unknown"),
]

TRANSACTION_TYPES = ["purchase", "purchase", "purchase", "payment", "withdrawal", "transfer"]

# Major US metro areas for realistic geo distribution
LOCATIONS = [
    (40.7128, -74.0060),   # New York
    (34.0522, -118.2437),  # Los Angeles
    (41.8781, -87.6298),   # Chicago
    (29.7604, -95.3698),   # Houston
    (33.4484, -112.0740),  # Phoenix
    (39.7392, -104.9903),  # Denver
    (47.6062, -122.3321),  # Seattle
    (25.7617, -80.1918),   # Miami
    (42.3601, -71.0589),   # Boston
    (38.9072, -77.0369),   # Washington DC
]


def generate_account_id(index: int) -> str:
    """Generate a deterministic account ID."""
    return f"acct_{index:05d}"


def generate_event(
    account_id: str,
    base_time: datetime,
    home_location: tuple[float, float],
    is_anomalous: bool = False,
) -> dict:
    """Generate a single transaction event.

    Normal transactions cluster around the home location with typical
    amounts. Anomalous transactions have unusual amounts, locations,
    and timing patterns.
    """
    event_id = f"evt_{uuid.uuid4().hex[:12]}"

    if is_anomalous:
        # Anomalous: large amounts, distant locations, odd hours
        amount = round(random.uniform(500, 5000), 2)
        hour_offset = random.choice([2, 3, 4, 23])  # Late night
        lat = home_location[0] + random.uniform(-10, 10)
        lon = home_location[1] + random.uniform(-10, 10)
        merchant = random.choice(MERCHANTS[-2:])  # Unknown merchants
        is_online = random.random() > 0.3
    else:
        # Normal: typical amounts, near home, business hours
        amount = round(random.lognormvariate(3.0, 1.0), 2)  # Median ~$20
        amount = min(amount, 500)  # Cap at $500
        hour_offset = random.choices(
            range(24),
            weights=[1, 0, 0, 0, 0, 1, 2, 5, 8, 10, 10, 12, 15, 12, 10, 8, 7, 8, 6, 5, 3, 2, 2, 1],
        )[0]
        lat = home_location[0] + random.gauss(0, 0.05)
        lon = home_location[1] + random.gauss(0, 0.05)
        merchant = random.choice(MERCHANTS[:-2])
        is_online = random.random() > 0.7

    timestamp = base_time.replace(hour=hour_offset, minute=random.randint(0, 59))
    timestamp += timedelta(seconds=random.randint(0, 59))

    return {
        "event_id": event_id,
        "account_id": account_id,
        "timestamp": timestamp.isoformat() + "Z",
        "amount": amount,
        "merchant_id": merchant[0],
        "merchant_category": merchant[1],
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "is_online": is_online,
        "card_present": not is_online,
        "country_code": "US",
    }


def generate_dataset(num_events: int, num_accounts: int) -> list[dict]:
    """Generate a full dataset of transaction events.

    Creates accounts with varying activity levels and sprinkles in
    ~5% anomalous transactions for fraud detection testing.
    """
    events = []
    now = datetime.utcnow()

    # Assign each account a home location
    account_homes = {}
    for i in range(num_accounts):
        account_id = generate_account_id(i)
        account_homes[account_id] = random.choice(LOCATIONS)

    # Distribute events across accounts (power-law: some accounts are much more active)
    account_weights = [random.paretovariate(1.5) for _ in range(num_accounts)]
    total_weight = sum(account_weights)
    account_weights = [w / total_weight for w in account_weights]

    for _ in range(num_events):
        # Pick account (weighted)
        account_idx = random.choices(range(num_accounts), weights=account_weights)[0]
        account_id = generate_account_id(account_idx)
        home = account_homes[account_id]

        # Pick a random day in the last 7 days
        days_ago = random.randint(0, 6)
        base_time = now - timedelta(days=days_ago)

        # 5% anomalous transactions
        is_anomalous = random.random() < 0.05

        event = generate_event(account_id, base_time, home, is_anomalous)
        events.append(event)

    # Sort by timestamp for realistic ordering
    events.sort(key=lambda e: e["timestamp"])
    return events


def main():
    parser = argparse.ArgumentParser(description="Generate sample transaction data")
    parser.add_argument("--num-events", type=int, default=10000, help="Number of events to generate")
    parser.add_argument("--num-accounts", type=int, default=200, help="Number of unique accounts")
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample/transactions.json",
        help="Output file path",
    )
    args = parser.parse_args()

    print(f"Generating {args.num_events} events for {args.num_accounts} accounts...")
    events = generate_dataset(args.num_events, args.num_accounts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"Written {len(events)} events to {output_path}")

    # Print summary stats
    accounts = set(e["account_id"] for e in events)
    amounts = [e["amount"] for e in events]
    anomalous = [e for e in events if e["amount"] > 500]
    print(f"  Unique accounts: {len(accounts)}")
    print(f"  Amount range: ${min(amounts):.2f} - ${max(amounts):.2f}")
    print(f"  Avg amount: ${sum(amounts)/len(amounts):.2f}")
    print(f"  High-value transactions (>$500): {len(anomalous)} ({len(anomalous)/len(events)*100:.1f}%)")


if __name__ == "__main__":
    main()
