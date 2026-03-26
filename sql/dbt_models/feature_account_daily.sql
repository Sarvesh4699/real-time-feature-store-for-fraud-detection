-- =============================================================================
-- dbt Model: feature_account_daily
-- Description: Compute daily batch features per account.
--              This is the primary batch feature table, refreshed nightly.
-- Materialization: table (full rebuild each run for consistency)
-- =============================================================================

{{
    config(
        materialized='table',
        schema='feature_store',
        alias='account_features_daily',
        cluster_by=['computation_date', 'account_id'],
        post_hook=[
            "ALTER TABLE {{ this }} SET DATA_RETENTION_TIME_IN_DAYS = 90"
        ]
    )
}}

{% set target_date = var('target_date', modules.datetime.date.today().isoformat()) %}

WITH txn_base AS (
    -- Pull 90 days of history for percentile calculations
    SELECT
        account_id,
        transaction_date,
        amount,
        merchant_id,
        merchant_category,
        is_online,
        event_timestamp
    FROM {{ ref('staging_transactions') }}
    WHERE transaction_date BETWEEN DATEADD(day, -90, '{{ target_date }}')
          AND '{{ target_date }}'
),

-- 30-day aggregates
agg_30d AS (
    SELECT
        account_id,
        AVG(amount) AS avg_txn_amount_30d,
        COUNT(*) AS txn_count_30d,
        COUNT(DISTINCT merchant_id) AS distinct_merchants_30d,
        AVG(CASE WHEN is_online THEN 1.0 ELSE 0.0 END) AS pct_online_txns_30d,
        COUNT(*) / 30.0 AS avg_daily_txn_count_30d
    FROM txn_base
    WHERE transaction_date >= DATEADD(day, -30, '{{ target_date }}')
    GROUP BY account_id
),

-- 7-day aggregates
agg_7d AS (
    SELECT
        account_id,
        COUNT(DISTINCT merchant_id) AS distinct_merchants_7d,
        COUNT(*) AS txn_count_7d
    FROM txn_base
    WHERE transaction_date >= DATEADD(day, -7, '{{ target_date }}')
    GROUP BY account_id
),

-- 90-day percentiles (expensive but high signal)
percentiles_90d AS (
    SELECT
        account_id,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount) AS txn_amount_p50_90d,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS txn_amount_p95_90d,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) AS txn_amount_p99_90d
    FROM txn_base
    GROUP BY account_id
),

-- Recency and behavioral features
behavioral AS (
    SELECT
        account_id,
        DATEDIFF(day, MAX(transaction_date), '{{ target_date }}') AS days_since_last_txn,
        MODE(merchant_category) AS most_common_merchant_category
    FROM txn_base
    GROUP BY account_id
),

-- Combine all feature groups
combined AS (
    SELECT
        COALESCE(a30.account_id, a7.account_id, p90.account_id, b.account_id) AS account_id,
        '{{ target_date }}'::DATE AS computation_date,
        CURRENT_TIMESTAMP() AS computed_at,

        -- 30-day
        COALESCE(a30.avg_txn_amount_30d, 0) AS avg_txn_amount_30d,
        COALESCE(a30.txn_count_30d, 0) AS txn_count_30d,
        COALESCE(a30.distinct_merchants_30d, 0) AS distinct_merchants_30d,
        COALESCE(a30.pct_online_txns_30d, 0) AS pct_online_txns_30d,
        COALESCE(a30.avg_daily_txn_count_30d, 0) AS avg_daily_txn_count_30d,

        -- 7-day
        COALESCE(a7.distinct_merchants_7d, 0) AS distinct_merchants_7d,
        COALESCE(a7.txn_count_7d, 0) AS txn_count_7d,

        -- 90-day percentiles
        COALESCE(p90.txn_amount_p50_90d, 0) AS txn_amount_p50_90d,
        COALESCE(p90.txn_amount_p95_90d, 0) AS txn_amount_p95_90d,
        COALESCE(p90.txn_amount_p99_90d, 0) AS txn_amount_p99_90d,

        -- Behavioral
        COALESCE(b.days_since_last_txn, 0) AS days_since_last_txn,
        COALESCE(b.most_common_merchant_category, 'unknown') AS most_common_merchant_category

    FROM agg_30d a30
    FULL OUTER JOIN agg_7d a7 ON a30.account_id = a7.account_id
    FULL OUTER JOIN percentiles_90d p90 ON COALESCE(a30.account_id, a7.account_id) = p90.account_id
    FULL OUTER JOIN behavioral b ON COALESCE(a30.account_id, a7.account_id, p90.account_id) = b.account_id
)

SELECT * FROM combined
