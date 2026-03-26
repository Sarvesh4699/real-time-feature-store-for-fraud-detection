-- =============================================================================
-- dbt Model: feature_account_rolling
-- Description: Compute rolling window features for trend analysis.
--              Stores metric_name/metric_value pairs for flexible querying.
-- Materialization: incremental (append new dates, don't recompute old)
-- =============================================================================

{{
    config(
        materialized='incremental',
        schema='feature_store',
        alias='account_features_rolling',
        unique_key=['account_id', 'computation_date', 'window_days', 'metric_name'],
        cluster_by=['computation_date']
    )
}}

{% set target_date = var('target_date', modules.datetime.date.today().isoformat()) %}

WITH txn_base AS (
    SELECT
        account_id,
        transaction_date,
        amount,
        merchant_id
    FROM {{ ref('staging_transactions') }}
    WHERE transaction_date BETWEEN DATEADD(day, -90, '{{ target_date }}')
          AND '{{ target_date }}'
),

-- Unpivot across multiple window sizes
rolling_metrics AS (
    {% for window_days in [7, 30, 90] %}
    SELECT
        account_id,
        '{{ target_date }}'::DATE AS computation_date,
        {{ window_days }} AS window_days,
        'avg_amount' AS metric_name,
        AVG(amount) AS metric_value
    FROM txn_base
    WHERE transaction_date >= DATEADD(day, -{{ window_days }}, '{{ target_date }}')
    GROUP BY account_id

    UNION ALL

    SELECT
        account_id,
        '{{ target_date }}'::DATE AS computation_date,
        {{ window_days }} AS window_days,
        'txn_count' AS metric_name,
        COUNT(*)::FLOAT AS metric_value
    FROM txn_base
    WHERE transaction_date >= DATEADD(day, -{{ window_days }}, '{{ target_date }}')
    GROUP BY account_id

    UNION ALL

    SELECT
        account_id,
        '{{ target_date }}'::DATE AS computation_date,
        {{ window_days }} AS window_days,
        'distinct_merchants' AS metric_name,
        COUNT(DISTINCT merchant_id)::FLOAT AS metric_value
    FROM txn_base
    WHERE transaction_date >= DATEADD(day, -{{ window_days }}, '{{ target_date }}')
    GROUP BY account_id

    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)

SELECT
    account_id,
    computation_date,
    window_days,
    metric_name,
    COALESCE(metric_value, 0) AS metric_value,
    CURRENT_TIMESTAMP() AS computed_at
FROM rolling_metrics

{% if is_incremental() %}
WHERE computation_date > (SELECT MAX(computation_date) FROM {{ this }})
{% endif %}
