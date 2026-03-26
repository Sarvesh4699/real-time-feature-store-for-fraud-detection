-- =============================================================================
-- dbt Model: staging_transactions
-- Description: Clean and standardize raw transaction data for feature computation.
-- Materialization: view (lightweight, always reads fresh data)
-- =============================================================================

{{
    config(
        materialized='view',
        schema='feature_store'
    )
}}

WITH source AS (
    SELECT
        event_id,
        account_id,
        event_timestamp,
        transaction_date,
        amount,
        merchant_id,
        COALESCE(LOWER(merchant_category), 'unknown') AS merchant_category,
        LOWER(transaction_type) AS transaction_type,
        latitude,
        longitude,
        COALESCE(is_online, FALSE) AS is_online,
        COALESCE(card_present, TRUE) AS card_present,
        COALESCE(country_code, 'US') AS country_code,
        ingested_at
    FROM {{ source('raw_data', 'transactions') }}
    WHERE
        -- Basic data quality filters
        amount >= 0
        AND account_id IS NOT NULL
        AND event_timestamp IS NOT NULL
        -- Exclude future-dated transactions (data quality issue)
        AND event_timestamp <= CURRENT_TIMESTAMP()
)

SELECT * FROM source
