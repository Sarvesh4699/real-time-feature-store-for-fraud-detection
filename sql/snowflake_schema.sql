-- =============================================================================
-- Snowflake Schema: Fraud Detection Feature Store
-- =============================================================================
-- This DDL creates the database, schema, and tables for the cold-path
-- feature pipeline. Run once during initial setup.
-- =============================================================================

-- Create database and schema
CREATE DATABASE IF NOT EXISTS FRAUD_DETECTION;
USE DATABASE FRAUD_DETECTION;

CREATE SCHEMA IF NOT EXISTS RAW_DATA;
CREATE SCHEMA IF NOT EXISTS FEATURE_STORE;

-- =============================================================================
-- RAW DATA TABLES
-- =============================================================================

-- Raw transaction events (loaded from Kafka via a CDC connector or batch ingest)
CREATE TABLE IF NOT EXISTS RAW_DATA.TRANSACTIONS (
    event_id            VARCHAR(64)     NOT NULL,
    account_id          VARCHAR(64)     NOT NULL,
    event_timestamp     TIMESTAMP_NTZ   NOT NULL,
    transaction_date    DATE            NOT NULL,  -- Derived from event_timestamp, used for partitioning
    amount              FLOAT           NOT NULL,
    merchant_id         VARCHAR(64)     NOT NULL,
    merchant_category   VARCHAR(64)     DEFAULT 'unknown',
    transaction_type    VARCHAR(20)     NOT NULL,
    latitude            FLOAT,
    longitude           FLOAT,
    is_online           BOOLEAN         DEFAULT FALSE,
    card_present        BOOLEAN         DEFAULT TRUE,
    country_code        VARCHAR(3)      DEFAULT 'US',
    ingested_at         TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),

    -- Clustering key on (transaction_date, account_id) for efficient
    -- range scans in batch feature computation
    CONSTRAINT pk_transactions PRIMARY KEY (event_id)
)
CLUSTER BY (transaction_date, account_id);

-- Account profiles (home location, registration date, etc.)
CREATE TABLE IF NOT EXISTS RAW_DATA.ACCOUNT_PROFILES (
    account_id          VARCHAR(64)     NOT NULL PRIMARY KEY,
    home_latitude       FLOAT,
    home_longitude      FLOAT,
    registration_date   DATE,
    account_type        VARCHAR(20)     DEFAULT 'personal',
    country_code        VARCHAR(3)      DEFAULT 'US',
    updated_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP()
);

-- =============================================================================
-- FEATURE STORE TABLES
-- =============================================================================

-- Daily computed batch features per account
-- This is the target table for dbt's feature_account_daily model.
CREATE TABLE IF NOT EXISTS FEATURE_STORE.ACCOUNT_FEATURES_DAILY (
    account_id                      VARCHAR(64)     NOT NULL,
    computation_date                DATE            NOT NULL,
    computed_at                     TIMESTAMP_NTZ   NOT NULL,

    -- 30-day aggregates
    avg_txn_amount_30d              FLOAT           DEFAULT 0,
    txn_count_30d                   INT             DEFAULT 0,
    distinct_merchants_30d          INT             DEFAULT 0,

    -- 7-day aggregates
    distinct_merchants_7d           INT             DEFAULT 0,
    txn_count_7d                    INT             DEFAULT 0,

    -- 90-day percentiles
    txn_amount_p50_90d              FLOAT           DEFAULT 0,
    txn_amount_p95_90d              FLOAT           DEFAULT 0,
    txn_amount_p99_90d              FLOAT           DEFAULT 0,

    -- Behavioral baselines
    days_since_last_txn             FLOAT           DEFAULT 0,
    avg_daily_txn_count_30d         FLOAT           DEFAULT 0,
    most_common_merchant_category   VARCHAR(64)     DEFAULT 'unknown',
    pct_online_txns_30d             FLOAT           DEFAULT 0,

    CONSTRAINT pk_features_daily PRIMARY KEY (account_id, computation_date)
)
CLUSTER BY (computation_date, account_id);

-- Rolling feature table with longer history (for trend analysis)
CREATE TABLE IF NOT EXISTS FEATURE_STORE.ACCOUNT_FEATURES_ROLLING (
    account_id                      VARCHAR(64)     NOT NULL,
    computation_date                DATE            NOT NULL,
    window_days                     INT             NOT NULL,  -- 7, 30, or 90
    metric_name                     VARCHAR(64)     NOT NULL,
    metric_value                    FLOAT           NOT NULL,
    computed_at                     TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT pk_features_rolling PRIMARY KEY (account_id, computation_date, window_days, metric_name)
)
CLUSTER BY (computation_date);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Latest features per account (convenience view)
CREATE OR REPLACE VIEW FEATURE_STORE.LATEST_ACCOUNT_FEATURES AS
SELECT *
FROM FEATURE_STORE.ACCOUNT_FEATURES_DAILY
WHERE computation_date = (
    SELECT MAX(computation_date)
    FROM FEATURE_STORE.ACCOUNT_FEATURES_DAILY
);

-- =============================================================================
-- GRANTS
-- =============================================================================

-- Grant usage to the feature store service role
GRANT USAGE ON DATABASE FRAUD_DETECTION TO ROLE DATA_ENGINEER;
GRANT USAGE ON SCHEMA FRAUD_DETECTION.RAW_DATA TO ROLE DATA_ENGINEER;
GRANT USAGE ON SCHEMA FRAUD_DETECTION.FEATURE_STORE TO ROLE DATA_ENGINEER;
GRANT SELECT ON ALL TABLES IN SCHEMA FRAUD_DETECTION.RAW_DATA TO ROLE DATA_ENGINEER;
GRANT ALL ON ALL TABLES IN SCHEMA FRAUD_DETECTION.FEATURE_STORE TO ROLE DATA_ENGINEER;
