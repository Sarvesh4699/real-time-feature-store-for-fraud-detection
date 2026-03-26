"""
Snowflake connection manager for the batch feature pipeline.

Handles connection lifecycle, query execution, and result iteration
for the cold-path feature computations. Uses connection pooling and
automatic retry for transient failures.

In production, this connects to Snowflake's FRAUD_DETECTION database
and executes dbt-generated SQL to compute batch features.
"""

import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from src.utils.config import get_snowflake_config
from src.utils.logging_config import get_logger

logger = get_logger("batch.snowflake")


class SnowflakeClient:
    """Manages Snowflake connections and query execution."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_snowflake_config()
        self._connection = None

    def _create_connection(self):
        """Create a new Snowflake connection.

        Uses snowflake-connector-python. In production, prefer key-pair
        authentication over password-based auth.
        """
        try:
            import snowflake.connector

            conn_config = self._config.get("connection", {})
            self._connection = snowflake.connector.connect(
                account=conn_config.get("account"),
                user=conn_config.get("user"),
                password=conn_config.get("password"),
                warehouse=conn_config.get("warehouse"),
                database=conn_config.get("database"),
                schema=conn_config.get("schema"),
                role=conn_config.get("role"),
                login_timeout=30,
                network_timeout=60,
            )
            logger.info(
                "snowflake_connected",
                account=conn_config.get("account"),
                warehouse=conn_config.get("warehouse"),
                database=conn_config.get("database"),
            )
            return self._connection
        except ImportError:
            logger.error("snowflake_connector_not_installed")
            raise
        except Exception as e:
            logger.error("snowflake_connection_failed", error=str(e))
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for Snowflake connections.

        Reuses existing connection if available, creates new one if not.
        Connection is NOT closed on context exit (reuse pattern).
        """
        if self._connection is None or self._connection.is_closed():
            self._create_connection()
        try:
            yield self._connection
        except Exception as e:
            logger.error("snowflake_query_error", error=str(e))
            # Reset connection on error — it may be in a bad state
            self._connection = None
            raise

    def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> list[dict]:
        """Execute a SQL query and return results as list of dicts.

        Args:
            sql: SQL query string.
            params: Optional query parameters for parameterized queries.
            timeout: Query timeout in seconds (defaults to config value).

        Returns:
            List of row dicts with column names as keys.
        """
        query_config = self._config.get("query", {})
        timeout = timeout or query_config.get("timeout_seconds", 600)
        retry_count = query_config.get("retry_count", 2)
        retry_delay = query_config.get("retry_delay_seconds", 10)

        last_error = None
        for attempt in range(1, retry_count + 2):  # +1 for initial attempt
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql, params, timeout=timeout)

                    columns = [desc[0].lower() for desc in cursor.description]
                    results = [dict(zip(columns, row)) for row in cursor.fetchall()]

                    logger.debug(
                        "query_executed",
                        rows_returned=len(results),
                        attempt=attempt,
                    )
                    return results

            except Exception as e:
                last_error = e
                if attempt <= retry_count:
                    logger.warning(
                        "query_retry",
                        attempt=attempt,
                        error=str(e),
                        retry_in_seconds=retry_delay,
                    )
                    time.sleep(retry_delay)

        raise last_error

    def execute_query_iter(
        self,
        sql: str,
        batch_size: int = 10000,
    ) -> Generator[list[dict], None, None]:
        """Execute a query and yield results in batches.

        For large result sets (e.g., all account features), this avoids
        loading everything into memory at once.

        Args:
            sql: SQL query string.
            batch_size: Number of rows per yielded batch.

        Yields:
            Lists of row dicts, each list up to batch_size rows.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)

            columns = [desc[0].lower() for desc in cursor.description]
            batch = []

            for row in cursor:
                batch.append(dict(zip(columns, row)))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

    def close(self) -> None:
        """Close the Snowflake connection."""
        if self._connection is not None and not self._connection.is_closed():
            self._connection.close()
            logger.info("snowflake_connection_closed")
