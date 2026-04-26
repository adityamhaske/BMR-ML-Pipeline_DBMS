"""
Redshift / DuckDB Loader — ETL Pillar
=======================================
Handles bulk loading of transformed data into the data warehouse.

Dual-mode:
  - DuckDB:   local development (zero infrastructure)
  - Redshift: production (AWS COPY from S3, columnar storage)

Both implement idempotent MERGE semantics — re-running is always safe.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import duckdb
import pandas as pd
from loguru import logger


class BaseLoader(ABC):
    """Abstract loader interface."""

    @abstractmethod
    def load_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        schema: str = "public",
        conflict_key: str | None = None,
    ) -> int:
        """Load a DataFrame into the warehouse. Returns rows loaded."""
        ...

    @abstractmethod
    def execute(self, sql: str, params: tuple | None = None) -> Any:
        """Execute raw SQL."""
        ...

    @abstractmethod
    def close(self) -> None: ...


class DuckDBLoader(BaseLoader):
    """
    Local dev loader using DuckDB.

    Supports Parquet, Arrow, and Pandas DataFrames.
    Uses DELETE + INSERT for idempotent loads (no MERGE in DuckDB by default).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.getenv("DUCKDB_PATH", "data/local/warehouse.duckdb")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = duckdb.connect(self.db_path)
        logger.info(f"DuckDBLoader connected | db={self.db_path}")

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        schema: str = "main",
        conflict_key: str | None = None,
    ) -> int:
        """
        Load a pandas DataFrame into a DuckDB table.

        If the table exists and conflict_key is provided, deletes matching
        rows first (idempotent). Creates the table if it doesn't exist.
        """
        full_table = f"{schema}.{table}" if schema != "main" else table

        # Auto-create schema
        if schema != "main":
            self._conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        # Create table from DataFrame schema if not exists
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {full_table} AS
            SELECT * FROM df WHERE 1=0
        """)

        # Idempotent: delete existing rows matching conflict key
        if conflict_key and conflict_key in df.columns:
            values = df[conflict_key].tolist()
            placeholders = ", ".join(["?" for _ in values])
            self._conn.execute(
                f"DELETE FROM {full_table} WHERE {conflict_key} IN ({placeholders})",
                values,
            )
            logger.debug(f"Deleted {len(values)} existing rows from {full_table}")

        # Insert
        self._conn.execute(f"INSERT INTO {full_table} SELECT * FROM df")
        row_count = len(df)
        logger.info(f"DuckDB loaded {row_count:,} rows → {full_table}")
        return row_count

    def load_parquet(self, parquet_path: str, table: str, schema: str = "main") -> int:
        """Load a parquet file directly into DuckDB (very fast, no pandas overhead)."""
        full_table = f"{schema}.{table}" if schema != "main" else table
        if schema != "main":
            self._conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {full_table} AS
            SELECT * FROM read_parquet('{parquet_path}') WHERE 1=0
        """)
        result = self._conn.execute(f"""
            INSERT INTO {full_table}
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        row_count = self._conn.execute(f"SELECT COUNT(*) FROM {full_table}").fetchone()[0]
        logger.info(f"DuckDB parquet load | {row_count:,} rows → {full_table}")
        return row_count

    def execute(self, sql: str, params: tuple | None = None) -> Any:
        return self._conn.execute(sql, params or [])

    def query(self, sql: str) -> pd.DataFrame:
        return self._conn.execute(sql).df()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DuckDBLoader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class RedshiftLoader(BaseLoader):
    """
    Production loader for AWS Redshift.

    Uses COPY command from S3 for maximum throughput (~1M rows/minute).
    Implements MERGE via DELETE + INSERT for idempotent loads.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        iam_role: str | None = None,
    ) -> None:
        import psycopg2

        self._conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password,
            connect_timeout=30,
        )
        self._conn.autocommit = False
        self._iam_role = iam_role
        logger.info(f"RedshiftLoader connected | host={host} | db={database}")

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        schema: str = "public",
        conflict_key: str | None = None,
    ) -> int:
        """
        Load a small DataFrame via psycopg2 execute_values.
        For large loads, use copy_from_s3() instead.
        """
        import psycopg2.extras

        full_table = f"{schema}.{table}"
        cols = list(df.columns)
        col_str = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        values = [tuple(row) for row in df.itertuples(index=False, name=None)]

        if conflict_key and conflict_key in df.columns:
            unique_vals = df[conflict_key].tolist()
            placeholders_del = ", ".join(["%s"] * len(unique_vals))
            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {full_table} WHERE {conflict_key} IN ({placeholders_del})",
                    unique_vals,
                )

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {full_table} ({col_str}) VALUES %s",
                values,
            )
        self._conn.commit()
        logger.info(f"Redshift loaded {len(df):,} rows → {full_table}")
        return len(df)

    def copy_from_s3(
        self,
        s3_uri: str,
        table: str,
        schema: str = "public",
        file_format: str = "PARQUET",
        delete_condition: str | None = None,
    ) -> None:
        """
        High-throughput bulk load using Redshift COPY from S3.

        Args:
            s3_uri: e.g. 's3://bucket/prefix/'
            table: Target table name
            schema: Target schema
            file_format: PARQUET | CSV | JSON
            delete_condition: SQL WHERE clause for idempotent delete before copy
        """
        if not self._iam_role:
            raise ValueError("iam_role is required for Redshift COPY from S3")

        full_table = f"{schema}.{table}"

        with self._conn.cursor() as cur:
            if delete_condition:
                cur.execute(f"DELETE FROM {full_table} WHERE {delete_condition}")
                logger.info(f"Deleted existing rows: WHERE {delete_condition}")

            copy_sql = f"""
                COPY {full_table}
                FROM '{s3_uri}'
                IAM_ROLE '{self._iam_role}'
                FORMAT AS {file_format}
                SERIALIZETOJSON
            """
            cur.execute(copy_sql)

        self._conn.commit()
        logger.info(f"Redshift COPY complete | {s3_uri} → {full_table}")

    def execute(self, sql: str, params: tuple | None = None) -> Any:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            self._conn.commit()
            try:
                return cur.fetchall()
            except Exception:
                return None

    def close(self) -> None:
        self._conn.close()


def get_loader() -> BaseLoader:
    """
    Factory: returns DuckDB loader for local dev, Redshift loader for production.
    Controlled by USE_LOCALSTACK environment variable.
    """
    use_local = os.getenv("USE_LOCALSTACK", "true").lower() == "true"

    if use_local:
        return DuckDBLoader()

    return RedshiftLoader(
        host=os.environ["REDSHIFT_HOST"],
        port=int(os.environ.get("REDSHIFT_PORT", "5439")),
        database=os.environ["REDSHIFT_DB"],
        user=os.environ["REDSHIFT_USER"],
        password=os.environ["REDSHIFT_PASSWORD"],
        iam_role=os.environ.get("REDSHIFT_IAM_ROLE"),
    )
