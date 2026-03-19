"""
S3 Parquet reader for SmartLoader Tier 2 data.

Reads Parquet files from S3, partitioned by market type.
Converts to the same dict structures that the Redis-based loaders produce.
"""

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

import boto3
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Tier 2 table → S3 path mapping
TABLE_PATHS = {
    "hist_adj": "historical_prices_adjusted",
    "hist_raw": "historical_prices_raw",
    "bond_clean": "bond_clean_prices",
    "yield_by_date": "yield_bonds/by_date",
    "yield_by_ticker": "yield_bonds/by_ticker",
}


class ParquetReader:
    """Reads Parquet files from S3 for Tier 2 historical data."""

    def __init__(self, bucket: str, prefix: str = "v1"):
        self._bucket = bucket
        self._prefix = prefix
        self._s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

    def read_ticker(self, table: str, ticker: str) -> List[Dict]:
        """
        Read historical series for a single ticker from S3 Parquet.

        For market-partitioned tables (hist_adj, hist_raw), this reads ALL market
        partitions and filters for the ticker. Consider using read_market_partition
        if you know the market type.

        For ticker-keyed tables (bond_clean, yield_by_ticker), reads the single file
        and extracts the ticker's data.

        Args:
            table: Tier 2 table name (e.g., "hist_adj", "bond_clean")
            ticker: Ticker symbol

        Returns:
            List of record dicts for the ticker, or empty list if not found.
        """
        s3_path = TABLE_PATHS.get(table)
        if not s3_path:
            logger.error(f"Unknown Tier 2 table: {table}")
            return []

        if table in ("hist_adj", "hist_raw"):
            return self._read_ticker_from_partitioned(s3_path, ticker)
        else:
            return self._read_ticker_from_single_file(s3_path, ticker)

    def read_market_partition(self, table: str, market: str) -> Dict[str, List[Dict]]:
        """
        Read all tickers for a market type from a partitioned Parquet file.

        Args:
            table: Tier 2 table name (e.g., "hist_adj")
            market: Market type (e.g., "stock", "cedear", "bond")

        Returns:
            Dict mapping ticker → list of records.
        """
        s3_path = TABLE_PATHS.get(table)
        if not s3_path:
            logger.error(f"Unknown Tier 2 table: {table}")
            return {}

        key = f"{self._prefix}/{s3_path}/market_type={market}/data.parquet"
        df = self._download_parquet(key)
        if df is None or df.empty:
            return {}

        return self._group_by_ticker(df)

    def read_full_table(self, table: str) -> Any:
        """
        Read a non-partitioned Tier 2 table in full.

        Used for tables like yield_by_date where the full dataset is needed.

        Returns:
            The deserialized data structure (dict), or empty dict on failure.
        """
        s3_path = TABLE_PATHS.get(table)
        if not s3_path:
            logger.error(f"Unknown Tier 2 table: {table}")
            return {}

        key = f"{self._prefix}/{s3_path}.parquet"
        df = self._download_parquet(key)
        if df is None or df.empty:
            return {}

        # Convert DataFrame to the nested dict structure expected by consumers
        if table == "yield_by_date":
            return self._to_yield_by_date_structure(df)
        elif table == "yield_by_ticker":
            return self._group_by_ticker(df, key_col="ticker")

        return df.to_dict(orient="records")

    # ── Internal methods ──

    def _read_ticker_from_partitioned(self, s3_path: str, ticker: str) -> List[Dict]:
        """Search across all market partitions for a ticker."""
        # List partitions
        prefix = f"{self._prefix}/{s3_path}/"
        try:
            response = self._s3.list_objects_v2(
                Bucket=self._bucket, Prefix=prefix, Delimiter="/"
            )
            partitions = [
                cp["Prefix"]
                for cp in response.get("CommonPrefixes", [])
            ]
        except Exception as e:
            logger.error(f"Error listing partitions for {s3_path}: {e}")
            return []

        # Search each partition for the ticker
        for partition in partitions:
            key = f"{partition}data.parquet"
            df = self._download_parquet(key)
            if df is not None and not df.empty and "ticker" in df.columns:
                ticker_df = df[df["ticker"] == ticker]
                if not ticker_df.empty:
                    return self._df_to_records(ticker_df)

        return []

    def _read_ticker_from_single_file(self, s3_path: str, ticker: str) -> List[Dict]:
        """Read a single Parquet file and extract one ticker's data."""
        key = f"{self._prefix}/{s3_path}.parquet"
        df = self._download_parquet(key)
        if df is None or df.empty:
            return []

        if "ticker" in df.columns:
            ticker_df = df[df["ticker"] == ticker]
            return self._df_to_records(ticker_df)

        return []

    def _download_parquet(self, key: str) -> Optional["pd.DataFrame"]:
        """Download a Parquet file from S3 and return as DataFrame."""
        import pandas as pd

        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            buf = io.BytesIO(response["Body"].read())
            table = pq.read_table(buf)
            df = table.to_pandas()
            logger.debug(f"Downloaded s3://{self._bucket}/{key}: {len(df)} rows")
            return df
        except self._s3.exceptions.NoSuchKey:
            logger.warning(f"Parquet file not found: s3://{self._bucket}/{key}")
            return None
        except Exception as e:
            logger.error(f"Error downloading s3://{self._bucket}/{key}: {e}")
            return None

    def _group_by_ticker(self, df: "pd.DataFrame", key_col: str = "ticker") -> Dict[str, List[Dict]]:
        """Group DataFrame rows by ticker into {ticker: [records]} structure."""
        result = {}
        if key_col not in df.columns:
            return result

        for ticker, group in df.groupby(key_col):
            result[ticker] = self._df_to_records(group)
        return result

    def _df_to_records(self, df: "pd.DataFrame") -> List[Dict]:
        """Convert DataFrame to list of dicts with proper date serialization."""
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, val in row.items():
                if hasattr(val, "isoformat"):
                    record[col] = val.isoformat()
                elif hasattr(val, "item"):
                    record[col] = val.item()
                else:
                    record[col] = val
            records.append(record)
        return records

    def _to_yield_by_date_structure(self, df: "pd.DataFrame") -> Dict[str, Dict[str, List[Dict]]]:
        """
        Convert yield bonds DataFrame to {date_str: {submarket: [records]}} structure.
        Matches the current DB_TABLES["historical_yield_bonds"] format.
        """
        result = {}
        if "date" not in df.columns:
            return result

        for date_val, date_group in df.groupby("date"):
            date_str = date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val)
            result[date_str] = {}

            if "submarket_type" in df.columns:
                for submarket, sub_group in date_group.groupby("submarket_type"):
                    result[date_str][submarket] = self._df_to_records(sub_group)
            else:
                result[date_str] = self._df_to_records(date_group)

        return result
