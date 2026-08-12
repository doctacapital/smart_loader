"""
S3 Parquet reader for SmartLoader Tier 2 data.

Reads Parquet files from S3, partitioned by market type.
Converts to the same dict structures that the Redis-based loaders produce.

DPM-815 P0-B: `_download_parquet` is the single choke-point through which
every materialization passes (`:90`, `:111`, `:144`, `:155`). It is now
single-flighted (one in-flight download per S3 key, process-wide), memoized
for a short TTL with an explicit purge (so retention is bounded, not just
staleness), and gated by a semaphore that caps concurrent materializations.
State lives at module level because nexus builds two `ParquetReader`
instances (module-level pre-fork + `initialize_cache_from_redis`) that must
share the same in-flight/memo bookkeeping (SPEC DPM-815 §4.2).
"""

import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Tier 2 table → S3 path mapping
TABLE_PATHS = {
    "hist_adj": "historical_prices_adjusted",
    "hist_raw": "historical_prices_raw",
    "bond_clean": "bond_clean_prices",
    "yield_by_date": "yield_bonds/by_date",
    "yield_by_ticker": "yield_bonds/by_ticker",
    "hist_fci": "historical_fcis",
}

# ── P0-B single-flight / memo / semaphore config (env-configurable per
# service — SPEC DPM-815 §4.3; nexus's defaults are the package defaults) ──
SINGLEFLIGHT_ENABLED_ENV = "SMART_LOADER_SINGLEFLIGHT"
MEMO_TTL_S_ENV = "SMART_LOADER_MEMO_TTL_S"
MEMO_MAX_ENTRIES_ENV = "SMART_LOADER_MEMO_MAX_ENTRIES"
MAX_CONCURRENT_MATERIALIZATIONS_ENV = "SMART_LOADER_MAX_CONCURRENT_MATERIALIZATIONS"

MEMO_TTL_S_DEFAULT = 5.0
MEMO_MAX_ENTRIES_DEFAULT = 3
MAX_CONCURRENT_MATERIALIZATIONS_DEFAULT = 2


def _singleflight_enabled() -> bool:
    return os.environ.get(SINGLEFLIGHT_ENABLED_ENV, "on").strip().lower() != "off"


def _memo_ttl_s() -> float:
    return float(os.environ.get(MEMO_TTL_S_ENV, MEMO_TTL_S_DEFAULT))


def _memo_max_entries() -> int:
    return int(os.environ.get(MEMO_MAX_ENTRIES_ENV, MEMO_MAX_ENTRIES_DEFAULT))


def _max_concurrent_materializations() -> int:
    return int(
        os.environ.get(
            MAX_CONCURRENT_MATERIALIZATIONS_ENV, MAX_CONCURRENT_MATERIALIZATIONS_DEFAULT
        )
    )


# Module-level, process-wide state (SPEC §4.2). `_MEMO` maps
# `f"{bucket}/{key}"` -> (materialized_at_monotonic, DataFrame).
# `_MATERIALIZE_SEM` is sized once at import time from env; tests that need
# a different concurrency cap replace it directly (it's a deploy-time knob,
# not a per-request one).
_MEMO_LOCK = threading.Lock()
_KEY_LOCKS: Dict[str, threading.Lock] = {}
_MEMO: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
_MATERIALIZE_SEM = threading.Semaphore(_max_concurrent_materializations())


def _purge_memo_locked(now: float) -> None:
    """Drop every memo entry older than the current TTL, releasing the
    DataFrame reference (SPEC §4.2 step 2 / M3 fix). Must be called with
    `_MEMO_LOCK` held."""
    ttl = _memo_ttl_s()
    stale_keys = [k for k, (ts, _df) in _MEMO.items() if now - ts > ttl]
    for k in stale_keys:
        del _MEMO[k]


class ParquetReader:
    """Reads Parquet files from S3 for Tier 2 historical data."""

    def __init__(self, bucket: str, prefix: str = "v1"):
        self._bucket = bucket
        self._prefix = prefix
        # max_pool_connections=32 to support concurrent ticker fetches from
        # asyncio.to_thread callers (nexus historical_controller uses 16 threads/worker).
        self._s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(max_pool_connections=32),
        )
        # Anti-poisoning (SPEC §3.5): per-thread flag, cleared at the top of
        # each public read_*() call, set whenever a read along the way could
        # not authoritatively determine "the ticker isn't there" (S3 error,
        # unknown table, partition listing failure). The loader consults
        # `consume_degraded()` right after the call to decide whether the
        # resulting empty list is safe to negative-cache.
        self._degraded = threading.local()

    def _clear_degraded(self) -> None:
        self._degraded.value = False

    def _mark_degraded(self) -> None:
        self._degraded.value = True

    def consume_degraded(self) -> bool:
        """Read-and-reset. Only meaningful immediately after a read_*() call
        on the same thread (SPEC §3.5 point 6)."""
        value = getattr(self._degraded, "value", False)
        self._degraded.value = False
        return value

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
        self._clear_degraded()

        s3_path = TABLE_PATHS.get(table)
        if not s3_path:
            logger.error(f"Unknown Tier 2 table: {table}")
            # Not authoritative: an unknown table name says nothing about
            # whether the ticker exists (SPEC §3.4 "table desconocida").
            self._mark_degraded()
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
        self._clear_degraded()

        s3_path = TABLE_PATHS.get(table)
        if not s3_path:
            logger.error(f"Unknown Tier 2 table: {table}")
            self._mark_degraded()
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
        self._clear_degraded()

        s3_path = TABLE_PATHS.get(table)
        if not s3_path:
            logger.error(f"Unknown Tier 2 table: {table}")
            self._mark_degraded()
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
            # Not authoritative: couldn't even enumerate partitions, so
            # absence here says nothing about the ticker (SPEC §3.5 point 3).
            self._mark_degraded()
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
        """Download a Parquet file from S3, single-flighted and memoized
        (DPM-815 P0-B). Falls through to the uncached body when the
        single-flight kill-switch is off, or when memoization can't help
        (miss, or a stale/absent entry)."""
        if not _singleflight_enabled():
            return self._download_parquet_uncached(key)

        memo_key = f"{self._bucket}/{key}"

        with _MEMO_LOCK:
            _purge_memo_locked(time.monotonic())
            entry = _MEMO.get(memo_key)
            if entry is not None:
                _MEMO.move_to_end(memo_key)
                return entry[1]
            key_lock = _KEY_LOCKS.setdefault(memo_key, threading.Lock())

        with key_lock:
            # Double-checked locking: another thread may have materialized
            # (or the entry may have gone stale) while we waited for the lock.
            with _MEMO_LOCK:
                _purge_memo_locked(time.monotonic())
                entry = _MEMO.get(memo_key)
                if entry is not None:
                    _MEMO.move_to_end(memo_key)
                    return entry[1]

            with _MATERIALIZE_SEM:
                df = self._download_parquet_uncached(key)

            if df is not None:
                with _MEMO_LOCK:
                    _MEMO[memo_key] = (time.monotonic(), df)
                    _MEMO.move_to_end(memo_key)
                    max_entries = _memo_max_entries()
                    while len(_MEMO) > max_entries:
                        _MEMO.popitem(last=False)
            # df is None: never memoized — an S3 error/miss must be able to
            # retry on the next call, not stick around as a cached failure.

            return df

    def _download_parquet_uncached(self, key: str) -> Optional["pd.DataFrame"]:
        """Download a Parquet file from S3 and return as DataFrame. This is
        the actual body — unchanged apart from the anti-poisoning marking
        (SPEC §3.5 point 2: ANY None return, NoSuchKey included, is a
        non-authoritative read)."""
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
            self._mark_degraded()
            return None
        except Exception as e:
            logger.error(f"Error downloading s3://{self._bucket}/{key}: {e}")
            self._mark_degraded()
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

            if "submarket" in df.columns:
                for submarket, sub_group in date_group.groupby("submarket"):
                    result[date_str][submarket] = self._df_to_records(sub_group)
            else:
                result[date_str] = self._df_to_records(date_group)

        return result
