"""
SmartLoader — Tiered data loader for Docta services.

Tier 1: Reference data loaded from Redis DB 2 at startup (same as current pattern).
Tier 2: Historical series loaded on-demand from Redis per-ticker cache + S3 Parquet fallback.

Cache freshness: cronos flushes ts:cache:* after writing new Parquet daily at 18:30 ART.
TTL 24h on per-ticker keys is a safety net only.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import redis

from smart_loader.parquet_reader import ParquetReader

logger = logging.getLogger(__name__)

# Redis DB 2 key prefix for Tier 1 bulk data (written by cronos)
TIER1_PREFIX = "ts:"

# Redis DB 2 key prefix for Tier 2 per-ticker cache (written by SmartLoader on miss)
TIER2_CACHE_PREFIX = "ts:cache:"

# TTL for per-ticker cache keys (safety net — cronos flush is the primary freshness mechanism)
TIER2_CACHE_TTL = 86400  # 24 hours

# Tier 1 tables stored as JSON-serialized DataFrames in Redis
TIER1_DATAFRAME_KEYS = [
    "bonds",
    "cer",
    "floating_bands",
    "docta_tickers",
    "species",
    "ratios",
    "historical_usd_prices",
    "tamar",
    "badlar",
    "a3500",
    "tc-minorista",
]

# Tier 1 tables stored as JSON dicts in Redis
# (db_tables_key, redis_key_suffix)
TIER1_DICT_KEYS = [
    ("tickers_issuers", "tickers_issuers"),
    ("tickers_laws", "tickers_laws"),
    ("tickers_sectors", "tickers_sectors"),
    ("fcis", "fcis"),
    ("fcis_by_lbo_code", "fcis_by_lbo_code"),
    ("replaced_fcis_by_lbo_code", "replaced_fcis_by_lbo_code"),
    ("cash_flows_adj", "cash_flows_adj"),
    ("future_contracts_serie", "future_contracts_serie"),
    ("mav_discount_rates", "mav_discount_rates"),
]

# Dict keys where JSON string keys should be converted back to int
INT_KEY_TABLES = {"tickers_issuers", "tickers_laws", "tickers_sectors"}

# Tier 2 table name mapping: SmartLoader name → S3 Parquet path prefix
TIER2_TABLES = {
    "hist_adj": "historical_prices_adjusted",
    "hist_raw": "historical_prices_raw",
    "bond_clean": "bond_clean_prices",
    "yield_by_date": "yield_bonds/by_date",
    "yield_by_ticker": "yield_bonds/by_ticker",
    "hist_fci": "historical_fcis",
}


class SmartLoader:
    """Tiered data loader: Tier 1 (Redis bulk) + Tier 2 (Redis per-ticker cache + S3 Parquet)."""

    def __init__(
        self,
        redis_host: str = None,
        redis_port: int = None,
        redis_db: int = 2,
        s3_bucket: str = None,
        s3_prefix: str = "v1",
    ):
        self._redis_host = redis_host or os.environ.get("REDIS_HOST", "localhost")
        self._redis_port = redis_port or int(os.environ.get("REDIS_PORT", 6379))

        self._redis = redis.Redis(
            host=self._redis_host,
            port=self._redis_port,
            db=redis_db,
            decode_responses=True,
        )

        self._s3_bucket = s3_bucket or os.environ.get("DB_TABLES_S3_BUCKET", "docta-db-tables")
        self._s3_prefix = s3_prefix
        self._parquet_reader = ParquetReader(self._s3_bucket, self._s3_prefix)

        # Tier 1 data loaded at startup
        self._tier1_data: Dict[str, Any] = {}

        logger.info(
            f"SmartLoader initialized: redis={self._redis_host}:{self._redis_port}/db{redis_db}, "
            f"s3={self._s3_bucket}/{self._s3_prefix}"
        )

    # ── Tier 1: reference data (startup, same as current pattern) ──

    def load_tier1(self) -> Dict[str, Any]:
        """
        Load all Tier 1 tables from Redis DB 2. Called once at startup.
        Returns dict compatible with current DB_TABLES interface.
        """
        start_time = datetime.now()
        logger.info("Loading Tier 1 tables from Redis DB 2...")

        self._redis.ping()

        # Load DataFrames
        for key in TIER1_DATAFRAME_KEYS:
            redis_key = f"{TIER1_PREFIX}{key}"
            try:
                data = self._redis.get(redis_key)
                if data:
                    self._tier1_data[key] = _deserialize_dataframe(data)
                    logger.info(f"  Tier1 [{key}]: {len(self._tier1_data[key])} rows")
                else:
                    logger.warning(f"  Tier1 [{key}]: missing in Redis")
                    self._tier1_data[key] = pd.DataFrame()
            except Exception as e:
                logger.error(f"  Tier1 [{key}]: error - {e}")
                self._tier1_data[key] = pd.DataFrame()

        # Load dicts
        for db_key, redis_suffix in TIER1_DICT_KEYS:
            redis_key = f"{TIER1_PREFIX}{redis_suffix}"
            try:
                data = self._redis.get(redis_key)
                if data:
                    parsed = json.loads(data)
                    if db_key in INT_KEY_TABLES:
                        parsed = {int(k): v for k, v in parsed.items() if k.isdigit()}
                    self._tier1_data[db_key] = parsed
                    size = len(parsed) if isinstance(parsed, dict) else "loaded"
                    logger.info(f"  Tier1 [{db_key}]: {size} items")
                else:
                    logger.warning(f"  Tier1 [{db_key}]: missing in Redis")
                    self._tier1_data[db_key] = {}
            except Exception as e:
                logger.error(f"  Tier1 [{db_key}]: error - {e}")
                self._tier1_data[db_key] = {}

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Tier 1 loaded: {len(self._tier1_data)} tables in {elapsed:.2f}s")

        return self._tier1_data

    def get_table(self, key: str) -> Any:
        """Get a Tier 1 table. O(1) dict lookup."""
        return self._tier1_data.get(key)

    @property
    def tier1_data(self) -> Dict[str, Any]:
        """Direct access to the Tier 1 dict. For backward-compatible DB_TABLES access."""
        return self._tier1_data

    # ── Tier 2: historical series (on-demand, per-ticker cache) ──

    def get_ticker_series(self, table: str, ticker: str) -> List[Dict]:
        """
        Get historical series for a single ticker.

        Storage: Redis Hash where each field is a date and value is the JSON record.
        Backward compat: auto-migrates old STRING format to HASH on read.

        Args:
            table: Tier 2 table name (e.g., "hist_adj", "bond_clean", "yield_by_ticker")
            ticker: Ticker symbol (e.g., "GGAL", "AL30")

        Returns:
            List of dicts representing the ticker's time series records.
        """
        cache_key = f"{TIER2_CACHE_PREFIX}{table}:{ticker}"

        # 1. Check key type for backward compat
        key_type = self._redis.type(cache_key)

        if key_type == "hash":
            hash_data = self._redis.hgetall(cache_key)
            if hash_data:
                return [json.loads(v) for v in hash_data.values()]

        elif key_type == "string":
            # Old format — read, migrate to hash
            cached = self._redis.get(cache_key)
            if cached is not None:
                data = json.loads(cached)
                self._redis.delete(cache_key)
                self._cache_as_hash(cache_key, data)
                return data

        # 2. Cache miss — load from S3 Parquet, cache as hash
        data = self._parquet_reader.read_ticker(table, ticker)
        if data:
            self._cache_as_hash(cache_key, data)

        return data

    def get_prices_for_dates(
        self, table: str, ticker_date_pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Batch fetch specific dates for specific tickers via pipelined HMGET.

        Much faster than get_ticker_series when only a few dates are needed
        per ticker (e.g., 20 dates out of 2500 records).

        Falls back to S3 for tickers not yet cached.

        Args:
            table: Tier 2 table name (e.g., "hist_adj")
            ticker_date_pairs: List of (ticker, date_iso_str) tuples

        Returns:
            Dict mapping (ticker, date_str) → record dict
        """
        if not ticker_date_pairs:
            return {}

        # Group by ticker
        by_ticker: Dict[str, List[str]] = {}
        for ticker, date_str in ticker_date_pairs:
            by_ticker.setdefault(ticker, []).append(date_str)

        # Pipeline HMGET — 1 round-trip for all tickers
        pipe = self._redis.pipeline()
        ticker_order = []
        for ticker, dates in by_ticker.items():
            cache_key = f"{TIER2_CACHE_PREFIX}{table}:{ticker}"
            pipe.hmget(cache_key, *dates)
            ticker_order.append((ticker, dates))
        results = pipe.execute()

        # Parse hits, identify tickers with complete miss (hash not populated)
        output: Dict[Tuple[str, str], Dict] = {}
        tickers_to_load = []
        for (ticker, dates), values in zip(ticker_order, results):
            if all(v is None for v in values):
                tickers_to_load.append(ticker)
            else:
                for date_str, val in zip(dates, values):
                    if val is not None:
                        output[(ticker, date_str)] = json.loads(val)

        # Load missing tickers from S3 → populate hash → retry HMGET
        if tickers_to_load:
            for ticker in tickers_to_load:
                self.get_ticker_series(table, ticker)

            pipe = self._redis.pipeline()
            retry_order = []
            for ticker in tickers_to_load:
                dates = by_ticker[ticker]
                pipe.hmget(f"{TIER2_CACHE_PREFIX}{table}:{ticker}", *dates)
                retry_order.append((ticker, dates))
            retry_results = pipe.execute()

            for (ticker, dates), values in zip(retry_order, retry_results):
                for date_str, val in zip(dates, values):
                    if val is not None:
                        output[(ticker, date_str)] = json.loads(val)

        return output

    def _cache_as_hash(self, cache_key: str, data: List[Dict]) -> None:
        """Store a list of records as a Redis Hash (field=date, value=JSON record)."""
        if not data:
            return
        pipe = self._redis.pipeline()
        for record in data:
            date_str = str(record.get("date", "unknown"))
            pipe.hset(cache_key, date_str, json.dumps(record, default=str))
        pipe.expire(cache_key, TIER2_CACHE_TTL)
        pipe.execute()

    def get_market_series(self, table: str, market: str) -> Dict[str, List[Dict]]:
        """
        Get all tickers for a market type.

        Used by endpoints that scan across a full market (e.g., yield curves, market indices).
        Downloads the market partition from S3 Parquet and caches each ticker individually.

        Args:
            table: Tier 2 table name (e.g., "hist_adj", "yield_by_date")
            market: Market type (e.g., "stock", "cedear", "bond")

        Returns:
            Dict mapping ticker → list of records (same structure as current DB_TABLES).
        """
        # For yield_by_date, the structure is {date_str: {submarket: [records]}} — no market partition
        if table == "yield_by_date":
            cache_key = f"{TIER2_CACHE_PREFIX}{table}:__all__"
            cached = self._redis.get(cache_key)
            if cached is not None:
                return json.loads(cached)
            data = self._parquet_reader.read_full_table(table)
            if data:
                self._redis.setex(cache_key, TIER2_CACHE_TTL, json.dumps(data, default=str))
            return data or {}

        # For market-partitioned tables, load the partition and cache per-ticker as hash
        all_tickers = self._parquet_reader.read_market_partition(table, market)

        if all_tickers:
            for ticker, records in all_tickers.items():
                cache_key = f"{TIER2_CACHE_PREFIX}{table}:{ticker}"
                self._cache_as_hash(cache_key, records)

        return all_tickers or {}

    # ── Special operations ──

    def save_cash_flows_adj(self, cash_flows_adj: Dict[str, Any]) -> bool:
        """Save calculated cash_flows_adj back to Redis (called by nexus on first startup)."""
        try:
            redis_key = f"{TIER1_PREFIX}cash_flows_adj"
            self._redis.set(redis_key, json.dumps(cash_flows_adj, default=str))
            self._tier1_data["cash_flows_adj"] = cash_flows_adj
            logger.info(f"Saved cash_flows_adj to Redis: {len(cash_flows_adj)} bonds")
            return True
        except Exception as e:
            logger.error(f"Error saving cash_flows_adj: {e}")
            return False

    def flush_tier2_cache(self) -> int:
        """
        Flush all Tier 2 per-ticker cache keys.
        Called by cronos after writing new Parquet files.

        Returns:
            Number of keys deleted.
        """
        pattern = f"{TIER2_CACHE_PREFIX}*"
        count = 0
        for key in self._redis.scan_iter(match=pattern, count=1000):
            self._redis.delete(key)
            count += 1
        logger.info(f"Flushed {count} Tier 2 cache keys")
        return count


def _deserialize_dataframe(json_str: str) -> pd.DataFrame:
    """Deserialize JSON string to DataFrame with proper date conversion."""
    try:
        data = json.loads(json_str)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        for col in df.columns:
            if col in ("date", "payment_date") or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col]).dt.date
                except Exception:
                    pass

        return df
    except Exception as e:
        logger.error(f"Error deserializing DataFrame: {e}")
        return pd.DataFrame()
