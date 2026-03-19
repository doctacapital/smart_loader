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
from typing import Any, Dict, List, Optional

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

        1. Check Redis per-ticker cache: ts:cache:{table}:{ticker}
        2. HIT → return (sub-ms)
        3. MISS → load from S3 Parquet → cache in Redis (TTL 24h) → return

        Args:
            table: Tier 2 table name (e.g., "hist_adj", "bond_clean", "yield_by_ticker")
            ticker: Ticker symbol (e.g., "GGAL", "AL30")

        Returns:
            List of dicts representing the ticker's time series records.
        """
        cache_key = f"{TIER2_CACHE_PREFIX}{table}:{ticker}"

        # 1. Check Redis cache
        cached = self._redis.get(cache_key)
        if cached is not None:
            return json.loads(cached)

        # 2. Cache miss — load from S3 Parquet
        data = self._parquet_reader.read_ticker(table, ticker)

        # 3. Cache in Redis with TTL
        if data:
            self._redis.setex(cache_key, TIER2_CACHE_TTL, json.dumps(data, default=str))

        return data

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

        # For market-partitioned tables, load the partition and cache per-ticker
        all_tickers = self._parquet_reader.read_market_partition(table, market)

        # Cache each ticker individually
        if all_tickers:
            pipe = self._redis.pipeline()
            for ticker, records in all_tickers.items():
                cache_key = f"{TIER2_CACHE_PREFIX}{table}:{ticker}"
                pipe.setex(cache_key, TIER2_CACHE_TTL, json.dumps(records, default=str))
            pipe.execute()

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
