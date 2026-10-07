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
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import redis

from smart_loader.parquet_reader import ParquetReader
from smart_loader.pg_reader import PostgresReader

logger = logging.getLogger(__name__)

# Redis DB 2 key prefix for Tier 1 bulk data (written by cronos)
TIER1_PREFIX = "ts:"

# Redis DB 2 key prefix for Tier 2 per-ticker cache (written by SmartLoader on miss)
TIER2_CACHE_PREFIX = "ts:cache:"

# TTL for per-ticker cache keys (safety net — cronos flush is the primary freshness mechanism)
TIER2_CACHE_TTL = 86400  # 24 hours

# Negative-cache (DPM-815 P0-A): key hermana del positivo, marca "se buscó y
# no está" para no volver a materializar el Parquet completo en cada miss.
# Sufijo, no campo de HASH ni STRING separada: convive sin romper lectores
# v0.1.6 durante el bump escalonado (SPEC DPM-815 §3.1).
TIER2_NEG_SUFFIX = ":__miss__"

# TTL del marker negativo. Default 3600s (decisión de Tomás, revisable por env
# sin rebuild — SPEC DPM-815 §0.1/§3.1).
TIER2_NEG_CACHE_TTL_ENV = "SMART_LOADER_NEG_CACHE_TTL"
TIER2_NEG_CACHE_TTL_DEFAULT = 3600

# Kill-switch del negative-cache. "on"/"off", default "on".
NEG_CACHE_ENABLED_ENV = "SMART_LOADER_NEG_CACHE"

# Tier 1 tables stored as JSON-serialized DataFrames in Redis
TIER1_DATAFRAME_KEYS = [
    "bonds",
    "cer",
    "uva",
    "floating_bands",
    "docta_tickers",
    "species",
    "ratios",
    "historical_usd_prices",
    "tamar",
    "badlar",
    "a3500",
    "tc-minorista",
    "eur_prices",
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

# ── yield_by_date por fecha (DPM-1388) ──
# La curva histórica se guarda como un HASH con un campo por fecha, armado por
# un solo proceso a la vez (candado compartido en Redis) y publicado completo
# o nada. Reemplaza al STRING `ts:cache:yield_by_date:__all__` (~215 MB, una
# escritura por worker en paralelo el 6-10 → desalojo de tablas fijas).
#
# Las dos claves `ts:cache:yield_by_date:*` caen en el borrado diario de
# cronos fase B. El candado vive fuera de `ts:` a propósito: ni el borrado de
# cronos ni la captura local de `ts:*` lo tocan.
YIELD_BY_DATE_TABLE = "yield_by_date"
YIELD_BY_DATE_KEY = f"{TIER2_CACHE_PREFIX}{YIELD_BY_DATE_TABLE}:__by_date__"
YIELD_BY_DATE_BUILDING_PREFIX = f"{TIER2_CACHE_PREFIX}{YIELD_BY_DATE_TABLE}:__building__:"
YIELD_BY_DATE_LOCK_KEY = "lock:smart_loader:yield_by_date"
YIELD_BY_DATE_META_FIELD = "__meta__"

YIELD_BY_DATE_TTL = TIER2_CACHE_TTL          # hash publicado: 86400 s
YIELD_BY_DATE_BUILDING_TTL = 300             # temporal: renovado en cada tanda
YIELD_BY_DATE_LOCK_TTL = 120                 # ~7 veces el armado medido (16,6 s)
YIELD_BY_DATE_BATCH_DATES = 50               # fechas por tanda de HSET
YIELD_BY_DATE_WAIT_S = 25.0                  # docto corta a 30 s
YIELD_BY_DATE_POLL_S = 0.5

# Publica la temporal solo si está completa (N fechas + __meta__); si no, la
# borra. RENAME + EXPIRE en el mismo script: nadie ve la final sin vencimiento.
# Los dos scripts van con EVAL (no EVALSHA): corren una vez por armado, y así
# no dependen del cache de scripts del servidor (se vacía en un failover).
_PUBLISH_YIELD_BY_DATE_LUA = """
if redis.call('HLEN', KEYS[1]) == tonumber(ARGV[1]) then
    redis.call('RENAME', KEYS[1], KEYS[2])
    redis.call('EXPIRE', KEYS[2], tonumber(ARGV[2]))
    return 1
end
redis.call('DEL', KEYS[1])
return 0
"""

# Suelta el candado solo si sigue siendo nuestro (compare-and-delete).
_RELEASE_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class YieldCacheBuilding(Exception):
    """Otro proceso está armando `yield_by_date` y no terminó dentro de la
    espera máxima (YIELD_BY_DATE_WAIT_S). nexus lo traduce a 503 con
    Retry-After."""


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
        # Fallback backend always built, regardless of SMART_LOADER_BACKEND — pg
        # only ever covers the tables listed in SMART_LOADER_PG_TABLES (DPM-395 §1).
        self._parquet_reader = ParquetReader(self._s3_bucket, self._s3_prefix)

        # Negative-cache (DPM-815 P0-A): kill-switch + TTL, both env-configurable
        # so the TTL can be raised without a package rebuild (SPEC §0.1).
        self._neg_cache_enabled = (
            os.environ.get(NEG_CACHE_ENABLED_ENV, "on").strip().lower() != "off"
        )
        self._neg_cache_ttl = int(
            os.environ.get(TIER2_NEG_CACHE_TTL_ENV, TIER2_NEG_CACHE_TTL_DEFAULT)
        )

        # Tier 1 data loaded at startup
        self._tier1_data: Dict[str, Any] = {}

        self._backend = os.environ.get("SMART_LOADER_BACKEND", "s3").lower()
        if self._backend not in ("s3", "pg"):
            logger.warning(
                f"Unknown SMART_LOADER_BACKEND={self._backend!r}, falling back to s3"
            )
            self._backend = "s3"

        pg_tables_raw = os.environ.get("SMART_LOADER_PG_TABLES", "hist_adj,hist_raw")
        self._pg_tables = {t.strip() for t in pg_tables_raw.split(",") if t.strip()}

        self._pg_reader: Optional[PostgresReader] = None
        if self._backend == "pg":
            dsn = os.environ.get("SMART_LOADER_PG_DSN")
            if not dsn:
                raise ValueError(
                    "SMART_LOADER_BACKEND=pg requires SMART_LOADER_PG_DSN (fail-fast at startup)"
                )
            pool_min = int(os.environ.get("SMART_LOADER_PG_POOL_MIN", 1))
            pool_max = int(os.environ.get("SMART_LOADER_PG_POOL_MAX", 8))
            self._pg_reader = PostgresReader(
                dsn,
                pool_min=pool_min,
                pool_max=pool_max,
                metadata_provider=self._tier1_ticker_meta,
            )

        logger.info(
            f"SmartLoader initialized: redis={self._redis_host}:{self._redis_port}/db{redis_db}, "
            f"s3={self._s3_bucket}/{self._s3_prefix}, backend={self._backend}, "
            f"pg_tables={sorted(self._pg_tables)}"
        )

    def _reader_for(self, table: str):
        """Tier 2 backend selector (§1): pg only for tables in
        SMART_LOADER_PG_TABLES, s3/Parquet otherwise (default fallback)."""
        if self._pg_reader is not None and table in self._pg_tables:
            return self._pg_reader
        return self._parquet_reader

    def _tier1_ticker_meta(self, ticker: str) -> Optional[Dict[str, Any]]:
        """metadata_provider for PostgresReader (§5.3): enriches dwh bar rows
        with the legacy per-ticker fields dwh doesn't carry on eod_bar.

        NOTE on the §5.3 halt (resolved by planner): `currency` is NOT part
        of this provider's contract anymore — PostgresReader sources it from
        dwh.series.trade_currency (per-serie) and injects it directly into
        the record, bypassing metadata_provider entirely. That's because the
        Tier1 `docta_tickers` DataFrame (ts:docta_tickers, Supabase `tickers`
        table) only has columns `tickers`/`market_type`/`submarket_type`/
        `name`/`sector`/*_id/`isin_code` — confirmed via nexus/utils.py and
        cronos/services/redis_timeseries_service.py:540-542 — with no
        per-ticker currency column at all.

        `specie`/`current_closing_price` stay None permanently (planner
        decision: their only historical source is real-time L2 quote
        payloads — cronos/utils/data_helpers.py:49-65 — which SmartLoader's
        Tier 1 never loads, and nothing downstream reads them materially).
        `submarket` is populated from the real `submarket_type` column;
        `settlement_period` uses the spec's explicit fallback constant
        "24hs" (§5.3), ratified by the planner."""
        tickers_df = self._tier1_data.get("docta_tickers")
        if tickers_df is None or getattr(tickers_df, "empty", True):
            return None
        if "tickers" not in tickers_df.columns:
            return None

        match = tickers_df.loc[tickers_df["tickers"] == ticker]
        if match.empty:
            return None

        row = match.iloc[0]
        return {
            "specie": None,
            "submarket": row.get("submarket_type"),
            "current_closing_price": None,
            "settlement_period": "24hs",
        }

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
        neg_key = cache_key + TIER2_NEG_SUFFIX

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

        # 2. Negative-cache (DPM-815 P0-A): a marker here means an earlier,
        # authoritative read already found nothing for this ticker. The
        # positive branches above always run first (I-1), so this never
        # shadows a real value written after the marker.
        if self._neg_cache_enabled and self._redis.exists(neg_key):
            return []

        # 3. Cache miss — load from Tier 2 backend (pg or S3 Parquet), cache as hash
        reader = self._reader_for(table)
        data = reader.read_ticker(table, ticker)
        if data:
            self._cache_as_hash(cache_key, data)
            if self._neg_cache_enabled:
                self._redis.delete(neg_key)
        elif self._neg_cache_enabled and _may_cache_negative(reader):
            # Anti-poisoning (§3.5): only mark "doesn't exist" when the read
            # was authoritative — never on an S3 error/timeout swallowed by
            # the reader, which would otherwise cache a false negative for
            # the full TTL.
            try:
                self._redis.setex(neg_key, self._neg_cache_ttl, "1")
            except Exception as e:
                logger.warning(f"Failed to set negative-cache marker {neg_key}: {e}")

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

        # Migrate any old STRING keys to HASH before HMGET
        type_pipe = self._redis.pipeline(transaction=False)
        tickers_list = list(by_ticker.keys())
        for ticker in tickers_list:
            type_pipe.type(f"{TIER2_CACHE_PREFIX}{table}:{ticker}")
        types = type_pipe.execute()

        for ticker, key_type in zip(tickers_list, types):
            if key_type == "string":
                self.get_ticker_series(table, ticker)  # auto-migrates to hash

        # Pipeline HMGET — 1 round-trip for all tickers
        pipe = self._redis.pipeline(transaction=False)
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

        VIEJO para yield_by_date (DPM-1388): usar `get_yield_for_date`. La rama
        `yield_by_date` de acá (STRING `ts:cache:yield_by_date:__all__`, ~215 MB,
        sin candado entre procesos) queda solo por compatibilidad y se borra en
        una versión posterior.
        """
        # For yield_by_date, the structure is {date_str: {submarket: [records]}} — no market partition
        if table == "yield_by_date":
            cache_key = f"{TIER2_CACHE_PREFIX}{table}:__all__"
            cached = self._redis.get(cache_key)
            if cached is not None:
                return json.loads(cached)
            data = self._reader_for(table).read_full_table(table)
            if data:
                self._redis.setex(cache_key, TIER2_CACHE_TTL, json.dumps(data, default=str))
            return data or {}

        # For market-partitioned tables, load the partition and cache per-ticker as hash
        all_tickers = self._reader_for(table).read_market_partition(table, market)

        if all_tickers:
            for ticker, records in all_tickers.items():
                cache_key = f"{TIER2_CACHE_PREFIX}{table}:{ticker}"
                self._cache_as_hash(cache_key, records)

        return all_tickers or {}

    # ── yield_by_date por fecha, con candado entre procesos (DPM-1388) ──

    def get_yield_for_date(self, date_str: str) -> Optional[Dict[str, List[Dict]]]:
        """Curva histórica de una fecha: `{submarket: [registros]}`, o None si
        la tabla no tiene esa fecha.

        BLOQUEA: puede hacer I/O de Redis y S3 y esperar hasta
        YIELD_BY_DATE_WAIT_S segundos a que otro proceso termine de armar la
        tabla. Desde código async hay que llamarlo fuera del bucle de eventos
        (`run_in_threadpool` / `asyncio.to_thread`).

        1. `HMGET __by_date__ <fecha> __meta__` en un solo comando:
           - `__meta__` y la fecha presentes → devuelve la fecha;
           - `__meta__` presente y la fecha no → None, sin rearmar;
           - `__meta__` ausente → la tabla no está publicada: se arma.
        2. Arma un solo proceso, el que gana `SET lock NX EX 120`. Los demás
           esperan (ver `_wait_for_yield_by_date`).

        Raises:
            YieldCacheBuilding: otro proceso tiene el candado y la tabla no
                apareció dentro de la espera máxima.
        """
        published, value = self._hmget_yield_by_date(date_str)
        if published:
            return value

        token = uuid.uuid4().hex
        if self._try_acquire_yield_lock(token):
            return self._build_yield_by_date(date_str, token)
        return self._wait_for_yield_by_date(date_str)

    def _hmget_yield_by_date(self, date_str: str) -> Tuple[bool, Optional[Dict]]:
        """(publicada, valor). Un solo HMGET: sin carrera con el RENAME de otro
        proceso entre "¿está la fecha?" y "¿está la tabla?"."""
        raw_date, raw_meta = self._redis.hmget(
            YIELD_BY_DATE_KEY, date_str, YIELD_BY_DATE_META_FIELD
        )
        if raw_meta is None:
            return False, None
        if raw_date is None:
            return True, None
        return True, json.loads(raw_date)

    def _try_acquire_yield_lock(self, token: str) -> bool:
        return bool(
            self._redis.set(
                YIELD_BY_DATE_LOCK_KEY, token, nx=True, ex=YIELD_BY_DATE_LOCK_TTL
            )
        )

    def _release_yield_lock(self, token: str) -> None:
        """Borra el candado solo si sigue teniendo nuestro token: si venció y
        lo tomó otro proceso, no se lo sacamos."""
        try:
            self._redis.eval(_RELEASE_LOCK_LUA, 1, YIELD_BY_DATE_LOCK_KEY, token)
        except redis.RedisError as e:
            # Vence solo a los YIELD_BY_DATE_LOCK_TTL segundos.
            logger.warning(f"yield_by_date lock_release_failed pid={os.getpid()}: {e}")

    def _build_yield_by_date(self, date_str: str, token: str) -> Optional[Dict]:
        """Con el candado tomado: vuelve a mirar, lee la tabla entera, la
        publica y suelta el candado. Contesta el pedido desde memoria."""
        try:
            # Doble chequeo: otro proceso pudo publicar entre nuestro HMGET y
            # el SET NX (soltó el candado justo antes de que lo tomáramos).
            published, value = self._hmget_yield_by_date(date_str)
            if published:
                return value

            t0 = time.monotonic()
            data = self._reader_for(YIELD_BY_DATE_TABLE).read_full_table(YIELD_BY_DATE_TABLE)
            if not data:
                logger.warning(
                    f"yield_by_date build_empty pid={os.getpid()}: "
                    f"read_full_table vacío, no se escribe nada"
                )
                return None

            published_ok = self._publish_yield_by_date(data, token)
            logger.info(
                f"yield_by_date build published={int(published_ok)} dates={len(data)} "
                f"dur_ms={int((time.monotonic() - t0) * 1000)} pid={os.getpid()}"
            )
            return data.get(date_str)
        finally:
            self._release_yield_lock(token)

    def _publish_yield_by_date(self, data: Dict[str, Any], token: str) -> bool:
        """Escribe la temporal en tandas y la publica con Lua solo si quedó
        completa. False si no se publicó (temporal borrada o desalojada a
        mitad de camino, o Redis rechazó la escritura): el que armó contesta
        igual desde memoria."""
        building_key = f"{YIELD_BY_DATE_BUILDING_PREFIX}{token}"
        dates = list(data.keys())
        meta = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "dates": len(dates),
            "rows": sum(_count_records(v) for v in data.values()),
            "max_date": max(dates),
        }
        try:
            for start in range(0, len(dates), YIELD_BY_DATE_BATCH_DATES):
                chunk = dates[start:start + YIELD_BY_DATE_BATCH_DATES]
                mapping = {d: json.dumps(data[d], default=str) for d in chunk}
                if start + YIELD_BY_DATE_BATCH_DATES >= len(dates):
                    mapping[YIELD_BY_DATE_META_FIELD] = json.dumps(meta)
                self._write_yield_building_batch(building_key, mapping)

            published = self._redis.eval(
                _PUBLISH_YIELD_BY_DATE_LUA, 2, building_key, YIELD_BY_DATE_KEY,
                len(dates) + 1, YIELD_BY_DATE_TTL,
            )
        except redis.RedisError as e:
            logger.warning(
                f"yield_by_date publish_failed pid={os.getpid()}: {e} "
                f"(se contesta desde memoria)"
            )
            try:
                self._redis.delete(building_key)
            except redis.RedisError:
                pass  # vence sola a los YIELD_BY_DATE_BUILDING_TTL segundos
            return False

        if int(published) != 1:
            logger.warning(
                f"yield_by_date publish_incomplete pid={os.getpid()}: la temporal "
                f"desapareció a mitad del armado, no se publica (se contesta desde memoria)"
            )
            return False
        return True

    def _write_yield_building_batch(self, building_key: str, mapping: Dict[str, str]) -> None:
        """Una tanda: HSET + EXPIRE en MULTI/EXEC. Si la temporal se borró
        entre tandas, el HSET la recrea pero nunca queda sin vencimiento."""
        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(building_key, mapping=mapping)
        pipe.expire(building_key, YIELD_BY_DATE_BUILDING_TTL)
        pipe.execute()

    def _wait_for_yield_by_date(self, date_str: str) -> Optional[Dict]:
        """Perdió el candado: mira cada YIELD_BY_DATE_POLL_S con el mismo HMGET
        hasta YIELD_BY_DATE_WAIT_S. Si el candado desaparece sin que la tabla
        aparezca (el que armaba falló o murió), intenta tomarlo una sola vez."""
        deadline = time.monotonic() + YIELD_BY_DATE_WAIT_S
        retried = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"yield_by_date wait_timeout wait_s={YIELD_BY_DATE_WAIT_S} pid={os.getpid()}"
                )
                raise YieldCacheBuilding(
                    f"yield_by_date is being built by another process "
                    f"(waited {YIELD_BY_DATE_WAIT_S:g}s)"
                )
            time.sleep(min(YIELD_BY_DATE_POLL_S, remaining))

            published, value = self._hmget_yield_by_date(date_str)
            if published:
                return value

            if not retried and not self._redis.exists(YIELD_BY_DATE_LOCK_KEY):
                retried = True
                token = uuid.uuid4().hex
                if self._try_acquire_yield_lock(token):
                    logger.info(f"yield_by_date lock_taken_after_wait pid={os.getpid()}")
                    return self._build_yield_by_date(date_str, token)

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


def _may_cache_negative(reader) -> bool:
    """Anti-poisoning gate (SPEC DPM-815 §3.5): only readers that expose
    `consume_degraded()` and report a clean (non-degraded) read are eligible
    for a negative-cache write. Readers without the hook (e.g. PostgresReader)
    are conservatively excluded — they never cache negatives."""
    consume = getattr(reader, "consume_degraded", None)
    return False if consume is None else not consume()


def _count_records(date_value: Any) -> int:
    """Registros de una fecha de yield_by_date: `{submarket: [registros]}`,
    o una lista plana si el parquet no trae `submarket`."""
    if isinstance(date_value, dict):
        return sum(len(v) for v in date_value.values() if isinstance(v, list))
    if isinstance(date_value, list):
        return len(date_value)
    return 0


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
