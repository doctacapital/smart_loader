"""
Postgres (dwh) backend for SmartLoader Tier 2 — mirrors ParquetReader's
interface so SmartLoader can route per-table without changing callers
(DPM-395 §1).

psycopg/psycopg_pool are imported lazily inside __init__ so the S3/Parquet
backend never needs them installed (DPM-395 §1).
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from smart_loader.legacy_shim import (
    build_bond_clean_record,
    build_fci_record,
    build_hist_adj_record,
    build_hist_raw_record,
    build_yield_record,
    group_records_by_ticker,
)

logger = logging.getLogger(__name__)

# Tier 2 table -> instrument.kind values eligible for resolution (DPM-395 §2).
# yield_* verified against dwh.instrument's kind CHECK constraint
# (docta-dwh/db/migrations/20260705000003_instruments.sql:4-7).
TABLE_KINDS = {
    "hist_raw": ("STOCK", "CEDEAR"),
    "hist_adj": ("STOCK", "CEDEAR"),
    "bond_clean": ("BOND", "LETTER", "ON", "ON_PYME"),
    "yield_by_ticker": ("BOND", "LETTER", "ON", "ON_PYME"),
    "yield_by_date": ("BOND", "LETTER", "ON", "ON_PYME"),
    "hist_fci": ("FCI",),
}

# Tier 2 table -> venue.mic used for series resolution. hist_fci is the only
# table on a non-BYMA venue (CAFCI) — the rest resolve on XBUE (DPM-395 §2
# hardcodes XBUE for the generic case; hist_fci is BLOQUEADA/not activated,
# but coded here for shape parity — see coder report on this deviation).
TABLE_MIC = {
    "hist_raw": "XBUE",
    "hist_adj": "XBUE",
    "bond_clean": "XBUE",
    "yield_by_ticker": "XBUE",
    "yield_by_date": "XBUE",
    "hist_fci": "CAFCI",
}

# market (SmartLoader/nexus vocabulary) -> instrument.kind, for
# read_market_partition (DPM-395 §4.6).
MARKET_KIND = {
    "stock": "STOCK",
    "cedear": "CEDEAR",
}

_RESOLVE_SQL = """
WITH direct AS (
    SELECT s.series_id, 1 AS pref
    FROM dwh.series s
    JOIN dwh.instrument i ON i.instrument_id = s.instrument_id
    WHERE s.symbol = %(ticker)s
      AND s.mic = %(mic)s
      AND s.settlement = '24H'
      AND i.kind = ANY(%(kinds)s)
), via_alias AS (
    SELECT s.series_id, 2 AS pref
    FROM dwh.series_alias a
    JOIN dwh.series s ON s.series_id = a.series_id
    JOIN dwh.instrument i ON i.instrument_id = s.instrument_id
    WHERE a.symbol = %(ticker)s
      AND s.mic = %(mic)s
      AND s.settlement = '24H'
      AND i.kind = ANY(%(kinds)s)
)
SELECT series_id FROM (
    SELECT * FROM direct UNION ALL SELECT * FROM via_alias
) u
ORDER BY pref
LIMIT 1
"""

_BAR_SQL = """
SELECT trade_date::text AS date, open, high, low, close, volume, turnover
FROM dwh.eod_bar
WHERE series_id = %(sid)s
ORDER BY trade_date
"""

_BAR_MEP_SQL = """
SELECT b.trade_date::text AS date, b.open, b.high, b.low, b.close, b.volume, b.turnover,
       fx.value AS mep_rate
FROM dwh.eod_bar b
LEFT JOIN dwh.fx_rate fx
       ON fx.rate_date = b.trade_date
      AND fx.base_currency = 'USD'
      AND fx.quote_currency = 'ARS'
      AND fx.rate_type = 'MEP'
WHERE b.series_id = %(sid)s
ORDER BY b.trade_date
"""

_BAR_MEP_PARTITION_SQL = """
SELECT s.symbol AS ticker, b.trade_date::text AS date,
       b.open, b.high, b.low, b.close, b.volume, b.turnover,
       fx.value AS mep_rate
FROM dwh.eod_bar b
JOIN dwh.series s ON s.series_id = b.series_id
JOIN dwh.instrument i ON i.instrument_id = s.instrument_id
LEFT JOIN dwh.fx_rate fx
       ON fx.rate_date = b.trade_date
      AND fx.base_currency = 'USD'
      AND fx.quote_currency = 'ARS'
      AND fx.rate_type = 'MEP'
WHERE s.mic = %(mic)s
  AND s.settlement = '24H'
  AND i.kind = %(kind)s
ORDER BY s.symbol, b.trade_date
"""

_BAR_PARTITION_SQL = """
SELECT s.symbol AS ticker, b.trade_date::text AS date,
       b.open, b.high, b.low, b.close, b.volume, b.turnover
FROM dwh.eod_bar b
JOIN dwh.series s ON s.series_id = b.series_id
JOIN dwh.instrument i ON i.instrument_id = s.instrument_id
WHERE s.mic = %(mic)s
  AND s.settlement = '24H'
  AND i.kind = %(kind)s
ORDER BY s.symbol, b.trade_date
"""

_BOND_CLEAN_SQL = """
SELECT m.trade_date::text AS date, m.clean_price, m.dirty_price,
       mep.value AS mep_rate, ccl.value AS ccl_rate
FROM dwh.eod_metric m
LEFT JOIN dwh.fx_rate mep
       ON mep.rate_date = m.trade_date
      AND mep.base_currency = 'USD' AND mep.quote_currency = 'ARS' AND mep.rate_type = 'MEP'
LEFT JOIN dwh.fx_rate ccl
       ON ccl.rate_date = m.trade_date
      AND ccl.base_currency = 'USD' AND ccl.quote_currency = 'ARS' AND ccl.rate_type = 'CCL'
WHERE m.series_id = %(sid)s
ORDER BY m.trade_date
"""

_YIELD_SQL = """
SELECT b.trade_date::text AS date, b.open, b.high, b.low, b.close, b.volume, b.turnover,
       m.tir, m.tem, m.tea, m.tna, m.duration, m.accrued_interest, m.parity, m.dtm,
       m.residual_value, m.clean_price, m.dirty_price, m.extra
FROM dwh.eod_bar b
JOIN dwh.eod_metric m ON m.series_id = b.series_id AND m.trade_date = b.trade_date
WHERE b.series_id = %(sid)s
ORDER BY b.trade_date
"""

_FCI_SQL = """
SELECT trade_date::text AS date, close AS vcp
FROM dwh.eod_bar
WHERE series_id = %(sid)s
ORDER BY trade_date
"""


class PostgresReader:
    """Reads Tier 2 historical data from the dwh Postgres schema.

    Interface mirrors ParquetReader so SmartLoader can route per-table
    without touching call sites (DPM-395 §1).
    """

    def __init__(
        self,
        dsn: str,
        *,
        pool_min: int = 1,
        pool_max: int = 8,
        metadata_provider: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ):
        # Lazy import: the S3/Parquet backend must not require psycopg installed.
        import psycopg  # noqa: F401
        from psycopg.rows import dict_row

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._metadata_provider = metadata_provider
        self._pool = None  # opened lazily on first query (§1)

        # In-process series_id resolution cache. Positive hits only — never
        # cache misses, so a symbol that resolves later (e.g. after a
        # promoter run) is picked up without a restart (§2).
        self._series_id_cache: Dict[tuple, Optional[int]] = {}

    # ── pool / connection plumbing ──

    def _get_pool(self):
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                kwargs={"row_factory": self._dict_row},
                open=True,
            )
        return self._pool

    def _query(self, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        pool = self._get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # ── series resolution (§2) ──

    def _resolve_series_id(self, table: str, ticker: str) -> Optional[int]:
        cache_key = (table, ticker)
        if cache_key in self._series_id_cache:
            return self._series_id_cache[cache_key]

        kinds = list(TABLE_KINDS.get(table, ()))
        mic = TABLE_MIC.get(table, "XBUE")
        rows = self._query(_RESOLVE_SQL, {"ticker": ticker, "mic": mic, "kinds": kinds})
        series_id = rows[0]["series_id"] if rows else None

        if series_id is not None:
            self._series_id_cache[cache_key] = series_id
        return series_id

    def _metadata_for(self, ticker: str) -> Optional[Dict[str, Any]]:
        if self._metadata_provider is None:
            return None
        try:
            return self._metadata_provider(ticker)
        except Exception:
            logger.exception("metadata_provider failed for ticker=%s", ticker)
            return None

    # ── ParquetReader-mirrored interface ──

    def read_ticker(self, table: str, ticker: str) -> List[Dict]:
        series_id = self._resolve_series_id(table, ticker)
        if series_id is None:
            return []

        meta = self._metadata_for(ticker)

        if table == "hist_raw":
            rows = self._query(_BAR_SQL, {"sid": series_id})
            return [build_hist_raw_record(ticker, row, meta) for row in rows]

        if table == "hist_adj":
            rows = self._query(_BAR_MEP_SQL, {"sid": series_id})
            return [build_hist_adj_record(ticker, row, meta) for row in rows]

        if table == "bond_clean":
            rows = self._query(_BOND_CLEAN_SQL, {"sid": series_id})
            return [build_bond_clean_record(ticker, row, meta) for row in rows]

        if table == "yield_by_ticker":
            rows = self._query(_YIELD_SQL, {"sid": series_id})
            return [build_yield_record(ticker, row, meta) for row in rows]

        if table == "hist_fci":
            rows = self._query(_FCI_SQL, {"sid": series_id})
            return [build_fci_record(ticker, row) for row in rows]

        logger.error("PostgresReader.read_ticker: unsupported table %s", table)
        return []

    def read_market_partition(self, table: str, market: str) -> Dict[str, List[Dict]]:
        kind = MARKET_KIND.get(market)
        if kind is None:
            logger.error("PostgresReader.read_market_partition: unknown market %s", market)
            return {}

        mic = TABLE_MIC.get(table, "XBUE")

        if table == "hist_raw":
            rows = self._query(_BAR_PARTITION_SQL, {"mic": mic, "kind": kind})
            records = [
                build_hist_raw_record(row["ticker"], row, self._metadata_for(row["ticker"]))
                for row in rows
            ]
            return group_records_by_ticker(records)

        if table == "hist_adj":
            rows = self._query(_BAR_MEP_PARTITION_SQL, {"mic": mic, "kind": kind})
            records = [
                build_hist_adj_record(row["ticker"], row, self._metadata_for(row["ticker"]))
                for row in rows
            ]
            return group_records_by_ticker(records)

        logger.error("PostgresReader.read_market_partition: unsupported table %s", table)
        return {}

    def read_full_table(self, table: str) -> Any:
        # yield_by_date's {date: {submarket: [records]}} shape needs a
        # full-table scan across eod_bar+eod_metric grouped by date+submarket;
        # eod_metric is empty today and this is explicitly BLOQUEADA (§0/§4.6)
        # — raising keeps it out of SMART_LOADER_PG_TABLES by construction.
        raise NotImplementedError(
            f"PostgresReader.read_full_table({table!r}) not supported yet — "
            "eod_metric unpopulated (DPM-401/402); do not add to SMART_LOADER_PG_TABLES."
        )
