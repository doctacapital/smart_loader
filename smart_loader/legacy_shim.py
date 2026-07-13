"""
Shim: converts dwh (schema `dwh`, Postgres) rows into the legacy Tier 2 record
shapes that parquet_reader.py / cronos historically produced (v1 contract).

Field names, ordering and the MEP formula here are NOT free choices — they
mirror cronos/utils/data_helpers.py:49-85 (record shape) and
cronos/utils/mep_calculator.py:101-135 (USD-MEP conversion, verbatim,
including the truthy-value / round(...,6) semantics) per DPM-395 spec §4/§5.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

# dwh.eod_bar column -> legacy v1 field name (DPM-395 §5)
BAR_COLUMN_MAP = {
    "open": "opening_price",
    "close": "closing_price",
    "low": "low_price",
    "high": "high_price",
    "volume": "trade_volume",
    "turnover": "effective_volume",
}

# Legacy v1 fields that get a *_usd_mep sibling (trade_volume has none — §5)
USD_MEP_FIELDS = [
    "opening_price",
    "closing_price",
    "low_price",
    "high_price",
    "effective_volume",
]

# Metadata enrichment fields sourced from metadata_provider(ticker) (§5.3)
META_FIELDS = [
    "settlement_period",
    "currency",
    "specie",
    "submarket",
    "current_closing_price",
]

EMPTY_META = {field: None for field in META_FIELDS}


def to_number(value: Any) -> Optional[float]:
    """PG numeric arrives as Decimal (or None) — always convert to float.
    NULL -> None (never NaN), per §3.2/§3.3."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def to_int(value: Any) -> Optional[int]:
    """Conceptually-integer fields (e.g. dtm) -> int, NULL -> None (§3.2)."""
    if value is None:
        return None
    return int(value)


def usd_mep_value(value: Optional[float], mep_rate: Optional[float]) -> Optional[float]:
    """Verbatim port of cronos/utils/mep_calculator.py:101-135:
    round(value / mep_rate, 6) if both are truthy, else None.

    Truthy (not just `is not None`) on purpose — matches the source, which
    uses `if peso_data.get(...)` / `if not mep_rate`. A 0 value collapses to
    None just like the original, even though it is not load-bearing for
    prices with `close > 0` (dwh CHECK constraint)."""
    if not mep_rate:
        return None
    if not value:
        return None
    return round(value / mep_rate, 6)


def build_bar_fields(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """dwh.eod_bar row (open/high/low/close/volume/turnover) -> legacy v1
    OHLCV field names (§4.1)."""
    return {
        legacy_name: to_number(row.get(pg_col))
        for pg_col, legacy_name in BAR_COLUMN_MAP.items()
    }


def build_hist_raw_record(ticker: str, row: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """§4.1: hist_raw record (no `id`). Field order mirrors
    cronos/utils/data_helpers.py:49-65 get_base_ticker_data."""
    meta = meta or EMPTY_META
    record: Dict[str, Any] = {
        "ticker": ticker,
        "settlement_period": meta.get("settlement_period"),
        "date": row["date"],
        "currency": meta.get("currency"),
        "specie": meta.get("specie"),
        "submarket": meta.get("submarket"),
    }
    record.update(build_bar_fields(row))
    record["current_closing_price"] = meta.get("current_closing_price")
    return record


def build_hist_adj_record(ticker: str, row: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """§4.2: hist_raw + USD-MEP columns."""
    record = build_hist_raw_record(ticker, row, meta)
    mep_rate = to_number(row.get("mep_rate"))
    for field in USD_MEP_FIELDS:
        record[f"{field}_usd_mep"] = usd_mep_value(record.get(field), mep_rate)
    return record


def build_bond_clean_record(ticker: str, row: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """§4.3 (codeada, no activada): eod_metric.clean_price/dirty_price + MEP/CCL."""
    meta = meta or EMPTY_META
    clean = to_number(row.get("clean_price"))
    dirty = to_number(row.get("dirty_price"))
    mep_rate = to_number(row.get("mep_rate"))
    ccl_rate = to_number(row.get("ccl_rate"))
    return {
        "date": row["date"],
        "ticker": ticker,
        "submarket": meta.get("submarket"),
        "closing_clean_price": clean,
        "closing_dirty_price": dirty,
        "closing_clean_price_usd_mep": usd_mep_value(clean, mep_rate),
        "closing_dirty_price_usd_mep": usd_mep_value(dirty, mep_rate),
        "closing_clean_price_usd_ccl": usd_mep_value(clean, ccl_rate),
        "closing_dirty_price_usd_ccl": usd_mep_value(dirty, ccl_rate),
    }


def build_yield_record(ticker: str, row: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """§4.4 (codeada, no activada): base hist_raw + L2 metrics.
    `accured_interest` keeps the legacy ES typo on purpose (contract)."""
    record = build_hist_raw_record(ticker, row, meta)
    extra = row.get("extra") or {}
    record.update({
        "tir": to_number(row.get("tir")),
        "tem": to_number(row.get("tem")),
        "tea": to_number(row.get("tea")),
        "tna": to_number(row.get("tna")),
        "tna_30_360": to_number(extra.get("tna_30_360")),
        "duration": to_number(row.get("duration")),
        "accured_interest": to_number(row.get("accrued_interest")),
        "margen": to_number(extra.get("margen")),
        "parity": to_number(row.get("parity")),
        "dtm": to_int(row.get("dtm")),
        "residual_value": to_number(row.get("residual_value")),
        "technical_value": to_number(extra.get("technical_value")),
        "clean_price": to_number(row.get("clean_price")),
        "dirty_price": to_number(row.get("dirty_price")),
    })
    return record


def build_fci_record(ticker: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """§4.5 (codeada, no activada): solo vcp+date se consumen realmente.
    `aum` no tiene columna dwh todavia -> None (pendiente, no decidir aca)."""
    return {
        "ticker": ticker,
        "vcp": to_number(row.get("vcp")),
        "aum": None,
        "date": row["date"],
    }


def group_records_by_ticker(records: List[Dict[str, Any]], key: str = "ticker") -> Dict[str, List[Dict[str, Any]]]:
    """Groups a flat record list into {ticker: [records]}, preserving order."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record[key], []).append(record)
    return grouped
