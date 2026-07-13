"""
Capa 2 golden (DPM-395 §7.3): v1 (S3 Parquet) vs v2 (PostgresReader/dwh),
for the tables that are actually servable today (hist_adj, hist_raw).

Fixtures come from scripts/capture_golden.py, run manually against prod
creds (not available to this harness/CI) — see tests/golden/. Every test
here skips cleanly if its fixture is missing, and the pg-backed ones also
skip if SMART_LOADER_TEST_PG_DSN isn't set (conftest.pg_dsn).
"""

import json
import math
import os

import pytest

from smart_loader.pg_reader import PostgresReader

pytestmark = pytest.mark.pg

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")

WITNESSES_SERVABLE = [
    ("hist_adj", "GGAL"),
    ("hist_raw", "GGAL"),
    ("hist_adj", "NVDA"),
    ("hist_raw", "NVDA"),
]

# §0/§7.3: eod_metric/FCI series are empty today — these witnesses stay
# skipped until DPM-401/402 populate the underlying dwh tables.
WITNESSES_BLOCKED = [
    ("bond_clean", "AL30"),
    ("yield_by_ticker", "AL30"),
    ("yield_by_ticker", "AY24"),
    ("hist_fci", "SOME_FCI"),
]


def _v1_path(table, ticker):
    return os.path.join(GOLDEN_DIR, "v1", f"{table}__{ticker}.json")


def _seed_path(ticker):
    return os.path.join(GOLDEN_DIR, "dwh_seed", f"{ticker}.sql")


def _meta_path(ticker):
    return os.path.join(GOLDEN_DIR, "meta", f"{ticker}.json")


def _load_v1(table, ticker):
    path = _v1_path(table, ticker)
    if not os.path.exists(path):
        pytest.skip(f"golden fixture missing: {path} (run scripts/capture_golden.py)")
    with open(path) as f:
        return json.load(f)


def _apply_seed(dsn, ticker):
    import psycopg

    path = _seed_path(ticker)
    if not os.path.exists(path):
        pytest.skip(f"dwh_seed fixture missing: {path} (run scripts/capture_golden.py)")
    with open(path) as f:
        sql = f.read()
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def _load_meta_stub(ticker):
    path = _meta_path(ticker)
    if not os.path.exists(path):
        return lambda t: None
    with open(path) as f:
        row = json.load(f)
    if row is None:
        return lambda t: None
    return lambda t: row if t == ticker else None


META_FIELDS = ("settlement_period", "currency", "specie", "submarket", "current_closing_price")

# Planner resolution of the §5.3 halt (DPM-395): only these two META_FIELDS
# are asserted on the last record. `currency`/`specie`/`current_closing_price`
# are report-only — logged as `meta_report`, never asserted — because
# `specie`/`current_closing_price` have no dwh source at all (permanent
# None) and `currency` now comes from dwh.series.trade_currency rather than
# the legacy per-ticker metadata, so a literal mismatch is expected/possible.
# TODO(planner): if capture_golden.py shows v1's `currency` is byte-for-byte
# identical to `series.trade_currency` across witnesses, promote `currency`
# to ASSERTED_META_FIELDS.
ASSERTED_META_FIELDS = ("settlement_period", "submarket")
REPORT_ONLY_META_FIELDS = ("currency", "specie", "current_closing_price")


def compare_records(v1_records, v2_records):
    """Classifies differences per §8: fechas-solo-v1 / solo-v2 / value diffs.
    NaN treated as equivalent to None. META_FIELDS compared only on the last
    record (§7.3); within META_FIELDS, only ASSERTED_META_FIELDS feed
    `value_diffs` — REPORT_ONLY_META_FIELDS go to `meta_report` instead.
    Never asserts opaquely — returns a structured report."""
    by_date_v1 = {r["date"]: r for r in v1_records}
    by_date_v2 = {r["date"]: r for r in v2_records}

    dates_only_v1 = sorted(set(by_date_v1) - set(by_date_v2))
    dates_only_v2 = sorted(set(by_date_v2) - set(by_date_v1))
    common_dates = sorted(set(by_date_v1) & set(by_date_v2))

    value_diffs = {}
    meta_report = {}
    last_date = common_dates[-1] if common_dates else None

    for date in common_dates:
        r1, r2 = by_date_v1[date], by_date_v2[date]
        keys = (set(r1) | set(r2)) - {"id"}
        for key in keys:
            if key in META_FIELDS and date != last_date:
                continue  # META_FIELDS: only meaningful on the last record (§7.3)
            v1v, v2v = r1.get(key), r2.get(key)
            if _nan_as_none(v1v) is None and _nan_as_none(v2v) is None:
                continue
            if isinstance(v1v, (int, float)) and isinstance(v2v, (int, float)):
                if math.isclose(v1v, v2v, rel_tol=1e-9, abs_tol=1e-9):
                    continue
            elif v1v == v2v:
                continue
            if key in REPORT_ONLY_META_FIELDS:
                meta_report.setdefault(date, {})[key] = (v1v, v2v)
            else:
                value_diffs.setdefault(date, {})[key] = (v1v, v2v)

    return {
        "dates_only_v1": dates_only_v1,
        "dates_only_v2": dates_only_v2,
        "value_diffs": value_diffs,
        "meta_report": meta_report,
    }


def _nan_as_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


@pytest.mark.parametrize("table,ticker", WITNESSES_SERVABLE)
def test_golden_v1_vs_v2(seeded_pg_dsn, table, ticker):
    v1_records = _load_v1(table, ticker)
    _apply_seed(seeded_pg_dsn, ticker)

    reader = PostgresReader(seeded_pg_dsn, metadata_provider=_load_meta_stub(ticker))
    try:
        v2_records = reader.read_ticker(table, ticker)
    finally:
        reader.close()

    # Anti-vacío guard (§7.3 pata 2): golden falla si v1 o v2 devuelven 0.
    assert len(v1_records) > 0, f"v1 golden for {table}/{ticker} is empty — fixture capture bug"
    assert len(v2_records) > 0, f"v2 (PostgresReader) for {table}/{ticker} returned 0 records"

    report = compare_records(v1_records, v2_records)

    # Report-only (planner decision, §5.3): log, never assert.
    if report["meta_report"]:
        print(f"[meta_report, non-blocking] {table}/{ticker}: {report['meta_report']}")

    assert report["value_diffs"] == {}, f"value diffs for {table}/{ticker}: {report['value_diffs']}"
    assert report["dates_only_v1"] == [], (
        f"dates only in v1 for {table}/{ticker} (v2 coverage gap): {report['dates_only_v1']}"
    )
    assert report["dates_only_v2"] == [], (
        f"dates only in v2 for {table}/{ticker} (v1 filtered these — see legacy_bars.py "
        f"ghost filter, §8): {report['dates_only_v2']}"
    )


@pytest.mark.parametrize("table,ticker", WITNESSES_BLOCKED)
def test_golden_blocked_tables(table, ticker):
    pytest.skip(reason="tabla sin poblar: eod_metric/FCI vacías — DPM-401/402")


def test_comparator_catches_injected_diff(seeded_pg_dsn):
    """Anti-placebo pata 1 (§7.3): mutate one dwh row, comparator MUST flag it."""
    import psycopg

    table, ticker = "hist_adj", "GGAL"
    v1_records = _load_v1(table, ticker)
    _apply_seed(seeded_pg_dsn, ticker)

    reader = PostgresReader(seeded_pg_dsn, metadata_provider=_load_meta_stub(ticker))
    try:
        baseline = reader.read_ticker(table, ticker)
        baseline_report = compare_records(v1_records, baseline)
        assert baseline_report["value_diffs"] == {}, "fixture already diverges before mutation"

        with psycopg.connect(seeded_pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE dwh.eod_bar SET close = close + 1
                    WHERE series_id = (SELECT series_id FROM dwh.series WHERE symbol = %s)
                    AND trade_date = (
                        SELECT trade_date FROM dwh.eod_bar
                        WHERE series_id = (SELECT series_id FROM dwh.series WHERE symbol = %s)
                        ORDER BY trade_date LIMIT 1
                    )
                    """,
                    (ticker, ticker),
                )
                try:
                    mutated = reader.read_ticker(table, ticker)
                    mutated_report = compare_records(v1_records, mutated)
                    assert mutated_report["value_diffs"] != {}, (
                        "comparator failed to catch an injected +1 diff — anti-placebo check failed"
                    )
                finally:
                    cur.execute(
                        """
                        UPDATE dwh.eod_bar SET close = close - 1
                        WHERE series_id = (SELECT series_id FROM dwh.series WHERE symbol = %s)
                        AND trade_date = (
                            SELECT trade_date FROM dwh.eod_bar
                            WHERE series_id = (SELECT series_id FROM dwh.series WHERE symbol = %s)
                            ORDER BY trade_date LIMIT 1
                        )
                        """,
                        (ticker, ticker),
                    )
    finally:
        reader.close()
