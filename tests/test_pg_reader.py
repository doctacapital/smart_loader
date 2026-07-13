"""
Capa 1 (DPM-395 §7.2): PostgresReader against the synthetic dwh fixture.
Requires a real local dwh Postgres (see conftest.pg_dsn / seeded_pg_dsn).
"""

import pytest

from smart_loader.pg_reader import PostgresReader

pytestmark = pytest.mark.pg


@pytest.fixture
def reader(seeded_pg_dsn):
    r = PostgresReader(seeded_pg_dsn)
    yield r
    r.close()


@pytest.fixture
def reader_with_meta(seeded_pg_dsn):
    def metadata_provider(ticker):
        if ticker == "TSTGGAL":
            return {
                "currency": "ARS",
                "specie": "P",
                "submarket": "STOCK",
                "current_closing_price": 108.0,
                "settlement_period": "24hs",
            }
        return None

    r = PostgresReader(seeded_pg_dsn, metadata_provider=metadata_provider)
    yield r
    r.close()


# ── §2 resolution ──

def test_resolve_direct_symbol(reader):
    records = reader.read_ticker("hist_raw", "TSTGGAL")
    assert len(records) == 6


def test_resolve_via_alias(reader):
    direct = reader.read_ticker("hist_raw", "TSTGGAL")
    via_alias = reader.read_ticker("hist_raw", "TSTOLD")
    assert len(via_alias) == 6
    # Same underlying series/bars — only the `ticker` field differs (each
    # record echoes back the ticker it was queried with).
    direct_no_ticker = [{k: v for k, v in r.items() if k != "ticker"} for r in direct]
    via_alias_no_ticker = [{k: v for k, v in r.items() if k != "ticker"} for r in via_alias]
    assert via_alias_no_ticker == direct_no_ticker
    assert all(r["ticker"] == "TSTOLD" for r in via_alias)


def test_resolve_unknown_ticker_returns_empty_list_never_raises(reader):
    assert reader.read_ticker("hist_raw", "DOES_NOT_EXIST") == []


def test_bond_excluded_from_hist_raw_by_kind_filter(reader):
    # TSTAL30 has eod_bar rows and a matching series.symbol, but kind=BOND
    # is not in hist_raw's allowed kinds (STOCK, CEDEAR) — must resolve to [].
    assert reader.read_ticker("hist_raw", "TSTAL30") == []


# ── §3 record shape ──

def test_hist_raw_record_shape_and_types(reader):
    records = reader.read_ticker("hist_raw", "TSTGGAL")
    assert len(records) > 0

    first = records[0]
    expected_keys = {
        "ticker", "settlement_period", "date", "currency", "specie", "submarket",
        "opening_price", "closing_price", "low_price", "high_price",
        "trade_volume", "effective_volume", "current_closing_price",
    }
    assert set(first.keys()) == expected_keys
    assert "id" not in first

    assert first["date"] == "2024-01-02"
    assert isinstance(first["date"], str)
    assert isinstance(first["closing_price"], float)
    assert first["ticker"] == "TSTGGAL"
    # currency comes from dwh.series.trade_currency, per-serie (planner
    # resolution of §5.3 halt) — TSTGGAL's synthetic series is ARS.
    assert first["currency"] == "ARS"


def test_hist_raw_dates_ascending(reader):
    records = reader.read_ticker("hist_raw", "TSTGGAL")
    dates = [r["date"] for r in records]
    assert dates == sorted(dates)


def test_hist_raw_null_open_becomes_none_not_nan(reader):
    records = reader.read_ticker("hist_raw", "TSTGGAL")
    day3 = next(r for r in records if r["date"] == "2024-01-04")
    assert day3["opening_price"] is None


def test_hist_adj_record_shape(reader):
    records = reader.read_ticker("hist_adj", "TSTGGAL")
    assert len(records) == 6
    expected_extra = {
        "opening_price_usd_mep", "closing_price_usd_mep", "low_price_usd_mep",
        "high_price_usd_mep", "effective_volume_usd_mep",
    }
    assert expected_extra <= set(records[0].keys())
    # trade_volume has no usd_mep sibling (§5)
    assert "trade_volume_usd_mep" not in records[0]


def test_hist_adj_mep_calculated_when_rate_present(reader):
    records = reader.read_ticker("hist_adj", "TSTGGAL")
    day1 = next(r for r in records if r["date"] == "2024-01-02")
    # closing_price=103, mep=1000 -> round(103/1000, 6)
    assert day1["closing_price_usd_mep"] == pytest.approx(0.103, rel=1e-9)


def test_hist_adj_mep_none_when_rate_missing_for_date(reader):
    records = reader.read_ticker("hist_adj", "TSTGGAL")
    last_day = next(r for r in records if r["date"] == "2024-01-09")
    assert last_day["closing_price_usd_mep"] is None
    assert last_day["opening_price_usd_mep"] is None


def test_hist_adj_mep_is_a_global_rate_shared_across_tickers(reader):
    # dwh.fx_rate has no series_id — MEP is one global rate per date, applied
    # to every ticker via the date join, not sourced per-symbol. TSTNVDA gets
    # the same MEP rows as TSTGGAL even though the fixture only inserted them
    # "for TSTGGAL" — that's the correct real-world semantics (§4.2).
    records = reader.read_ticker("hist_adj", "TSTNVDA")
    assert len(records) == 6
    last_day = next(r for r in records if r["date"] == "2024-01-09")
    assert last_day["closing_price_usd_mep"] is None  # no MEP row for this date at all
    other_day = next(r for r in records if r["date"] == "2024-01-02")
    assert other_day["closing_price_usd_mep"] == pytest.approx(905 / 1000, rel=1e-9)


# ── §5.3 metadata provider ──

def test_metadata_provider_present_enriches_record(reader_with_meta):
    records = reader_with_meta.read_ticker("hist_raw", "TSTGGAL")
    assert records[0]["specie"] == "P"
    assert records[0]["submarket"] == "STOCK"
    assert records[0]["current_closing_price"] == 108.0
    assert records[0]["settlement_period"] == "24hs"


def test_metadata_provider_currency_key_is_ignored_series_wins(seeded_pg_dsn):
    # Planner resolution of §5.3: currency is per-serie (dwh.series), never
    # sourced from metadata_provider — even if a provider returns a
    # (wrong/stale) "currency" key, it must not leak into the record.
    def metadata_provider(ticker):
        return {"currency": "XXX", "specie": None, "submarket": None,
                "current_closing_price": None, "settlement_period": "24hs"}

    reader = PostgresReader(seeded_pg_dsn, metadata_provider=metadata_provider)
    try:
        records = reader.read_ticker("hist_raw", "TSTGGAL")
        assert records[0]["currency"] == "ARS"  # dwh.series.trade_currency, not "XXX"
    finally:
        reader.close()


def test_metadata_provider_absent_fields_are_none_except_currency(reader):
    # currency is populated from dwh.series regardless of metadata_provider.
    records = reader.read_ticker("hist_raw", "TSTGGAL")
    assert records[0]["currency"] == "ARS"
    assert records[0]["specie"] is None
    assert records[0]["submarket"] is None
    assert records[0]["current_closing_price"] is None
    assert records[0]["settlement_period"] is None


def test_metadata_provider_unknown_ticker_currency_still_resolves(reader_with_meta):
    # TSTNVDA isn't in the metadata_provider's stub, but currency doesn't
    # depend on it — it still resolves from dwh.series.
    records = reader_with_meta.read_ticker("hist_raw", "TSTNVDA")
    assert records[0]["currency"] == "ARS"
    assert records[0]["specie"] is None


# ── §4.6 read_market_partition ──

def test_read_market_partition_groups_by_ticker(reader):
    partition = reader.read_market_partition("hist_raw", "stock")
    assert "TSTGGAL" in partition
    assert "TSTNVDA" not in partition  # NVDA is CEDEAR, not STOCK
    assert len(partition["TSTGGAL"]) == 6
    assert partition["TSTGGAL"][0]["currency"] == "ARS"


def test_read_market_partition_cedear(reader):
    partition = reader.read_market_partition("hist_adj", "cedear")
    assert "TSTNVDA" in partition
    assert "TSTGGAL" not in partition
    assert len(partition["TSTNVDA"]) == 6
    # MEP is a global per-date rate (no series_id on fx_rate) — first day has one.
    assert partition["TSTNVDA"][0]["closing_price_usd_mep"] is not None


def test_read_market_partition_unknown_market_returns_empty(reader):
    assert reader.read_market_partition("hist_raw", "not_a_market") == {}


# ── §4.6 read_full_table (yield_by_date) ──

def test_read_full_table_yield_by_date_not_implemented(reader):
    with pytest.raises(NotImplementedError):
        reader.read_full_table("yield_by_date")


# ── §7.2 bloqueadas (codeadas, sin datos hoy) ──

@pytest.mark.parametrize("table", ["bond_clean", "yield_by_ticker", "hist_fci"])
def test_blocked_tables_resolve_empty_without_data(reader, table):
    # No eod_metric/FCI data seeded (DPM-401/402) — must return [] cleanly,
    # not raise, even though the SQL path is coded.
    assert reader.read_ticker(table, "TSTAL30" if table != "hist_fci" else "TSTFCI") == []
