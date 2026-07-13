"""
Capa 1 (DPM-395 §7.2): SmartLoader backend routing, with fakeredis and a
mocked Tier 2 backend (no real S3/Postgres needed for these).
"""

from unittest.mock import MagicMock

import pytest
import redis as redis_module

from smart_loader.loader import SmartLoader

DUMMY_PG_DSN = "postgresql://user:pass@localhost:1/dwh"  # never actually connected to


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "SMART_LOADER_BACKEND",
        "SMART_LOADER_PG_DSN",
        "SMART_LOADER_PG_TABLES",
        "SMART_LOADER_PG_POOL_MIN",
        "SMART_LOADER_PG_POOL_MAX",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_loader(monkeypatch, fake_redis):
    monkeypatch.setattr(redis_module, "Redis", lambda **kwargs: fake_redis)
    return SmartLoader(s3_bucket="unused-bucket")


def test_default_backend_is_s3_and_no_pg_reader_instantiated(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)
    assert loader._backend == "s3"
    assert loader._pg_reader is None


def test_unknown_backend_falls_back_to_s3_with_warning(monkeypatch, fake_redis):
    monkeypatch.setenv("SMART_LOADER_BACKEND", "weird")
    loader = _make_loader(monkeypatch, fake_redis)
    assert loader._backend == "s3"
    assert loader._pg_reader is None


def test_pg_backend_without_dsn_raises_value_error_fail_fast(monkeypatch, fake_redis):
    monkeypatch.setenv("SMART_LOADER_BACKEND", "pg")
    monkeypatch.setattr(redis_module, "Redis", lambda **kwargs: fake_redis)
    with pytest.raises(ValueError):
        SmartLoader(s3_bucket="unused-bucket")


def test_pg_backend_default_tables(monkeypatch, fake_redis):
    monkeypatch.setenv("SMART_LOADER_BACKEND", "pg")
    monkeypatch.setenv("SMART_LOADER_PG_DSN", DUMMY_PG_DSN)
    loader = _make_loader(monkeypatch, fake_redis)
    assert loader._pg_tables == {"hist_adj", "hist_raw"}
    assert loader._pg_reader is not None


def test_pg_tables_env_restricts_routing(monkeypatch, fake_redis):
    monkeypatch.setenv("SMART_LOADER_BACKEND", "pg")
    monkeypatch.setenv("SMART_LOADER_PG_DSN", DUMMY_PG_DSN)
    monkeypatch.setenv("SMART_LOADER_PG_TABLES", "hist_adj")
    loader = _make_loader(monkeypatch, fake_redis)

    loader._pg_reader.read_ticker = MagicMock(return_value=[{"date": "2024-01-02"}])
    loader._parquet_reader.read_ticker = MagicMock(return_value=[{"date": "2024-01-03"}])

    adj = loader.get_ticker_series("hist_adj", "TSTGGAL")
    raw = loader.get_ticker_series("hist_raw", "TSTGGAL")

    loader._pg_reader.read_ticker.assert_called_once_with("hist_adj", "TSTGGAL")
    loader._parquet_reader.read_ticker.assert_called_once_with("hist_raw", "TSTGGAL")
    assert adj == [{"date": "2024-01-02"}]
    assert raw == [{"date": "2024-01-03"}]


def test_write_through_second_call_hits_cache_not_reader(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)
    loader._parquet_reader.read_ticker = MagicMock(
        return_value=[{"date": "2024-01-02", "close": 1}]
    )

    first = loader.get_ticker_series("hist_adj", "TSTGGAL")
    second = loader.get_ticker_series("hist_adj", "TSTGGAL")

    assert first == second == [{"date": "2024-01-02", "close": 1}]
    loader._parquet_reader.read_ticker.assert_called_once()


def test_get_prices_for_dates_e2e(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)
    loader._parquet_reader.read_ticker = MagicMock(
        return_value=[
            {"date": "2024-01-02", "closing_price": 103.0},
            {"date": "2024-01-03", "closing_price": 104.0},
        ]
    )

    result = loader.get_prices_for_dates("hist_raw", [("TSTGGAL", "2024-01-02")])
    assert result[("TSTGGAL", "2024-01-02")]["closing_price"] == 103.0


def test_read_market_partition_routes_to_pg_when_activated(monkeypatch, fake_redis):
    monkeypatch.setenv("SMART_LOADER_BACKEND", "pg")
    monkeypatch.setenv("SMART_LOADER_PG_DSN", DUMMY_PG_DSN)
    monkeypatch.setenv("SMART_LOADER_PG_TABLES", "hist_adj")
    loader = _make_loader(monkeypatch, fake_redis)

    loader._pg_reader.read_market_partition = MagicMock(
        return_value={"TSTGGAL": [{"date": "2024-01-02"}]}
    )

    result = loader.get_market_series("hist_adj", "stock")
    loader._pg_reader.read_market_partition.assert_called_once_with("hist_adj", "stock")
    assert result == {"TSTGGAL": [{"date": "2024-01-02"}]}
