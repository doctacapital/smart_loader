import io
import os
import threading

import fakeredis
import pytest

from smart_loader.parquet_reader import ParquetReader

PG_DSN_ENV = "SMART_LOADER_TEST_PG_DSN"
_SEED_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_seed.sql")
_REAL_PARQUET_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "by_ticker_real.parquet")
_REAL_PARQUET_S3_URI = "s3://docta-db-tables-025066256977/v1/yield_bonds/by_ticker.parquet"


@pytest.fixture(scope="session")
def pg_dsn():
    """Real dwh Postgres DSN for @pytest.mark.pg tests.

    Local setup (DPM-395 §7): cd docta-dwh && make up && make migrate, then
    export SMART_LOADER_TEST_PG_DSN=postgresql://postgres:postgres@localhost:54329/dwh
    """
    dsn = os.environ.get(PG_DSN_ENV)
    if not dsn:
        pytest.skip(f"{PG_DSN_ENV} not set — skipping pg-backed test (see docta-dwh compose)")
    return dsn


@pytest.fixture(scope="session")
def seeded_pg_dsn(pg_dsn):
    """Applies the synthetic fixture (§7.2) once per session, idempotently."""
    import psycopg

    with open(_SEED_PATH) as f:
        seed_sql = f.read()

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM dwh.series WHERE symbol = 'TSTGGAL'")
            if cur.fetchone() is None:
                cur.execute(seed_sql)
    return pg_dsn


@pytest.fixture
def fake_redis():
    """Fresh in-memory Redis per test (fakeredis) — no real server needed."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture(scope="session")
def real_parquet_bytes():
    """Real by_ticker.parquet snapshot (SPEC DPM-815 §11) — 1.331 tickers,
    288.694 rows, 2023-01-02..2026-08-11. Not committed (§6.2/.gitignore);
    skips cleanly if missing."""
    if not os.path.exists(_REAL_PARQUET_PATH):
        pytest.skip(
            f"real parquet fixture missing: {_REAL_PARQUET_PATH} "
            f"(copy from {_REAL_PARQUET_S3_URI})"
        )
    with open(_REAL_PARQUET_PATH, "rb") as f:
        return f.read()


@pytest.fixture(autouse=True)
def _reset_parquet_reader_module_state():
    """The P0-B single-flight/memo state lives at module level (SPEC §4.2),
    shared process-wide by design — but that means it leaks across tests
    unless reset. Runs before AND after every test."""
    from smart_loader import parquet_reader as pr_module

    if not hasattr(pr_module, "_MEMO_LOCK"):
        # Anti-placebo runs (SPEC §11) install a pre-fix smart-loader pin
        # (e.g. v0.1.7) that has none of the P0-B module state — nothing to
        # reset. No-op rather than AttributeError-ing every single test.
        yield
        return

    def _reset():
        with pr_module._MEMO_LOCK:
            pr_module._MEMO.clear()
            pr_module._KEY_LOCKS.clear()
        pr_module._MATERIALIZE_SEM = threading.Semaphore(
            pr_module._max_concurrent_materializations()
        )

    _reset()
    yield
    _reset()


class NoSuchKeyStub(Exception):
    """Stand-in for boto3's `s3.exceptions.NoSuchKey`."""


class _S3ExceptionsStub:
    NoSuchKey = NoSuchKeyStub


class RealYieldByTickerS3Stub:
    """Serves the real by_ticker.parquet fixture bytes for its one real S3
    key (v1/yield_bonds/by_ticker.parquet); raises NoSuchKey for anything
    else. Tracks call count/keys so tests can assert on S3 traffic."""

    def __init__(self, parquet_bytes: bytes):
        self._bytes = parquet_bytes
        self.exceptions = _S3ExceptionsStub()
        self.get_object_calls = 0
        self.get_object_keys = []

    def get_object(self, Bucket, Key):
        self.get_object_calls += 1
        self.get_object_keys.append(Key)
        if Key.endswith("yield_bonds/by_ticker.parquet"):
            return {"Body": io.BytesIO(self._bytes)}
        raise self.exceptions.NoSuchKey()

    def list_objects_v2(self, **kwargs):
        return {"CommonPrefixes": []}


@pytest.fixture
def real_parquet_reader(real_parquet_bytes):
    """A real ParquetReader wired to the real by_ticker.parquet fixture via
    an S3 stub — no network, no AWS creds needed."""
    reader = ParquetReader(bucket="docta-db-tables-025066256977", prefix="v1")
    reader._s3 = RealYieldByTickerS3Stub(real_parquet_bytes)
    return reader
