import os

import fakeredis
import pytest

PG_DSN_ENV = "SMART_LOADER_TEST_PG_DSN"
_SEED_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_seed.sql")


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
