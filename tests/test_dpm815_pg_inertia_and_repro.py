"""
DPM-815 — T13 (G3: pg backend inert without psycopg installed) and T4
(real-data repro of the pre-fix OOM ratchet, in a fresh subprocess).

T13/T4 pass on both sides of the fix by design / are environment-gated —
not anti-placebo in the T1-T7/T10/T12/T14 sense.
"""
import io
import json
import os
import subprocess
import sys
import textwrap

import pytest
import redis as redis_module

REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
    cwd=os.path.dirname(__file__),
).stdout.strip()


# ── T13 — G3: PostgresReader/psycopg stay inert on backend=s3 ──────────


class _BlockPsycopgFinder:
    """Meta-path finder that makes `import psycopg` / `import psycopg_pool`
    fail as if they weren't installed — simulates the prod install (no [pg]
    extra) without needing a second venv."""

    def find_spec(self, name, path, target=None):
        if name in ("psycopg", "psycopg_pool"):
            raise ModuleNotFoundError(f"simulated: no module named {name!r} (T13)")
        return None


def test_t13_pg_backend_inert_without_psycopg(monkeypatch, fake_redis):
    for mod in list(sys.modules):
        if mod == "psycopg" or mod.startswith("psycopg."):
            del sys.modules[mod]
        if mod == "psycopg_pool" or mod.startswith("psycopg_pool."):
            del sys.modules[mod]

    blocker = _BlockPsycopgFinder()
    sys.meta_path.insert(0, blocker)
    try:
        import smart_loader
        from smart_loader.parquet_reader import ParquetReader

        monkeypatch.setattr(redis_module, "Redis", lambda **kwargs: fake_redis)
        loader = smart_loader.SmartLoader(s3_bucket="unused-bucket")

        assert loader._pg_reader is None
        assert isinstance(loader._reader_for("hist_adj"), ParquetReader)

        monkeypatch.setenv("SMART_LOADER_BACKEND", "pg")
        monkeypatch.delenv("SMART_LOADER_PG_DSN", raising=False)
        with pytest.raises(ValueError):
            smart_loader.SmartLoader(s3_bucket="unused-bucket")
    finally:
        sys.meta_path.remove(blocker)


# ── T4 — real-data OOM repro, fresh subprocess, instantaneous RSS ──────

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "by_ticker_real.parquet")

_REPRO_SCRIPT = textwrap.dedent(
    """
    import io, json, os, sys, threading, time

    sys.path.insert(0, {repo_smart_loader!r})

    import redis as redis_module
    import fakeredis

    from smart_loader.loader import SmartLoader
    from smart_loader.parquet_reader import ParquetReader

    FIXTURE_PATH = {fixture_path!r}
    ABSENT_TICKERS = ["TVPP", "BC36D", "COD7", "NDT11", "OZC8C", "PSXB"]

    with open(FIXTURE_PATH, "rb") as f:
        PARQUET_BYTES = f.read()


    class NoSuchKeyStub(Exception):
        pass


    class _Exc:
        NoSuchKey = NoSuchKeyStub


    class S3Stub:
        def __init__(self, data):
            self._data = data
            self.exceptions = _Exc()

        def get_object(self, Bucket, Key):
            if Key.endswith("yield_bonds/by_ticker.parquet"):
                return {{"Body": io.BytesIO(self._data)}}
            raise self.exceptions.NoSuchKey()

        def list_objects_v2(self, **kwargs):
            return {{"CommonPrefixes": []}}


    def rss_bytes():
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")


    peak = {{"bytes": 0}}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["bytes"] = max(peak["bytes"], rss_bytes())
            time.sleep(0.02)

    t_sampler = threading.Thread(target=sampler, daemon=True)
    t_sampler.start()

    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    redis_module.Redis = lambda **kwargs: fake_redis

    loader = SmartLoader(s3_bucket="unused-bucket")
    reader = ParquetReader(bucket="unused-bucket")
    reader._s3 = S3Stub(PARQUET_BYTES)
    loader._parquet_reader = reader

    def worker(ticker):
        loader.get_ticker_series("yield_by_ticker", ticker)

    threads = [
        threading.Thread(target=worker, args=(ABSENT_TICKERS[i % len(ABSENT_TICKERS)],))
        for i in range(16)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    stop.set()
    t_sampler.join(timeout=1)

    print(json.dumps({{"peak_rss_mb": peak["bytes"] / (1024 * 1024)}}))
    """
)


@pytest.mark.slow
def test_t4_oom_repro_real_data_fresh_subprocess():
    if sys.platform != "linux":
        pytest.skip(
            "T4 needs /proc/self/statm for a true instantaneous RSS sample "
            "(SPEC §11: 'no ru_maxrss') — Linux only. Run in CI or an EC2/container box."
        )
    if not os.path.exists(_FIXTURE_PATH):
        pytest.skip(f"real parquet fixture missing: {_FIXTURE_PATH}")

    repo_smart_loader = REPO_ROOT
    script = _REPRO_SCRIPT.format(repo_smart_loader=repo_smart_loader, fixture_path=_FIXTURE_PATH)

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"repro subprocess failed:\\nstdout={proc.stdout}\\nstderr={proc.stderr}"

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    peak_mb = result["peak_rss_mb"]
    assert peak_mb < 700, (
        f"post-fix: 16 threads x 6 absent tickers over 1 materialization must stay well under "
        f"the pre-fix repeated-materialization footprint (measured {peak_mb:.0f} MB)"
    )
