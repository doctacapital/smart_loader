"""
DPM-815 — T5 (golden: positive path unchanged) and T8 (mixed-fleet
compat: v0.1.6 reader against Redis written by the fixed loader).

T5/T8/T9 pass on both sides of the fix by design (no-regression tests,
SPEC §11) — not anti-placebo.
"""
import subprocess
import types

import pytest
import redis as redis_module

from smart_loader.loader import SmartLoader

REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
    cwd=__import__("os").path.dirname(__file__),
).stdout.strip()


def _make_loader(monkeypatch, fake_redis):
    monkeypatch.setattr(redis_module, "Redis", lambda **kwargs: fake_redis)
    return SmartLoader(s3_bucket="unused-bucket")


# ── T5 — golden: the positive path is byte-identical to today ──────────


def test_t5_golden_al30_gd30(monkeypatch, fake_redis, real_parquet_reader):
    loader = _make_loader(monkeypatch, fake_redis)
    loader._parquet_reader = real_parquet_reader

    al30 = loader.get_ticker_series("yield_by_ticker", "AL30")
    gd30 = loader.get_ticker_series("yield_by_ticker", "GD30")

    assert len(al30) == 881
    assert len(gd30) == 881

    record = next(r for r in al30 if r["date"] == "2023-01-02")
    assert record["closing_price"] == 8185.0
    assert record["opening_price"] == 8166.5
    assert record["low_price"] == 8000.0
    assert record["high_price"] == 8268.0
    assert record["settlement_period"] == "48hs"
    assert record["submarket"] == "HARD_DOLLAR"
    assert record["currency"] == "ARS"
    assert record["specie"] == "P"
    assert record["id"] == 174031


# ── T8 — mixed-fleet compat: v0.1.6 loader reads Redis written by the ──
# fixed loader without breaking (SPEC §7.1 row 1 + §11 T8) ──────────────


def _load_v016_loader_module():
    """Materializes v0.1.6's loader.py source via `git show` into a throwaway
    module. It resolves `from smart_loader.parquet_reader import ParquetReader`
    to whatever is currently installed (the fixed reader) — inert for what
    T8 asserts (marker/key shape access), per SPEC §11 T8 caveat (a)."""
    if not REPO_ROOT:
        pytest.skip("not a git checkout — can't `git show v0.1.6:...`")
    proc = subprocess.run(
        ["git", "show", "v0.1.6:smart_loader/loader.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"tag v0.1.6 not available in this checkout: {proc.stderr.strip()}")

    module = types.ModuleType("smart_loader_v016_loader")
    exec(compile(proc.stdout, "v0.1.6/smart_loader/loader.py", "exec"), module.__dict__)
    return module


def test_t8_v016_loader_against_redis_written_by_fixed_loader(monkeypatch, fake_redis, real_parquet_reader):
    v016 = _load_v016_loader_module()

    # New (fixed) writer populates the shared Redis: AL30 positive + TVPP
    # negative marker — as would happen with nexus already bumped to v0.1.8
    # while some other consumer is still on v0.1.6.
    new_loader = _make_loader(monkeypatch, fake_redis)
    new_loader._parquet_reader = real_parquet_reader
    new_loader.get_ticker_series("yield_by_ticker", "AL30")
    new_loader.get_ticker_series("yield_by_ticker", "TVPP")
    assert fake_redis.exists("ts:cache:yield_by_ticker:TVPP:__miss__")

    # Old (v0.1.6) reader, same fake_redis, same real S3-backed reader.
    monkeypatch.setattr(redis_module, "Redis", lambda **kwargs: fake_redis)
    old_loader = v016.SmartLoader(s3_bucket="unused-bucket")
    old_loader._parquet_reader = real_parquet_reader

    al30 = old_loader.get_ticker_series("yield_by_ticker", "AL30")
    assert len(al30) == 881, "v0.1.6 must read the positive hash exactly as before"

    tvpp = old_loader.get_ticker_series("yield_by_ticker", "TVPP")
    assert tvpp == [], "v0.1.6 ignores the marker key entirely and goes to S3, gets [] without excepting"

    pairs = old_loader.get_prices_for_dates(
        "yield_by_ticker", [("AL30", "2023-01-02"), ("TVPP", "2023-01-02")]
    )
    assert ("AL30", "2023-01-02") in pairs
    assert pairs[("AL30", "2023-01-02")]["closing_price"] == 8185.0
    assert ("TVPP", "2023-01-02") not in pairs
    # No WRONGTYPE: get_prices_for_dates completing without raising is the assertion.
