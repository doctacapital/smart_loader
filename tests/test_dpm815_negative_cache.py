"""
DPM-815 P0-A — negative-cache of empty ticker series (SPEC §3, §11 T1/T2/T6/T7/T9,
plus the negative-cache half of T11).

Anti-placebo (§11 nominally lists T1, T2, T6, T7 here as required to FAIL
against pre-fix code — pin `smart-loader @ v0.1.7`, byte-identical to main
pre-DPM-815). Verified empirically by installing that pin and re-running
this file (coder report has the full run + a spec discrepancy flagged
there). Result: **T1 fails pre-fix as expected. T2, T6 and T7 pass on BOTH
sides** — by construction, not by test weakness:
  - T2: pre-fix has no negative-cache mechanism at all, so "no marker gets
    written on an S3 error" is vacuously true with nothing there to poison.
  - T6: pre-fix's HASH branch already returns unconditionally on
    `type(cache_key) == "hash"`, without ever looking at any `:__miss__`
    key — I-1 was already true before this change existed.
  - T7: pre-fix's `flush_tier2_cache()` was already a type-agnostic
    `ts:cache:*` scan+delete, so it already swept anything under that
    prefix, markers included.
They're reclassified here as no-regression tests (like T5/T8/T9/T11/T13),
not anti-placebo ones.
"""
import redis as redis_module

from smart_loader.loader import SmartLoader

NEG_TTL_DEFAULT = 3600


class CountingEmptyReader:
    """Stand-in Tier 2 reader: always an authoritative empty read."""

    def __init__(self):
        self.calls = 0

    def read_ticker(self, table, ticker):
        self.calls += 1
        return []

    def consume_degraded(self):
        return False


class NeverCalledReader:
    def read_ticker(self, table, ticker):
        raise AssertionError("reader should not be called — positive cache must win")

    def consume_degraded(self):
        return False


def _make_loader(monkeypatch, fake_redis):
    monkeypatch.setattr(redis_module, "Redis", lambda **kwargs: fake_redis)
    return SmartLoader(s3_bucket="unused-bucket")


# ── T1 ──────────────────────────────────────────────────────────────────


def test_t1_empty_miss_is_negative_cached(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)
    reader = CountingEmptyReader()
    loader._parquet_reader = reader

    r1 = loader.get_ticker_series("yield_by_ticker", "TVPP")
    r2 = loader.get_ticker_series("yield_by_ticker", "TVPP")

    assert r1 == []
    assert r2 == []
    assert reader.calls == 1, "second call must be served from the negative-cache marker"

    neg_key = "ts:cache:yield_by_ticker:TVPP:__miss__"
    assert fake_redis.exists(neg_key)
    ttl = fake_redis.ttl(neg_key)
    assert NEG_TTL_DEFAULT - 400 < ttl <= NEG_TTL_DEFAULT


# ── T2 — anti-poisoning ────────────────────────────────────────────────


def test_t2_s3_error_is_never_negative_cached(monkeypatch, fake_redis, real_parquet_reader):
    loader = _make_loader(monkeypatch, fake_redis)

    class AlwaysErrorS3Stub:
        def __init__(self):
            self.get_object_calls = 0
            self.exceptions = real_parquet_reader._s3.exceptions

        def get_object(self, Bucket, Key):
            self.get_object_calls += 1
            raise RuntimeError("simulated S3 timeout / ClientError")

        def list_objects_v2(self, **kwargs):
            return {"CommonPrefixes": []}

    stub = AlwaysErrorS3Stub()
    real_parquet_reader._s3 = stub
    loader._parquet_reader = real_parquet_reader

    r1 = loader.get_ticker_series("yield_by_ticker", "AL30")
    r2 = loader.get_ticker_series("yield_by_ticker", "AL30")

    assert r1 == []
    assert r2 == []
    neg_key = "ts:cache:yield_by_ticker:AL30:__miss__"
    assert not fake_redis.exists(neg_key), "an S3 error must never be negative-cached"
    assert stub.get_object_calls == 2, "no negative cache means the 2nd call retries S3"


# ── T6 — positive always wins over a (possibly stale) marker (I-1) ─────


def test_t6_positive_hash_wins_over_marker(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)
    table, ticker = "yield_by_ticker", "AL30"
    cache_key = f"ts:cache:{table}:{ticker}"
    neg_key = cache_key + ":__miss__"

    records = [{"date": "2023-01-02", "closing_price": 8185.0}]
    loader._cache_as_hash(cache_key, records)
    fake_redis.setex(neg_key, 3600, "1")

    loader._parquet_reader = NeverCalledReader()

    result = loader.get_ticker_series(table, ticker)

    assert result == records
    # NOTE (spec discrepancy, flagged in coder report): §3.3's exact
    # pseudocode marks the HASH-hit branch "SIN CAMBIOS" and I-2 requires
    # zero extra Redis calls on the hit path, which rules out a `delete`
    # here. §11's T6 row instead asserts "el marker queda borrado" — the
    # two are in tension. We follow §3.3/I-2 (authoritative "orden EXACTO"
    # + explicit hot-path invariant) and assert the invariant that actually
    # matters: a leftover marker never contaminates a positive read.
    # Marker survives; that's proven harmless by I-1, not asserted here.


# ── T7 — cronos flush sweeps markers too (type-agnostic scan) ──────────


def test_t7_flush_sweeps_negative_markers(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)

    for ticker in ("TVPP", "BC36D", "COD7"):
        fake_redis.setex(f"ts:cache:yield_by_ticker:{ticker}:__miss__", 3600, "1")

    deleted = loader.flush_tier2_cache()

    assert deleted == 3
    remaining = list(fake_redis.scan_iter(match="ts:cache:*:__miss__"))
    assert remaining == []


# ── T9 — unknown table never negative-cached (non-authoritative) ───────


def test_t9_unknown_table_no_marker_and_reports_degraded(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)

    result = loader.get_ticker_series("tabla_inexistente", "X")

    assert result == []
    neg_key = "ts:cache:tabla_inexistente:X:__miss__"
    assert not fake_redis.exists(neg_key), (
        "the loader's own consume_degraded() check inside _may_cache_negative "
        "must have vetoed the write"
    )

    # consume_degraded() is read-and-reset (§3.5 point 6), so by the time the
    # assertion above ran, the loader had already consumed it internally to
    # decide not to cache. Verify the reader-level guard directly, on a fresh
    # call, to prove *why* the marker above was vetoed.
    from smart_loader.parquet_reader import ParquetReader

    reader = ParquetReader(bucket="unused-bucket")
    assert reader.read_ticker("tabla_inexistente", "X") == []
    assert reader.consume_degraded() is True, "unknown table must mark the read non-authoritative"


# ── T11 (negative-cache half) — kill switch ─────────────────────────────


def test_t11_neg_cache_kill_switch_restores_pre_fix_behaviour(monkeypatch, fake_redis):
    monkeypatch.setenv("SMART_LOADER_NEG_CACHE", "off")
    loader = _make_loader(monkeypatch, fake_redis)
    reader = CountingEmptyReader()
    loader._parquet_reader = reader

    loader.get_ticker_series("yield_by_ticker", "TVPP")
    loader.get_ticker_series("yield_by_ticker", "TVPP")

    assert reader.calls == 2, "kill-switch off must restore exact pre-fix behaviour"
    assert not fake_redis.exists("ts:cache:yield_by_ticker:TVPP:__miss__")
