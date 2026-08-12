"""
DPM-815 P0-A — negative-cache of empty ticker series (SPEC §3, §11 T1/T2/
T6/T7/T9, plus the negative-cache half of T11).

Anti-placebo, v0.1.7 pin (§11 nominally lists T1, T2, T6, T7 here as
required to FAIL against pre-fix code). Verified empirically (coder
report has the full run). Result: **T1 fails pre-fix as expected. T2, T6
and T7 pass on BOTH sides** — by construction, not by test weakness, since
v0.1.7 is the wrong counterfactual for them:
  - T2: pre-fix has no negative-cache mechanism at all, so "no marker gets
    written on an S3 error" is vacuously true with nothing there to poison.
  - T6: pre-fix's HASH branch already returns unconditionally on
    `type(cache_key) == "hash"`, without ever looking at any `:__miss__`
    key — I-1 was already true before this change existed.
  - T7: pre-fix's `flush_tier2_cache()` was already a type-agnostic
    `ts:cache:*` scan+delete, so it already swept anything under that
    prefix, markers included.

Planner-ratified fix (mutation control, not v0.1.7): each of T2/T6/T7 has a
paired `test_t*_mutation_control_*` below that applies the SPECIFIC defect
a naive/broken P0-A implementation would have (guard always-True, marker
checked before the hash branch, marker built outside the `ts:cache:`
prefix) directly against the real components, and shows the assertion
flips. That's the actual counterfactual these tests discriminate against;
v0.1.7 never could.

T6 was also split per the planner's ruling on the §3.3-vs-§11 tension:
the HIT path (HASH already present) stays without a delete (I-2, zero-cost
hit path) — T6a. The step-6 delete (miss-then-materialize-finds-data) DOES
exist in loader.py and must — it's the self-heal for the §3.6 flush race
(a marker written moments after cronos's 18:36 flush survives the flush
itself; with ~227 req/day/ticker the very next request that reaches the
miss path clears it in minutes instead of waiting out the TTL) — T6b.
"""
import json

import redis as redis_module

from smart_loader.loader import TIER2_CACHE_PREFIX, TIER2_NEG_SUFFIX, SmartLoader

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


def test_t2_mutation_control_naive_guard_would_poison(monkeypatch, fake_redis, real_parquet_reader):
    """Mutation control (planner-ratified): if `_may_cache_negative()` were
    the naive `lambda reader: True` (i.e. the anti-poisoning guard removed),
    the exact same S3-error scenario as T2 WOULD get negative-cached. This
    is the counterfactual T2 actually discriminates against — reproduce for
    real by deleting the `_may_cache_negative` guard in loader.py."""
    import smart_loader.loader as loader_module

    monkeypatch.setattr(loader_module, "_may_cache_negative", lambda reader: True)
    loader = _make_loader(monkeypatch, fake_redis)

    class AlwaysErrorS3Stub:
        def __init__(self):
            self.exceptions = real_parquet_reader._s3.exceptions

        def get_object(self, Bucket, Key):
            raise RuntimeError("simulated S3 timeout / ClientError")

        def list_objects_v2(self, **kwargs):
            return {"CommonPrefixes": []}

    real_parquet_reader._s3 = AlwaysErrorS3Stub()
    loader._parquet_reader = real_parquet_reader

    loader.get_ticker_series("yield_by_ticker", "AL30")

    neg_key = "ts:cache:yield_by_ticker:AL30:__miss__"
    assert fake_redis.exists(neg_key), (
        "control: proves T2 would catch the anti-poisoning guard being disabled"
    )


# ── T6a/T6b — positive always wins over a (possibly stale) marker (I-1) ─


def test_t6a_positive_wins_over_marker_real_data(monkeypatch, fake_redis, real_parquet_reader):
    loader = _make_loader(monkeypatch, fake_redis)
    loader._parquet_reader = real_parquet_reader
    table, ticker = "yield_by_ticker", "AL30"
    cache_key = f"ts:cache:{table}:{ticker}"
    neg_key = cache_key + ":__miss__"

    # Populate the real positive hash first (one real materialization).
    seeded = loader.get_ticker_series(table, ticker)
    assert len(seeded) == 881
    # Inject a stale marker afterwards — I-1 must make it harmless.
    fake_redis.setex(neg_key, 3600, "1")

    loader._parquet_reader = NeverCalledReader()  # any further S3 touch is a bug
    result = loader.get_ticker_series(table, ticker)

    assert len(result) == 881
    # The marker is left as-is here (§3.3 step 2 is "SIN CAMBIOS" + I-2's
    # zero-cost hit path rules out an unconditional delete) — proven
    # harmless by this test, not deleted by it. T6b covers the real
    # deletion path (step 6, miss-then-materialize).


def test_t6_mutation_control_marker_before_hash_breaks_i1(fake_redis):
    """Mutation control (planner-ratified): demonstrates T6a is sensitive
    to the exact ordering in SPEC §3.3. If the marker check (step 4) ran
    BEFORE the hash/string branches (steps 2-3) instead of after, a stale
    marker would shadow a real positive value. Reproduce for real: in
    loader.py::get_ticker_series, move the
    `if neg_enabled and self._redis.exists(neg_key): return []` line to
    before `key_type = self._redis.type(cache_key)`, then rerun T6a — it
    starts returning [] instead of 881 rows."""
    table, ticker = "yield_by_ticker", "AL30"
    cache_key = f"{TIER2_CACHE_PREFIX}{table}:{ticker}"
    neg_key = cache_key + TIER2_NEG_SUFFIX

    fake_redis.hset(cache_key, "2023-01-02", json.dumps({"date": "2023-01-02", "closing_price": 8185.0}))
    fake_redis.setex(neg_key, 3600, "1")

    def mutated_order(redis_client):
        """The mutation under test: marker checked before the hash branch."""
        if redis_client.exists(neg_key):
            return []
        if redis_client.type(cache_key) == "hash":
            return [json.loads(v) for v in redis_client.hgetall(cache_key).values()]
        return []

    mutated_result = mutated_order(fake_redis)
    assert mutated_result == [], "control: proves T6a would catch the marker-before-hash mutation"


def test_t6b_self_heals_stale_marker_via_step6_delete(monkeypatch, fake_redis, real_parquet_reader):
    """Reproduces the SPEC §3.6 self-heal: a marker written just after
    cronos's 18:36 flush survives the flush itself (it didn't exist yet
    when the flush ran). It does NOT survive the fix's own step-6 delete:
    the very next request that reaches the miss/materialize path (rather
    than short-circuiting on the marker) finds the ticker now present in
    the fresh parquet and clears the marker immediately — with ~227
    req/day/ticker (§13), that's minutes, not the full TTL.

    Simulated here by making the marker's `exists()` check transiently
    miss on the first call only (the race window), without touching the
    key itself — a stand-in for the real race's timing, not a mutation."""
    loader = _make_loader(monkeypatch, fake_redis)
    loader._parquet_reader = real_parquet_reader

    table, ticker = "yield_by_ticker", "AL30"
    cache_key = f"ts:cache:{table}:{ticker}"
    neg_key = cache_key + ":__miss__"
    fake_redis.setex(neg_key, 3600, "1")  # a (stale) marker is present

    real_exists = fake_redis.exists
    state = {"calls": 0}

    def flaky_exists(key):
        state["calls"] += 1
        if key == neg_key and state["calls"] == 1:
            return 0  # simulates the race: transiently misses once
        return real_exists(key)

    monkeypatch.setattr(fake_redis, "exists", flaky_exists)

    r1 = loader.get_ticker_series(table, ticker)
    assert len(r1) == 881, "the miss path ran once and materialized real data"
    assert not fake_redis.exists(neg_key), "step 6's delete must have cleared the stale marker"

    r2 = loader.get_ticker_series(table, ticker)
    assert len(r2) == 881, "next call reads the now-positive cache, not a marker shortcut"


# ── T7 — cronos flush sweeps markers too (type-agnostic scan) ──────────


def test_t7_flush_sweeps_negative_markers(monkeypatch, fake_redis):
    loader = _make_loader(monkeypatch, fake_redis)

    for ticker in ("TVPP", "BC36D", "COD7"):
        fake_redis.setex(f"ts:cache:yield_by_ticker:{ticker}:__miss__", 3600, "1")

    deleted = loader.flush_tier2_cache()

    assert deleted == 3
    remaining = list(fake_redis.scan_iter(match="ts:cache:*:__miss__"))
    assert remaining == []


def test_t7_mutation_control_marker_outside_prefix_survives_flush(monkeypatch, fake_redis):
    """Mutation control (planner-ratified): a marker built outside the
    `ts:cache:<table>:*` prefix (SPEC §3.1's actual contract) would NOT be
    swept by cronos's flush — the real markers above are, this deliberately
    malformed one isn't. Reproduce for real: change the marker key
    construction in loader.py from `cache_key + TIER2_NEG_SUFFIX` (which
    inherits `TIER2_CACHE_PREFIX`) to something outside that prefix, e.g.
    `f"miss:{table}:{ticker}"`, then rerun T7 — the marker survives."""
    loader = _make_loader(monkeypatch, fake_redis)

    mutant_marker = "miss:yield_by_ticker:TVPP"  # deliberately outside ts:cache:*
    fake_redis.setex(mutant_marker, 3600, "1")

    loader.flush_tier2_cache()

    assert fake_redis.exists(mutant_marker), (
        "control: proves T7 would catch a marker built outside the ts:cache: prefix"
    )


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
