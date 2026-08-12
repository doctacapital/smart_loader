"""
DPM-815 P0-B — single-flight + memo with TTL purge + materialization
semaphore, all inside `ParquetReader._download_parquet` (SPEC §4, §11
T3/T10/T12/T14, plus the single-flight half of T11).

Anti-placebo (§11): T3, T10, T12, T14 must FAIL against pre-fix code
(pin `smart-loader @ v0.1.7`, byte-identical to main pre-DPM-815) — see
coder report for the actual run.
"""
import io
import threading
import time
import weakref

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from smart_loader.parquet_reader import ParquetReader


def _fake_df():
    return pd.DataFrame({"ticker": ["X"], "date": ["2023-01-02"]})


def _tiny_parquet_bytes() -> bytes:
    table = pa.table({"ticker": ["X"], "date": ["2023-01-02"]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


class _SlowGetObjectS3Stub:
    """Stubs the actual S3 boundary (`get_object`), not an internal helper —
    this call shape exists identically pre- and post-DPM-815, so it's a
    faithful anti-placebo probe of the choke-point's concurrency behaviour
    on either side of the fix."""

    class _Exceptions:
        NoSuchKey = Exception

    def __init__(self, data: bytes, delay: float = 0.2):
        self._data = data
        self._delay = delay
        self._lock = threading.Lock()
        self.exceptions = self._Exceptions()
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0

    def get_object(self, Bucket, Key):
        with self._lock:
            self.calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        time.sleep(self._delay)
        with self._lock:
            self.concurrent -= 1
        return {"Body": io.BytesIO(self._data)}

    def list_objects_v2(self, **kwargs):
        return {"CommonPrefixes": []}


# ── T3 — single-flight dedupes concurrent identical materializations ───


def test_t3_singleflight_dedupes_concurrent_downloads():
    reader = ParquetReader(bucket="unused")
    stub = _SlowGetObjectS3Stub(_tiny_parquet_bytes())
    reader._s3 = stub

    threads = [
        threading.Thread(target=lambda: reader._download_parquet("v1/same/key.parquet"))
        for _ in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert stub.calls == 1, "16 threads on the same key must materialize once"
    assert stub.max_concurrent == 1


# ── T10 — the memoized DataFrame is never mutated (I-7) ────────────────


def test_t10_memoized_dataframe_not_mutated(real_parquet_reader):
    from smart_loader import parquet_reader as pr_module

    r1 = real_parquet_reader.read_ticker("yield_by_ticker", "AL30")
    assert len(r1) == 881

    memo_key = f"{real_parquet_reader._bucket}/v1/yield_bonds/by_ticker.parquet"
    with pr_module._MEMO_LOCK:
        entry = pr_module._MEMO.get(memo_key)
    assert entry is not None
    assert len(entry[1]) == 288694, "memoized full table must stay full-sized after filtering AL30"

    r2 = real_parquet_reader.read_ticker("yield_by_ticker", "AL30")
    assert len(r2) == 881
    # NaN-tolerant comparison: some real records carry NaN fields
    # (current_closing_price), and NaN != NaN under plain equality.
    ids1 = sorted(r["id"] for r in r1)
    ids2 = sorted(r["id"] for r in r2)
    assert ids1 == ids2
    dates1 = sorted(r["date"] for r in r1)
    dates2 = sorted(r["date"] for r in r2)
    assert dates1 == dates2

    with pr_module._MEMO_LOCK:
        entry2 = pr_module._MEMO.get(memo_key)
    assert len(entry2[1]) == 288694
    assert real_parquet_reader._s3.get_object_calls == 1, "2nd read must be served from the memo"


# ── T11 (single-flight half) — kill switch ──────────────────────────────


def test_t11_singleflight_kill_switch_restores_pre_fix_behaviour(monkeypatch):
    monkeypatch.setenv("SMART_LOADER_SINGLEFLIGHT", "off")
    reader = ParquetReader(bucket="unused")
    stub = _SlowGetObjectS3Stub(_tiny_parquet_bytes(), delay=0.05)
    reader._s3 = stub

    threads = [
        threading.Thread(target=lambda: reader._download_parquet("v1/same/key.parquet"))
        for _ in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert stub.calls == 16, "kill-switch off must restore exact pre-fix behaviour (no dedup)"


# ── T12 — materialization semaphore caps true concurrency ──────────────


def test_t12_semaphore_caps_concurrent_materializations(monkeypatch):
    from smart_loader import parquet_reader as pr_module

    monkeypatch.setattr(pr_module, "_MATERIALIZE_SEM", threading.Semaphore(2))

    reader = ParquetReader(bucket="unused")
    lock = threading.Lock()
    state = {"concurrent": 0, "max_concurrent": 0}
    df = _fake_df()

    def stub(key):
        with lock:
            state["concurrent"] += 1
            state["max_concurrent"] = max(state["max_concurrent"], state["concurrent"])
        time.sleep(0.2)
        with lock:
            state["concurrent"] -= 1
        return df

    monkeypatch.setattr(reader, "_download_parquet_uncached", stub)

    keys = [f"v1/fake/key{i}.parquet" for i in range(8)]  # 8 distinct keys: no single-flight dedup
    threads = [threading.Thread(target=lambda k=k: reader._download_parquet(k)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max_concurrent"] == 2, "8 distinct materializations must never exceed the semaphore cap"


# ── T14 — memo purges by TTL, not just by LRU count (M3 fix) ───────────


def test_t14_memo_purges_stale_entries_by_ttl(monkeypatch, real_parquet_reader):
    from smart_loader import parquet_reader as pr_module

    monkeypatch.setenv("SMART_LOADER_MEMO_TTL_S", "0.05")

    real_parquet_reader.read_ticker("yield_by_ticker", "AL30")
    assert real_parquet_reader._s3.get_object_calls == 1

    memo_key = f"{real_parquet_reader._bucket}/v1/yield_bonds/by_ticker.parquet"
    with pr_module._MEMO_LOCK:
        old_df = pr_module._MEMO[memo_key][1]
    old_ref = weakref.ref(old_df)
    del old_df

    time.sleep(0.1)  # > MEMO_TTL_S

    real_parquet_reader.read_ticker("yield_by_ticker", "AL30")
    assert real_parquet_reader._s3.get_object_calls == 2, (
        "TTL expired: must re-materialize, not serve the stale memo entry"
    )

    with pr_module._MEMO_LOCK:
        assert len(pr_module._MEMO) == 1, "the stale entry must be replaced, not accumulated"

    import gc

    gc.collect()
    assert old_ref() is None, (
        "the purged DataFrame must be garbage-collectable — retention is bounded by TTL, "
        "not just by MEMO_MAX_ENTRIES (M3)"
    )
