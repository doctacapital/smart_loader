"""
DPM-1388 — yield_by_date por fecha con candado entre procesos (SPEC v2 §1,
§6 Tests T1-T8, T13, T14).

Contexto: el 6-10 a las 19:01 los 4 workers de nexus reconstruyeron a la vez
el STRING `ts:cache:yield_by_date:__all__` (~215 MB cada uno) y el Redis de
marketdata desalojó 7 tablas fijas. Estos tests prueban lo que cambia en
smart_loader v0.1.9:

- un solo armado entre procesos (candado `lock:smart_loader:yield_by_date`);
- la curva como HASH por fecha, publicada completa o nada;
- todo lo que escribe vence.

Backends: cada test corre contra fakeredis y, si está instalado, contra un
`redis-server` real local (Lua y RENAME de verdad). T1 necesita procesos
distintos compartiendo un Redis (P7): usa `fakeredis.TcpFakeServer` y el
`redis-server` real.

Datos: parquet con la forma real de `v1/yield_bonds/by_date.parquet`
(columnas y tipos tomados de `tests/fixtures/by_ticker_real.parquet`, que
cronos escribe desde el mismo DataFrame: `cronos/services/s3_parquet_service.py:
247-270`) con tickers y montos inventados.
"""
import io
import json
import logging
import multiprocessing
import os
import shutil
import socket
import subprocess
import threading
import time
from datetime import date, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fakeredis
import pytest
import redis as redis_module

import smart_loader.loader as loader_module
from smart_loader.loader import (
    TIER1_DATAFRAME_KEYS,
    TIER1_DICT_KEYS,
    TIER1_PREFIX,
    YIELD_BY_DATE_BUILDING_PREFIX,
    YIELD_BY_DATE_KEY,
    YIELD_BY_DATE_LOCK_KEY,
    YIELD_BY_DATE_META_FIELD,
    SmartLoader,
    YieldCacheBuilding,
)
from smart_loader.parquet_reader import ParquetReader

REDIS_DB = 2  # SmartLoader usa la base 2 (loader.py, __init__)

# ── Datos sintéticos con la forma real de by_date.parquet ──────────────────

_FLOAT_COLS = [
    "closing_price", "opening_price", "low_price", "high_price",
    "effective_volume", "trade_volume", "tir", "tem", "tea", "tna",
    "current_closing_price", "duration", "accured_interest", "margen",
    "parity", "tna_30_360", "dtm", "residual_value", "technical_value",
    "clean_price", "dirty_price",
]
_SUBMARKETS = {
    "HARD_DOLLAR": ("USD", ["TSTHD1", "TSTHD2", "TSTHD3", "TSTHD4"]),
    "CER": ("ARS", ["TSTCE1", "TSTCE2", "TSTCE3"]),
    "ON": ("USD", ["TSTON1", "TSTON2", "TSTON3", "TSTON4", "TSTON5"]),
}
N_DATES = 120  # 3 tandas de 50 (50 + 50 + 20)
FIRST_DATE = date(2025, 1, 2)


def _dates(n=N_DATES):
    return [FIRST_DATE + timedelta(days=i) for i in range(n)]


def make_by_date_df(n_dates=N_DATES, date_as="date32") -> pd.DataFrame:
    rows = []
    rid = 1
    for i, d in enumerate(_dates(n_dates)):
        for sub, (currency, tickers) in _SUBMARKETS.items():
            for j, tk in enumerate(tickers):
                for specie in ("P", "D"):
                    base = 0.04 + 0.013 * j + 0.0002 * i
                    row = {
                        "id": rid,
                        "date": d if date_as == "date32" else d.isoformat(),
                        "ticker": tk,
                        "settlement_period": "24hs",
                        "submarket": sub,
                        "currency": currency,
                        "specie": specie,
                    }
                    for k, col in enumerate(_FLOAT_COLS):
                        row[col] = round(50.0 + 7.0 * j + 0.31 * i + 0.017 * k, 6)
                    row["tir"] = base
                    row["tem"] = base / 12
                    row["tea"] = base * 1.01
                    row["tna"] = base * 0.98
                    row["duration"] = 0.5 + 0.9 * j
                    row["dtm"] = 180.0 + 300.0 * j
                    # Como en el parquet real: columnas con nulos.
                    row["current_closing_price"] = float("nan")
                    if (i + j) % 3 == 0:
                        row["margen"] = float("nan")
                    rows.append(row)
                    rid += 1
    return pd.DataFrame(rows)


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    return buf.getvalue()


class _S3Exceptions:
    class NoSuchKey(Exception):
        pass


class ByDateS3Stub:
    """Sirve el parquet sintético en la key real de by_date."""

    def __init__(self, data: bytes):
        self._data = data
        self.exceptions = _S3Exceptions()

    def get_object(self, Bucket, Key):
        if Key == "v1/yield_bonds/by_date.parquet":
            return {"Body": io.BytesIO(self._data)}
        raise self.exceptions.NoSuchKey()

    def list_objects_v2(self, **kwargs):
        return {"CommonPrefixes": []}


class CountingReader:
    """Envuelve un ParquetReader real (S3 stubeado) y cuenta las lecturas de
    tabla entera. `delay` simula los 12,6 s medidos de lectura; `during_read`
    corre adentro de la lectura (para simular lo que pasa mientras tanto)."""

    def __init__(self, data: bytes, delay: float = 0.0, during_read=None, counter=None):
        self._reader = ParquetReader(bucket="unused-bucket", prefix="v1")
        self._reader._s3 = ByDateS3Stub(data)
        self._delay = delay
        self._during_read = during_read
        self._counter = counter  # multiprocessing.Value para T1
        self.calls = 0

    def read_full_table(self, table):
        self.calls += 1
        if self._counter is not None:
            with self._counter.get_lock():
                self._counter.value += 1
        if self._during_read is not None:
            self._during_read()
        if self._delay:
            time.sleep(self._delay)
        return self._reader.read_full_table(table)

    def read_ticker(self, table, ticker):
        return self._reader.read_ticker(table, ticker)

    def consume_degraded(self):
        return self._reader.consume_degraded()


class NeverReader:
    def read_full_table(self, table):
        raise AssertionError("no debía leer el parquet")


class EmptyReader:
    def __init__(self):
        self.calls = 0

    def read_full_table(self, table):
        self.calls += 1
        return {}


def canon(value) -> str:
    """Forma canónica para comparar: NaN != NaN en Python, pero en el JSON que
    viaja son el mismo 'NaN'."""
    return json.dumps(value, sort_keys=True, default=str)


@pytest.fixture(scope="module")
def synthetic_bytes():
    return parquet_bytes(make_by_date_df())


@pytest.fixture(scope="module")
def expected_by_date(synthetic_bytes):
    """Lo que ParquetReader arma desde el parquet: {fecha: {submarket: [...]}}."""
    reader = ParquetReader(bucket="unused-bucket", prefix="v1")
    reader._s3 = ByDateS3Stub(synthetic_bytes)
    data = reader.read_full_table("yield_by_date")
    assert len(data) == N_DATES
    return data


# ── Backends ──────────────────────────────────────────────────────────────
#
# - `backend` (un solo proceso): fakeredis en memoria y, si está instalado, un
#   redis-server real local. Cada loader tiene su propio cliente (conexión),
#   como dos procesos de nexus contra el mismo Redis.
# - `tcp_backend` (T1, varios procesos): fakeredis.TcpFakeServer y redis-server.
#   El TcpFakeServer de fakeredis 2.39 no aguanta respuestas grandes (su socket
#   no bloqueante tira BlockingIOError al mandar MB), así que solo se usa donde
#   hace falta compartir el Redis entre procesos.


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ping(port, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if redis_module.Redis(host="127.0.0.1", port=port).ping():
                return
        except redis_module.ConnectionError:
            time.sleep(0.05)
    raise RuntimeError(f"redis en :{port} no respondió")


def _start_redis_server():
    exe = shutil.which("redis-server")
    if exe is None:
        pytest.skip("redis-server no instalado: corre solo el backend fakeredis")
    port = _free_port()
    proc = subprocess.Popen(
        [exe, "--port", str(port), "--bind", "127.0.0.1", "--save", "",
         "--appendonly", "no"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _wait_ping(port)
    return proc, port


class Backend:
    def __init__(self, kind, fake_server=None, port=None):
        self.kind = kind
        self.fake_server = fake_server
        self.port = port

    def client(self):
        if self.fake_server is not None:
            return fakeredis.FakeRedis(
                server=self.fake_server, db=REDIS_DB, decode_responses=True
            )
        return redis_module.Redis(
            host="127.0.0.1", port=self.port, db=REDIS_DB, decode_responses=True
        )


@pytest.fixture(scope="module", params=["fakeredis", "redis_server"])
def backend(request):
    if request.param == "fakeredis":
        # Prod corre Redis 7.0.7 (INV DPM-1388 §1).
        yield Backend("fakeredis", fake_server=fakeredis.FakeServer(version=(7, 0)))
        return
    proc, port = _start_redis_server()
    try:
        yield Backend("redis_server", port=port)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module", params=["fakeredis_tcp", "redis_server"])
def tcp_backend(request):
    """(host, port) de un Redis compartible entre procesos (P7)."""
    if request.param == "fakeredis_tcp":
        from fakeredis import TcpFakeServer

        port = _free_port()
        server = TcpFakeServer(("127.0.0.1", port), server_type="redis", server_version=(7, 0))
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        _wait_ping(port)
        yield ("127.0.0.1", port)
        server.shutdown()
        server.server_close()
        return
    proc, port = _start_redis_server()
    try:
        yield ("127.0.0.1", port)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def client(backend):
    c = backend.client()
    c.flushall()
    yield c
    c.flushall()


@pytest.fixture
def make_loader(backend):
    def _make(reader):
        loader = SmartLoader(s3_bucket="unused-bucket")
        loader._redis = backend.client()
        loader._parquet_reader = reader
        return loader

    return _make


def _target_date() -> str:
    return (FIRST_DATE + timedelta(days=77)).isoformat()


TIER1_EXEMPT = {f"{TIER1_PREFIX}{k}" for k in TIER1_DATAFRAME_KEYS} | {
    f"{TIER1_PREFIX}{suffix}" for _db_key, suffix in TIER1_DICT_KEYS
}
assert f"{TIER1_PREFIX}cash_flows_adj" in TIER1_EXEMPT


def assert_everything_expires(client):
    """T8: toda clave de la base, salvo las tablas fijas y ts:cash_flows_adj,
    tiene vencimiento > 0."""
    offenders = []
    for key in client.scan_iter(match="*", count=1000):
        if key in TIER1_EXEMPT:
            continue
        ttl = client.ttl(key)
        if ttl is None or ttl <= 0:
            offenders.append((key, ttl))
    assert not offenders, f"claves sin vencimiento: {offenders}"


def _seed_tier1(client):
    """Tablas fijas sin vencimiento, como las deja cronos fase A."""
    client.set(f"{TIER1_PREFIX}bonds", json.dumps([{"ticker": "TSTHD1"}]))
    client.set(f"{TIER1_PREFIX}cash_flows_adj", json.dumps({"TSTHD1": []}))
    client.set(f"{TIER1_PREFIX}tickers_laws", json.dumps({"1": "NY"}))


# ── Constantes del contrato ───────────────────────────────────────────────


def test_contract_constants():
    assert YIELD_BY_DATE_KEY == "ts:cache:yield_by_date:__by_date__"
    assert YIELD_BY_DATE_BUILDING_PREFIX == "ts:cache:yield_by_date:__building__:"
    assert YIELD_BY_DATE_LOCK_KEY == "lock:smart_loader:yield_by_date"
    assert YIELD_BY_DATE_META_FIELD == "__meta__"
    assert loader_module.YIELD_BY_DATE_TTL == 86400
    assert loader_module.YIELD_BY_DATE_BUILDING_TTL == 300
    assert loader_module.YIELD_BY_DATE_LOCK_TTL == 120
    assert loader_module.YIELD_BY_DATE_BATCH_DATES == 50
    assert loader_module.YIELD_BY_DATE_WAIT_S == 25.0
    assert loader_module.YIELD_BY_DATE_POLL_S == 0.5


# ── T1: entre procesos, una sola lectura y una sola publicación ───────────


def _t1_child(host, port, data, target, barrier, reads, publishes, out):
    loader = SmartLoader(redis_host=host, redis_port=port, s3_bucket="unused-bucket")
    loader._parquet_reader = CountingReader(data, delay=1.5, counter=reads)

    orig_publish = loader._publish_yield_by_date

    def counting_publish(d, token):
        with publishes.get_lock():
            publishes.value += 1
        return orig_publish(d, token)

    loader._publish_yield_by_date = counting_publish
    barrier.wait()
    result = loader.get_yield_for_date(target)
    out.put(canon(result))


def test_t1_four_processes_one_read_one_publish(tcp_backend, synthetic_bytes, expected_by_date):
    host, port = tcp_backend
    client = redis_module.Redis(host=host, port=port, db=REDIS_DB, decode_responses=True)
    client.flushall()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(4)
    reads = ctx.Value("i", 0)
    publishes = ctx.Value("i", 0)
    out = ctx.Queue()
    target = _target_date()

    procs = [
        ctx.Process(
            target=_t1_child,
            args=(host, port, synthetic_bytes, target, barrier, reads, publishes, out),
        )
        for _ in range(4)
    ]
    for p in procs:
        p.start()
    results = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    assert reads.value == 1, "una sola lectura del parquet entre los 4 procesos"
    assert publishes.value == 1, "una sola publicación entre los 4 procesos"
    assert results == [canon(expected_by_date[target])] * 4
    assert client.hlen(YIELD_BY_DATE_KEY) == N_DATES + 1
    assert not client.exists(YIELD_BY_DATE_LOCK_KEY)
    assert_everything_expires(client)


def test_t1_double_check_after_winning_the_lock(make_loader, client, synthetic_bytes, expected_by_date):
    """P1: vio la clave ausente, otro publicó y soltó el candado, y recién
    ahí lo ganó. Vuelve a mirar y no arma de nuevo."""
    first = make_loader(CountingReader(synthetic_bytes))
    second_reader = CountingReader(synthetic_bytes)
    second = make_loader(second_reader)
    target = _target_date()

    orig_acquire = second._try_acquire_yield_lock

    def acquire_after_other_published(token):
        assert first.get_yield_for_date(target) is not None  # arma y suelta
        assert not client.exists(YIELD_BY_DATE_LOCK_KEY)
        return orig_acquire(token)

    second._try_acquire_yield_lock = acquire_after_other_published

    result = second.get_yield_for_date(target)

    assert canon(result) == canon(expected_by_date[target])
    assert second_reader.calls == 0, "el doble chequeo evita la segunda lectura"
    assert not client.exists(YIELD_BY_DATE_LOCK_KEY)


# ── T2: el que pierde el candado ──────────────────────────────────────────


def test_t2_loser_returns_as_soon_as_key_appears(make_loader, client, synthetic_bytes, expected_by_date):
    client.set(YIELD_BY_DATE_LOCK_KEY, "other-process", ex=120)
    waiter_reader = CountingReader(synthetic_bytes)
    waiter = make_loader(waiter_reader)
    builder = make_loader(CountingReader(synthetic_bytes))
    target = _target_date()

    box = {}

    def run():
        box["result"] = waiter.get_yield_for_date(target)
        box["done_at"] = time.monotonic()

    t = threading.Thread(target=run)
    t.start()
    time.sleep(1.2)
    assert t.is_alive(), "sin tabla y con el candado ajeno, espera"
    # El "otro proceso" publica y suelta su candado.
    assert builder._publish_yield_by_date(expected_by_date, "other-process") is True
    published_at = time.monotonic()
    builder._release_yield_lock("other-process")
    t.join(timeout=5)

    assert not t.is_alive()
    assert canon(box["result"]) == canon(expected_by_date[target])
    assert box["done_at"] - published_at < loader_module.YIELD_BY_DATE_POLL_S + 0.3
    assert waiter_reader.calls == 0


class FakeTime:
    """Reloj falso para el módulo loader: `sleep` avanza el tiempo y corre
    las acciones agendadas."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []
        self.actions = []  # (at_elapsed, fn)
        self.start = self.now

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s
        elapsed = self.now - self.start
        due = [a for a in self.actions if a[0] <= elapsed + 1e-9]
        for a in due:
            self.actions.remove(a)
            a[1]()


def test_t2_no_key_after_25s_raises(monkeypatch, make_loader, client, synthetic_bytes):
    client.set(YIELD_BY_DATE_LOCK_KEY, "other-process", ex=120)
    waiter = make_loader(NeverReader())
    fake = FakeTime()
    monkeypatch.setattr(loader_module, "time", fake)

    hmgets = []
    orig_hmget = waiter._redis.hmget

    def counting_hmget(*a, **kw):
        hmgets.append(fake.now - fake.start)
        return orig_hmget(*a, **kw)

    monkeypatch.setattr(waiter._redis, "hmget", counting_hmget)

    with pytest.raises(YieldCacheBuilding):
        waiter.get_yield_for_date(_target_date())

    assert fake.now - fake.start == pytest.approx(25.0)
    assert all(s <= 0.5 for s in fake.sleeps)
    assert len(fake.sleeps) == 50, "mira cada 0,5 s durante 25 s"
    assert len(hmgets) == 1 + 50, "el HMGET inicial + uno por vuelta"
    assert client.get(YIELD_BY_DATE_LOCK_KEY) == "other-process"


def test_t2_lock_vanishes_without_key_takes_it_and_builds(make_loader, client, synthetic_bytes, expected_by_date):
    # El que armaba murió: su candado vence solo a los 0,7 s.
    client.set(YIELD_BY_DATE_LOCK_KEY, "dead-builder", px=700)
    reader = CountingReader(synthetic_bytes)
    waiter = make_loader(reader)
    target = _target_date()

    t0 = time.monotonic()
    result = waiter.get_yield_for_date(target)

    assert canon(result) == canon(expected_by_date[target])
    assert reader.calls == 1
    assert time.monotonic() - t0 < 5
    assert client.hlen(YIELD_BY_DATE_KEY) == N_DATES + 1
    assert not client.exists(YIELD_BY_DATE_LOCK_KEY)


class ScriptedRedis:
    """Proxy del cliente real que cuenta los SET NX y deja intercalar
    acciones justo después de un EXISTS (para forzar carreras)."""

    def __init__(self, real, after_exists=None):
        self._real = real
        self._after_exists = after_exists
        self.nx_attempts = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def set(self, *a, **kw):
        if kw.get("nx"):
            self.nx_attempts += 1
        return self._real.set(*a, **kw)

    def exists(self, *keys):
        r = self._real.exists(*keys)
        if self._after_exists is not None:
            self._after_exists(r)
        return r


def test_t2_lock_vanishes_twice_only_one_takeover_attempt(monkeypatch, make_loader, client):
    """"Intenta tomarlo UNA vez": si otro gana esa carrera, sigue esperando,
    y una segunda desaparición del candado no dispara otro intento."""
    client.set(YIELD_BY_DATE_LOCK_KEY, "other-1", ex=120)
    waiter = make_loader(NeverReader())
    fake = FakeTime()
    monkeypatch.setattr(loader_module, "time", fake)

    state = {"raced": False}

    def other_wins_race(exists_result):
        if exists_result == 0 and not state["raced"]:
            state["raced"] = True
            client.set(YIELD_BY_DATE_LOCK_KEY, "other-2", ex=120)

    proxy = ScriptedRedis(waiter._redis, after_exists=other_wins_race)
    waiter._redis = proxy
    fake.actions.append((2.0, lambda: client.delete(YIELD_BY_DATE_LOCK_KEY)))
    fake.actions.append((5.0, lambda: client.delete(YIELD_BY_DATE_LOCK_KEY)))

    with pytest.raises(YieldCacheBuilding):
        waiter.get_yield_for_date(_target_date())

    assert state["raced"]
    assert proxy.nx_attempts == 2, "el intento inicial + un único reintento"
    assert not client.exists(YIELD_BY_DATE_KEY)


# ── T3: tabla publicada sin la fecha → None sin leer el parquet ───────────


def test_t3_published_without_the_date_returns_none_without_reading(make_loader, client, synthetic_bytes):
    builder = make_loader(CountingReader(synthetic_bytes))
    assert builder.get_yield_for_date(_target_date()) is not None

    proxy_target = make_loader(NeverReader())
    proxy = ScriptedRedis(proxy_target._redis)
    proxy_target._redis = proxy

    assert proxy_target.get_yield_for_date("2024-06-03") is None
    assert proxy.nx_attempts == 0, "ni intenta tomar el candado"


# ── T4: un lector concurrente nunca ve un hash parcial ────────────────────


def test_t4_concurrent_reader_never_sees_partial_hash(make_loader, backend, client, synthetic_bytes, expected_by_date):
    builder = make_loader(CountingReader(synthetic_bytes))
    observer = backend.client()
    some_dates = sorted(expected_by_date)[::7]
    anomalies = []
    observations = {"published": 0, "absent": 0}
    stop = threading.Event()

    def observe_once():
        pipe = observer.pipeline(transaction=False)
        pipe.hlen(YIELD_BY_DATE_KEY)
        pipe.hmget(YIELD_BY_DATE_KEY, *some_dates, YIELD_BY_DATE_META_FIELD)
        hlen, values = pipe.execute()
        if hlen == 0:
            observations["absent"] += 1
            return
        observations["published"] += 1
        if hlen != N_DATES + 1:
            anomalies.append(("hlen", hlen))
        if values[-1] is None or any(v is None for v in values[:-1]):
            anomalies.append(("missing", values.count(None)))

    def hammer():
        while not stop.is_set():
            observe_once()

    orig_batch = builder._write_yield_building_batch

    def batch_then_observe(key, mapping):
        orig_batch(key, mapping)
        observe_once()
        assert observer.hlen(YIELD_BY_DATE_KEY) == 0, "nada publicado a mitad del armado"
        time.sleep(0.05)

    builder._write_yield_building_batch = batch_then_observe

    t = threading.Thread(target=hammer)
    t.start()
    try:
        builder.get_yield_for_date(_target_date())
        time.sleep(0.1)
    finally:
        stop.set()
        t.join(timeout=5)

    assert not anomalies, anomalies
    assert observations["absent"] > 0 and observations["published"] > 0


# ── T5: vencimientos de la temporal y de la final ─────────────────────────


def test_t5_building_ttl_after_each_batch_and_final_ttl(make_loader, client, synthetic_bytes):
    builder = make_loader(CountingReader(synthetic_bytes))
    seen = []
    orig_batch = builder._write_yield_building_batch

    def batch_then_check(key, mapping):
        orig_batch(key, mapping)
        seen.append((key, client.ttl(key), client.hlen(key)))

    builder._write_yield_building_batch = batch_then_check
    builder.get_yield_for_date(_target_date())

    assert [h for _k, _ttl, h in seen] == [50, 100, 121], "3 tandas, __meta__ en la última"
    for key, ttl, _h in seen:
        assert key.startswith(YIELD_BY_DATE_BUILDING_PREFIX)
        assert 0 < ttl <= 300
    assert not client.exists(seen[0][0]), "la temporal pasó a ser la final"
    assert 86400 - 10 < client.ttl(YIELD_BY_DATE_KEY) <= 86400

    meta = json.loads(client.hget(YIELD_BY_DATE_KEY, YIELD_BY_DATE_META_FIELD))
    assert meta["dates"] == N_DATES
    assert meta["max_date"] == (FIRST_DATE + timedelta(days=N_DATES - 1)).isoformat()
    rows_per_date = sum(len(tks) for _c, tks in _SUBMARKETS.values()) * 2
    assert meta["rows"] == N_DATES * rows_per_date
    assert "built_at" in meta


# ── T6: el borrado de cronos se lleva final y temporal, no el candado ─────


def _cronos_flush_yield_by_date(client):
    """Mismo patrón que cronos fase B: `ts:cache:` + `yield_by_date:*`
    (cronos/tier2_publish.py:148-150 → services/s3_parquet_service.py:138-155,
    tabla yield_bonds → ["yield_by_date", ...] en :768-771)."""
    for key in client.scan_iter(match="ts:cache:yield_by_date:*", count=1000):
        client.delete(key)


def test_t6_cronos_flush_takes_final_and_building_not_the_lock(make_loader, client, synthetic_bytes):
    builder = make_loader(CountingReader(synthetic_bytes))
    builder.get_yield_for_date(_target_date())
    building_key = f"{YIELD_BY_DATE_BUILDING_PREFIX}in-progress"
    builder._write_yield_building_batch(building_key, {"2025-01-02": "{}"})
    client.set(YIELD_BY_DATE_LOCK_KEY, "in-progress", ex=120)
    assert client.exists(YIELD_BY_DATE_KEY) and client.exists(building_key)

    _cronos_flush_yield_by_date(client)

    assert not client.exists(YIELD_BY_DATE_KEY)
    assert not client.exists(building_key)
    assert client.get(YIELD_BY_DATE_LOCK_KEY) == "in-progress"

    # El flush propio de smart_loader (ts:cache:*) tampoco lo toca.
    builder._write_yield_building_batch(building_key, {"2025-01-02": "{}"})
    builder.flush_tier2_cache()
    assert not client.exists(building_key)
    assert client.get(YIELD_BY_DATE_LOCK_KEY) == "in-progress"


# ── T7: misma respuesta que la rama vieja, fecha por fecha ────────────────


@pytest.mark.parametrize("date_as", ["date32", "string"])
def test_t7_same_answer_as_get_market_series_for_every_date(make_loader, client, date_as):
    data = parquet_bytes(make_by_date_df(date_as=date_as))
    new = make_loader(CountingReader(data))
    old = make_loader(CountingReader(data))

    old_all = old.get_market_series("yield_by_date", "bond")
    assert len(old_all) == N_DATES
    for d in sorted(old_all):
        assert canon(new.get_yield_for_date(d)) == canon(old_all[d]), d
    # Fecha ausente: las dos ramas dicen "no hay".
    assert new.get_yield_for_date("2024-06-03") is None
    assert old_all.get("2024-06-03") is None
    # Con la clave vieja ya cacheada, la rama vieja sigue igual.
    old_cached = old.get_market_series("yield_by_date", "bond")
    for d in sorted(old_cached)[::11]:
        assert canon(new.get_yield_for_date(d)) == canon(old_cached[d])


# ── T8: después de cada operación, todo vence (salvo tablas fijas) ────────


def test_t8_everything_expires_after_each_public_operation(monkeypatch, make_loader, client, synthetic_bytes):
    _seed_tier1(client)
    reader = CountingReader(synthetic_bytes)
    loader = make_loader(reader)
    target = _target_date()

    loader.get_yield_for_date(target)                    # arma
    assert_everything_expires(client)
    loader.get_yield_for_date("2025-01-02")              # hit
    assert_everything_expires(client)
    assert loader.get_yield_for_date("2024-06-03") is None  # fecha ausente
    assert_everything_expires(client)

    _cronos_flush_yield_by_date(client)                  # borrado diario
    assert_everything_expires(client)

    # Armado que falla a mitad (T13) → nada sin vencimiento.
    orig_batch = loader._write_yield_building_batch
    calls = {"n": 0}

    def batch_then_delete(key, mapping):
        orig_batch(key, mapping)
        calls["n"] += 1
        if calls["n"] == 1:
            client.delete(key)

    loader._write_yield_building_batch = batch_then_delete
    loader.get_yield_for_date(target)
    assert_everything_expires(client)
    loader._write_yield_building_batch = orig_batch

    # Espera agotada con candado ajeno.
    client.set(YIELD_BY_DATE_LOCK_KEY, "other-process", ex=120)
    monkeypatch.setattr(loader_module, "YIELD_BY_DATE_WAIT_S", 0.3)
    monkeypatch.setattr(loader_module, "YIELD_BY_DATE_POLL_S", 0.1)
    with pytest.raises(YieldCacheBuilding):
        loader.get_yield_for_date(target)
    assert_everything_expires(client)
    client.delete(YIELD_BY_DATE_LOCK_KEY)

    # La rama vieja también vence.
    loader.get_market_series("yield_by_date", "bond")
    assert_everything_expires(client)

    # Rearmado normal.
    loader.get_yield_for_date(target)
    assert_everything_expires(client)
    assert client.hlen(YIELD_BY_DATE_KEY) == N_DATES + 1


# ── T13: la temporal desaparece entre dos tandas ──────────────────────────


def test_t13_building_key_deleted_between_batches_publishes_nothing(caplog, make_loader, client, synthetic_bytes, expected_by_date):
    builder = make_loader(CountingReader(synthetic_bytes))
    orig_batch = builder._write_yield_building_batch
    calls = {"n": 0}
    ttls = []

    def batch_then_delete(key, mapping):
        orig_batch(key, mapping)
        ttls.append(client.ttl(key))
        calls["n"] += 1
        if calls["n"] == 1:
            client.delete(key)  # cronos la borró o Redis la desalojó

    builder._write_yield_building_batch = batch_then_delete
    target = _target_date()

    with caplog.at_level(logging.WARNING, logger="smart_loader.loader"):
        result = builder.get_yield_for_date(target)

    assert calls["n"] == 3
    # C2.1: la tanda que recrea la temporal también le pone vencimiento.
    assert all(0 < t <= 300 for t in ttls), ttls
    assert canon(result) == canon(expected_by_date[target]), "contesta desde memoria"
    assert not client.exists(YIELD_BY_DATE_KEY), "no publica un hash parcial"
    assert list(client.scan_iter(match=f"{YIELD_BY_DATE_BUILDING_PREFIX}*")) == []
    assert not client.exists(YIELD_BY_DATE_LOCK_KEY)
    assert_everything_expires(client)
    assert any("publish_incomplete" in r.getMessage() for r in caplog.records)


def test_redis_rejects_writes_answers_from_memory(make_loader, client, synthetic_bytes, expected_by_date):
    """Con volatile-lru un Redis lleno rechaza escrituras ("OOM command not
    allowed"). El pedido se contesta igual y no queda nada a medias."""
    builder = make_loader(CountingReader(synthetic_bytes))
    orig_batch = builder._write_yield_building_batch
    calls = {"n": 0}

    def oom_on_second(key, mapping):
        calls["n"] += 1
        if calls["n"] == 2:
            raise redis_module.ResponseError(
                "OOM command not allowed when used memory > 'maxmemory'."
            )
        orig_batch(key, mapping)

    builder._write_yield_building_batch = oom_on_second
    target = _target_date()

    result = builder.get_yield_for_date(target)

    assert canon(result) == canon(expected_by_date[target])
    assert not client.exists(YIELD_BY_DATE_KEY)
    assert list(client.scan_iter(match=f"{YIELD_BY_DATE_BUILDING_PREFIX}*")) == []
    assert not client.exists(YIELD_BY_DATE_LOCK_KEY)


def test_empty_read_writes_nothing_and_releases_the_lock(make_loader, client):
    reader = EmptyReader()
    loader = make_loader(reader)

    assert loader.get_yield_for_date(_target_date()) is None
    assert reader.calls == 1
    assert list(client.scan_iter(match="*")) == []


# ── T14: no borra un candado ajeno ────────────────────────────────────────


def test_t14_expired_lock_taken_by_other_is_not_deleted(make_loader, client, synthetic_bytes):
    def lock_expires_and_other_takes_it():
        # Nuestro candado venció (120 s) y otro proceso lo tomó.
        client.set(YIELD_BY_DATE_LOCK_KEY, "other-process", ex=120)

    builder = make_loader(CountingReader(synthetic_bytes, during_read=lock_expires_and_other_takes_it))
    builder.get_yield_for_date(_target_date())

    assert client.get(YIELD_BY_DATE_LOCK_KEY) == "other-process"


def test_t14_release_is_compare_and_delete(make_loader, client):
    loader = make_loader(NeverReader())
    client.set(YIELD_BY_DATE_LOCK_KEY, "theirs", ex=120)
    loader._release_yield_lock("mine")
    assert client.get(YIELD_BY_DATE_LOCK_KEY) == "theirs"
    loader._release_yield_lock("theirs")
    assert not client.exists(YIELD_BY_DATE_LOCK_KEY)
