"""
Capa 2 golden capture (DPM-395 §7.3). Manual, NOT run in CI — needs AWS creds
(to read v1 Parquet from S3) and a read-only DSN against the real prod dwh
Postgres. Produces the fixtures tests/test_golden.py needs:

    tests/golden/v1/{table}__{ticker}.json     -- v1 ParquetReader.read_ticker
    tests/golden/dwh_seed/{ticker}.sql         -- dwh rows for that ticker
    tests/golden/meta/{ticker}.json            -- ts:docta_tickers row snapshot

Usage:
    export AWS_PROFILE=...                      # or AWS_ACCESS_KEY_ID/SECRET
    export SMART_LOADER_GOLDEN_PG_DSN=postgresql://readonly@prod-host:5432/dwh
    export SMART_LOADER_GOLDEN_S3_BUCKET=docta-db-tables
    export SMART_LOADER_GOLDEN_REDIS_HOST=... SMART_LOADER_GOLDEN_REDIS_PORT=...
    python scripts/capture_golden.py

Witnesses (§7.3): hist_adj/hist_raw x {GGAL, NVDA (cedear split), 1 ticker
con alias LEGACY_SUPABASE, 1 stock ilíquido con huecos}. Extend WITNESSES
below with the real illiquid/alias tickers once identified against prod.
"""
import datetime as dt
import json
import os
import sys

CAP_DATE = dt.date(2026, 7, 3)  # eod_bar coverage cap per DPM-395 §0

TABLES = ["hist_adj", "hist_raw"]
WITNESSES = [
    "GGAL",
    "NVDA",
    # TODO(coder/planner): fill in the real alias witness (series_alias
    # source='LEGACY_SUPABASE') and the illiquid-with-gaps witness once
    # identified against prod dwh — do not fabricate ticker names.
]

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")


def _sort_and_cap(records):
    kept = [r for r in records if r.get("date") and r["date"] <= CAP_DATE.isoformat()]
    return sorted(kept, key=lambda r: r["date"])


def capture_v1(bucket: str):
    from smart_loader.parquet_reader import ParquetReader

    reader = ParquetReader(bucket, "v1")
    for table in TABLES:
        for ticker in WITNESSES:
            records = _sort_and_cap(reader.read_ticker(table, ticker))
            out_path = os.path.join(GOLDEN_DIR, "v1", f"{table}__{ticker}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2, sort_keys=True)
            print(f"wrote {out_path} ({len(records)} records)")


def dump_dwh_seed(dsn: str):
    import psycopg

    os.makedirs(os.path.join(GOLDEN_DIR, "dwh_seed"), exist_ok=True)
    with psycopg.connect(dsn) as conn:
        for ticker in WITNESSES:
            out_path = os.path.join(GOLDEN_DIR, "dwh_seed", f"{ticker}.sql")
            with conn.cursor() as cur:
                # Resolve series_id via direct symbol or alias (mirrors
                # pg_reader._RESOLVE_SQL, kinds relaxed since this is a dump).
                cur.execute(
                    """
                    SELECT s.series_id, s.instrument_id
                    FROM dwh.series s WHERE s.symbol = %(t)s
                    UNION
                    SELECT s.series_id, s.instrument_id
                    FROM dwh.series_alias a JOIN dwh.series s ON s.series_id = a.series_id
                    WHERE a.symbol = %(t)s
                    """,
                    {"t": ticker},
                )
                rows = cur.fetchall()
                if not rows:
                    print(f"WARNING: {ticker} did not resolve in prod dwh — skipping seed dump")
                    continue
                series_id, instrument_id = rows[0]

                statements = []
                cur.execute("SELECT * FROM dwh.instrument WHERE instrument_id = %s", (instrument_id,))
                statements.append(_insert_stmt("dwh.instrument", cur))
                cur.execute("SELECT * FROM dwh.series WHERE series_id = %s", (series_id,))
                statements.append(_insert_stmt("dwh.series", cur))
                cur.execute("SELECT * FROM dwh.series_alias WHERE series_id = %s", (series_id,))
                statements.append(_insert_stmt("dwh.series_alias", cur))
                cur.execute(
                    "SELECT * FROM dwh.eod_bar WHERE series_id = %s AND trade_date <= %s ORDER BY trade_date",
                    (series_id, CAP_DATE),
                )
                statements.append(_insert_stmt("dwh.eod_bar", cur))
                cur.execute(
                    """
                    SELECT * FROM dwh.fx_rate
                    WHERE base_currency='USD' AND quote_currency='ARS' AND rate_type='MEP'
                      AND rate_date <= %s
                    ORDER BY rate_date
                    """,
                    (CAP_DATE,),
                )
                statements.append(_insert_stmt("dwh.fx_rate", cur))

            with open(out_path, "w") as f:
                f.write("\n".join(s for s in statements if s))
            print(f"wrote {out_path}")


def _insert_stmt(table: str, cur) -> str:
    cols = [d.name for d in cur.description]
    lines = []
    for row in cur.fetchall():
        values = ", ".join(_sql_literal(v) for v in row)
        lines.append(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({values});")
    return "\n".join(lines)


def _sql_literal(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        return "'" + json.dumps(v).replace("'", "''") + "'::jsonb"
    return "'" + str(v).replace("'", "''") + "'"


def snapshot_meta(redis_host: str, redis_port: int):
    import pandas as pd
    import redis

    os.makedirs(os.path.join(GOLDEN_DIR, "meta"), exist_ok=True)
    client = redis.Redis(host=redis_host, port=redis_port, db=2, decode_responses=True)
    raw = client.get("ts:docta_tickers")
    if not raw:
        print("WARNING: ts:docta_tickers missing in Redis DB2 — no meta snapshot taken")
        return
    df = pd.DataFrame(json.loads(raw))
    for ticker in WITNESSES:
        row = df.loc[df.get("tickers", pd.Series(dtype=object)) == ticker]
        out_path = os.path.join(GOLDEN_DIR, "meta", f"{ticker}.json")
        with open(out_path, "w") as f:
            json.dump(
                row.iloc[0].to_dict() if not row.empty else None,
                f, indent=2, default=str,
            )
        print(f"wrote {out_path}")


def main():
    bucket = os.environ.get("SMART_LOADER_GOLDEN_S3_BUCKET", "docta-db-tables")
    dsn = os.environ.get("SMART_LOADER_GOLDEN_PG_DSN")
    redis_host = os.environ.get("SMART_LOADER_GOLDEN_REDIS_HOST")
    redis_port = int(os.environ.get("SMART_LOADER_GOLDEN_REDIS_PORT", 6379))

    if not dsn:
        print("SMART_LOADER_GOLDEN_PG_DSN not set — aborting (this script needs prod read-only creds)")
        sys.exit(1)

    capture_v1(bucket)
    dump_dwh_seed(dsn)
    if redis_host:
        snapshot_meta(redis_host, redis_port)
    else:
        print("SMART_LOADER_GOLDEN_REDIS_HOST not set — skipping meta snapshot")


if __name__ == "__main__":
    main()
