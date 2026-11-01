"""SQLite storage with point-in-time semantics.

Every value is stored with two dates:
  - ref_date:     the period the value describes (e.g. the price of 2025-03-14)
  - available_at: when that value could first have been known

Revisions never overwrite: a changed value is inserted as a new row with a
later available_at. `as_of()` then answers "what did we know on date D?",
which is what keeps backtests (and the conformal calibration) honest.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "cio_da_terra.sqlite"
RAW_DIR = DATA_DIR / "raw"

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    series_id   TEXT PRIMARY KEY,          -- e.g. 'bcb:ptax_venda'
    source      TEXT NOT NULL,             -- e.g. 'bcb_sgs'
    description TEXT,
    unit        TEXT,
    frequency   TEXT                       -- daily | monthly | yearly ...
);

CREATE TABLE IF NOT EXISTS observation (
    series_id    TEXT NOT NULL REFERENCES series(series_id),
    ref_date     TEXT NOT NULL,            -- ISO date of the period described
    value        REAL,
    available_at TEXT NOT NULL,            -- ISO datetime first knowable
    fetched_at   TEXT NOT NULL,            -- ISO datetime we fetched it
    snapshot_id  INTEGER REFERENCES raw_snapshot(snapshot_id),
    PRIMARY KEY (series_id, ref_date, available_at)
);

CREATE TABLE IF NOT EXISTS raw_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    url         TEXT,
    fetched_at  TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    path        TEXT NOT NULL              -- file under data/raw/
);

CREATE TABLE IF NOT EXISTS fetch_log (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    job        TEXT NOT NULL,
    started_at TEXT NOT NULL,
    status     TEXT NOT NULL,              -- ok | error
    rows_new   INTEGER,
    message    TEXT
);

-- Latest known value per (series, ref_date): what the app shows today.
CREATE VIEW IF NOT EXISTS observation_latest AS
SELECT o.* FROM observation o
JOIN (
    SELECT series_id, ref_date, MAX(available_at) AS available_at
    FROM observation GROUP BY series_id, ref_date
) m USING (series_id, ref_date, available_at);
"""


@dataclass(frozen=True)
class Obs:
    series_id: str
    ref_date: date
    value: float | None


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def register_series(con, series_id, source, description="", unit="", frequency=""):
    con.execute(
        """INSERT INTO series VALUES (?,?,?,?,?)
           ON CONFLICT(series_id) DO UPDATE SET
             description=excluded.description, unit=excluded.unit,
             frequency=excluded.frequency""",
        (series_id, source, description, unit, frequency),
    )


def save_raw(con, source: str, url: str, payload: bytes, suffix: str) -> int:
    """Keep the untouched download so any parsed value can be traced back."""
    fetched = now_utc()
    digest = hashlib.sha256(payload).hexdigest()
    folder = RAW_DIR / source / fetched.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{fetched.strftime('%H%M%S')}_{digest[:10]}{suffix}"
    path.write_bytes(payload)
    cur = con.execute(
        "INSERT INTO raw_snapshot(source,url,fetched_at,sha256,path) VALUES (?,?,?,?,?)",
        (source, url, fetched.isoformat(), digest, str(path.relative_to(ROOT))),
    )
    return cur.lastrowid


def upsert_observations(
    con,
    rows: Iterable[Obs],
    publication_lag: timedelta,
    snapshot_id: int | None = None,
) -> int:
    """Insert new points and revisions; skip values we already know.

    available_at rule:
      - first time we see a point  -> min(fetched_at, ref_date + publication_lag)
        (so a historical backfill does not pretend the data appeared today)
      - a revised value            -> fetched_at (we only learned it now)
    """
    fetched = now_utc()
    new = 0
    for r in rows:
        prev = con.execute(
            """SELECT value FROM observation WHERE series_id=? AND ref_date=?
               ORDER BY available_at DESC LIMIT 1""",
            (r.series_id, r.ref_date.isoformat()),
        ).fetchone()
        if prev is not None and _same(prev[0], r.value):
            continue
        if prev is None:
            est = datetime.combine(r.ref_date, datetime.min.time(), timezone.utc) + publication_lag
            available = min(fetched, est)
        else:
            available = fetched
        con.execute(
            "INSERT OR IGNORE INTO observation VALUES (?,?,?,?,?,?)",
            (r.series_id, r.ref_date.isoformat(), r.value,
             available.isoformat(), fetched.isoformat(), snapshot_id),
        )
        new += 1
    con.commit()
    return new


def _same(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * max(1.0, abs(a))


def as_of(con, series_id: str, when: datetime) -> list[tuple[str, float]]:
    """The series exactly as it was knowable at `when` (for backtests)."""
    return con.execute(
        """SELECT ref_date, value FROM (
               SELECT ref_date, value,
                      ROW_NUMBER() OVER (PARTITION BY ref_date
                                         ORDER BY available_at DESC) AS rn
               FROM observation
               WHERE series_id=? AND available_at<=?
           ) WHERE rn=1 ORDER BY ref_date""",
        (series_id, when.isoformat()),
    ).fetchall()


def last_ref_date(con, series_id: str) -> date | None:
    row = con.execute(
        "SELECT MAX(ref_date) FROM observation WHERE series_id=?", (series_id,)
    ).fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None
