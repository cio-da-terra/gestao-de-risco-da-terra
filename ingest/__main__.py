"""Run ingestion jobs.

    python -m ingest daily      # BCB (FX, Selic), NASA POWER weather
    python -m ingest monthly    # + IPCA, ONI, Pink Sheet, CEPEA manual files
    python -m ingest yearly     # + IBGE PAM yield
    python -m ingest daily --only bcb

Each source runs independently: one failing source is logged and the rest
continue, so a broken spreadsheet layout never blocks the daily prices.
"""

from __future__ import annotations

import argparse
import traceback

from . import db
from .sources import bcb, climate, ibge, prices

# Example locations. Replace with the municipalities you pick in week 1.
# Coordinates are approximate town centres.
LOCATIONS = [
    {"name": "patrocinio_mg", "municipio": "Patrocínio", "uf": "MG", "lat": -18.94, "lon": -46.99},
    {"name": "jaguare_es", "municipio": "Jaguaré", "uf": "ES", "lat": -18.91, "lon": -40.08},
]


def _power(con):
    return sum(climate.fetch_nasa_power(con, l["name"], l["lat"], l["lon"]) for l in LOCATIONS)


def _pam(con):
    codes = [ibge.municipio_code(l["municipio"], l["uf"]) for l in LOCATIONS]
    return ibge.fetch_pam(con, codes)


JOBS = {
    "daily": {"bcb": bcb.fetch, "power": _power},
    "monthly": {"oni": climate.fetch_oni, "pink_sheet": prices.fetch_pink_sheet,
                "cepea_manual": prices.load_cepea_manual},
    "yearly": {"pam": _pam},
}
CADENCE = ["daily", "monthly", "yearly"]  # a longer cadence also runs the shorter ones


def main():
    ap = argparse.ArgumentParser(prog="python -m ingest")
    ap.add_argument("cadence", choices=CADENCE)
    ap.add_argument("--only", nargs="*", help="run only these sources")
    args = ap.parse_args()

    tasks = {}
    for c in CADENCE[: CADENCE.index(args.cadence) + 1]:
        tasks.update(JOBS[c])
    if args.only:
        tasks = {k: v for k, v in tasks.items() if k in args.only}

    con = db.connect()
    failed = False
    for name, fn in tasks.items():
        started = db.now_utc().isoformat()
        try:
            n = fn(con)
            con.execute("INSERT INTO fetch_log(job,started_at,status,rows_new) VALUES (?,?,?,?)",
                        (name, started, "ok", n))
            print(f"[ok]    {name}: {n} new rows")
        except Exception as e:  # keep going; log and report at the end
            failed = True
            con.execute("INSERT INTO fetch_log(job,started_at,status,message) VALUES (?,?,?,?)",
                        (name, started, "error", f"{e!r}\n{traceback.format_exc()[-1500:]}"))
            print(f"[error] {name}: {e!r}")
        con.commit()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
