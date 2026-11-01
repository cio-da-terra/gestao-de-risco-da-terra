"""Coffee prices.

- World Bank Pink Sheet (CC BY 4.0): monthly arabica/robusta. Automatable.
- CEPEA/ESALQ indicators: daily, best Brazilian reference, but downloads are
  manual and redistribution is restricted. We parse a file you drop in
  data/manual/cepea/ and never republish it raw.
- Conab producer prices / production costs: stubs (spreadsheet layouts
  change; write the parser once you have a sample file).

Needs openpyxl for .xlsx (pip install openpyxl). CEPEA serves old .xls,
which needs xlrd, or open+resave as .xlsx.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

from .. import db
from ..http import get

# The Pink Sheet URL changes occasionally; check
# https://www.worldbank.org/en/research/commodity-markets
PINK_SHEET_URL = ("https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-"
                  "0050012025/related/CMO-Historical-Data-Monthly.xlsx")
PINK_COLUMNS = {"Coffee, Arabica": "wb:coffee_arabica", "Coffee, Robusta": "wb:coffee_robusta"}


def fetch_pink_sheet(con, url: str = PINK_SHEET_URL) -> int:
    from openpyxl import load_workbook

    payload = get(url, timeout=180)
    snap = db.save_raw(con, "worldbank_pink", url, payload, ".xlsx")
    ws = load_workbook(BytesIO(payload), read_only=True, data_only=True)["Monthly Prices"]
    rows = list(ws.iter_rows(values_only=True))

    # Header row is the one containing the commodity names; layout has moved before.
    hdr_i = next(i for i, r in enumerate(rows) if r and any(c in PINK_COLUMNS for c in r if c))
    header = rows[hdr_i]
    cols = {header.index(name): sid for name, sid in PINK_COLUMNS.items()}
    for sid in cols.values():
        db.register_series(con, sid, "worldbank_pink", sid.split(":")[1], "USD/kg", "monthly")

    obs = []
    for r in rows[hdr_i + 1:]:
        m = re.fullmatch(r"(\d{4})M(\d{2})", str(r[0] or ""))
        if not m:
            continue
        d = date(int(m[1]), int(m[2]), 1)
        for i, sid in cols.items():
            v = r[i]
            obs.append(db.Obs(sid, d, float(v) if isinstance(v, (int, float)) else None))
    # Published ~2nd-3rd business day of the following month
    return db.upsert_observations(con, obs, timedelta(days=35), snap)


MANUAL_DIR = db.DATA_DIR / "manual" / "cepea"
CEPEA_FILES = {  # file stem in data/manual/cepea -> series id
    "arabica": "cepea:arabica_brl",
    "conilon": "cepea:conilon_brl",
}


def load_cepea_manual(con) -> int:
    """Parse CEPEA spreadsheets saved as data/manual/cepea/{arabica,conilon}.xlsx.

    Expected: a header row with 'Data' and 'À vista R$' (layout per CEPEA's
    export; adjust if it differs). Values are BRL per 60 kg bag.
    """
    from openpyxl import load_workbook

    total = 0
    for stem, sid in CEPEA_FILES.items():
        path = MANUAL_DIR / f"{stem}.xlsx"
        if not path.exists():
            continue
        payload = path.read_bytes()
        snap = db.save_raw(con, "cepea_manual", str(path), payload, ".xlsx")
        ws = load_workbook(BytesIO(payload), read_only=True, data_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        hdr_i = next(i for i, r in enumerate(rows) if r and r[0] and str(r[0]).strip() == "Data")
        price_col = next(i for i, c in enumerate(rows[hdr_i]) if c and "R$" in str(c))
        db.register_series(con, sid, "cepea", f"Indicador CEPEA/ESALQ café {stem}", "BRL/saca", "daily")
        obs = []
        for r in rows[hdr_i + 1:]:
            d = _parse_date(r[0])
            if d and isinstance(r[price_col], (int, float)):
                obs.append(db.Obs(sid, d, float(r[price_col])))
        total += db.upsert_observations(con, obs, timedelta(hours=20), snap)
    return total


def _parse_date(x) -> date | None:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    try:
        return datetime.strptime(str(x).strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def fetch_conab_producer_prices(con) -> int:
    """TODO: Conab 'Preços Agropecuários' (producer prices by UF). Grab a
    sample export, then parse like the Pink Sheet: find header, map columns."""
    raise NotImplementedError


def fetch_conab_costs(con) -> int:
    """TODO: Conab 'Custos de Produção' sheets for café arábica/conilon.
    Yearly; store total cost per saca per reference municipality as
    series 'conab:custo:{cultura}:{municipio}'. Needed for break-even."""
    raise NotImplementedError
