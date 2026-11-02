"""Banco Central SGS: FX, Selic, IPCA.

API: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json
Daily series are capped at 10 years per request, so we fetch in chunks.
Codes below are from memory; confirm at https://www3.bcb.gov.br/sgspub
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta

from .. import db
from ..http import get

SERIES = {
    # series_id           code  description                      unit   freq     lag
    "bcb:ptax_venda":    (1,   "USD/BRL PTAX, venda",            "BRL", "daily",   timedelta(hours=18)),
    "bcb:selic_meta":    (432, "Meta Selic",                     "% a.a.", "daily", timedelta(days=0)),
    "bcb:ipca_mensal":   (433, "IPCA, variação mensal",          "%",   "monthly", timedelta(days=40)),
}

CHUNK_YEARS = 9


def _fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def fetch(con, start: date | None = None, end: date | None = None) -> int:
    end = end or date.today()
    total = 0
    for sid, (code, desc, unit, freq, lag) in SERIES.items():
        db.register_series(con, sid, "bcb_sgs", desc, unit, freq)
        s = start or _resume_from(con, sid) or date(2000, 1, 1)
        while s <= end:
            e = min(end, date(s.year + CHUNK_YEARS, 12, 31))
            url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
                   f"?formato=json&dataInicial={_fmt(s)}&dataFinal={_fmt(e)}")
            payload, data = _get_json_retry(url)
            snap = db.save_raw(con, "bcb_sgs", url, payload, ".json")
            rows = [
                db.Obs(sid, datetime.strptime(r["data"], "%d/%m/%Y").date(),
                       float(r["valor"]) if r["valor"] not in ("", None) else None)
                for r in data
            ]
            total += db.upsert_observations(con, rows, lag, snap)
            s = e + timedelta(days=1)
    return total


def _get_json_retry(url: str, attempts: int = 3):
    """SGS sometimes answers 200 with an empty or HTML body; retry those."""
    for i in range(attempts):
        payload = get(url)
        try:
            return payload, json.loads(payload)
        except json.JSONDecodeError:
            if i == attempts - 1:
                raise RuntimeError(f"SGS returned non-JSON {attempts} times: {url}")
            time.sleep(5 * (i + 1))


def _resume_from(con, sid: str) -> date | None:
    """Re-fetch the last 30 days too, to catch late revisions."""
    last = db.last_ref_date(con, sid)
    return last - timedelta(days=30) if last else None
