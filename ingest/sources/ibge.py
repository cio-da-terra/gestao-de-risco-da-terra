"""IBGE: municipality codes and PAM yield (SIDRA).

PAM table 1613 = lavouras permanentes (coffee lives here).
Instead of hard-coding variable/category ids (easy to get wrong), we look
them up by name in the table's metadata.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import date, timedelta
from functools import lru_cache

from .. import db
from ..http import get, get_json

PAM_TABLE = 1613
# PAM for year Y is usually released around Sept/Oct of Y+1.
PAM_LAG = timedelta(days=365 + 270)

VARIABLES = {  # our short name -> substring of the SIDRA variable name
    "area_colhida": "Área colhida",
    "producao": "Quantidade produzida",
    "rendimento": "Rendimento médio",
}
PRODUCTS = {  # our short name -> substring of the c82 category name
    "cafe_total": "Café (em grão) Total",
    "cafe_arabica": "Café (em grão) Arábica",
    "cafe_canephora": "Café (em grão) Canephora",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().strip()


def municipio_code(name: str, uf: str) -> int:
    """'Patrocínio', 'MG' -> 7-digit IBGE code."""
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    for m in get_json(url):
        if _norm(m["nome"]) == _norm(name):
            return int(m["id"])
    raise KeyError(f"{name}/{uf} not found")


@lru_cache
def _metadata(table: int) -> dict:
    return get_json(f"https://servicodados.ibge.gov.br/api/v3/agregados/{table}/metadados")


def _variable_id(table: int, label: str) -> int:
    for v in _metadata(table)["variaveis"]:
        if _norm(label) in _norm(v["nome"]):
            return v["id"]
    raise KeyError(label)


def _category(table: int, label: str) -> tuple[int, int]:
    for c in _metadata(table)["classificacoes"]:
        for cat in c["categorias"]:
            if _norm(label) in _norm(cat["nome"]):
                return c["id"], cat["id"]
    raise KeyError(label)


def fetch_pam(con, municipios: list[int], products=("cafe_arabica", "cafe_canephora")) -> int:
    total = 0
    muns = ",".join(map(str, municipios))
    for prod in products:
        cls_id, cat_id = _category(PAM_TABLE, PRODUCTS[prod])
        for var, label in VARIABLES.items():
            var_id = _variable_id(PAM_TABLE, label)
            url = (f"https://apisidra.ibge.gov.br/values/t/{PAM_TABLE}/n6/{muns}"
                   f"/v/{var_id}/p/all/c{cls_id}/{cat_id}")
            payload = get(url, timeout=180)
            snap = db.save_raw(con, "ibge_pam", url, payload, ".json")
            data = json.loads(payload)
            header, body = data[0], data[1:]
            # SIDRA columns: D1C = municipality code, D3C = year (order follows the URL)
            rows_by_series: dict[str, list[db.Obs]] = {}
            for r in body:
                mun, year, raw = r["D1C"], r["D3C"], r["V"]
                sid = f"pam:{prod}:{var}:{mun}"
                val = float(raw) if raw not in ("-", "...", "X", "..") else None
                rows_by_series.setdefault(sid, []).append(db.Obs(sid, date(int(year), 12, 31), val))
            for sid, rows in rows_by_series.items():
                unit = body[0].get("MN", "") if body else ""
                db.register_series(con, sid, "ibge_pam", f"PAM {prod} {var}", unit, "yearly")
                total += db.upsert_observations(con, rows, PAM_LAG, snap)
    return total
