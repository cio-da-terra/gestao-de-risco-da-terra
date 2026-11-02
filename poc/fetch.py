"""Fill the project database with what the PoC needs, via the ingest package.

    .venv/bin/python -m poc.fetch
"""

from __future__ import annotations

from datetime import date

from ingest import db
from ingest.http import get_json
from ingest.sources import bcb, climate, ibge, prices

from . import config


def municipio_centroid(code: int) -> tuple[float, float]:
    """Rough centroid (mean of boundary vertices) from IBGE's mesh API.
    Good enough to pick the NASA POWER grid cell (0.5 x 0.625 degrees)."""
    gj = get_json(f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{code}"
                  "?formato=application/vnd.geo+json")
    pts = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            pts.append(c)
        else:
            for x in c:
                walk(x)

    for f in gj["features"]:
        walk(f["geometry"]["coordinates"])
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return lat, lon


def resolve_municipios(con) -> list[dict]:
    """Name -> IBGE code + centroid, cached in a small table."""
    con.execute("""CREATE TABLE IF NOT EXISTS poc_municipio
                   (code INTEGER PRIMARY KEY, name TEXT, uf TEXT, lat REAL, lon REAL)""")
    out = []
    for name, uf in config.MUNICIPIOS:
        row = con.execute("SELECT code, lat, lon FROM poc_municipio WHERE name=? AND uf=?",
                          (name, uf)).fetchone()
        if not row:
            try:
                code = ibge.municipio_code(name, uf)
            except KeyError:
                print(f"  skip {name}/{uf}: not found in IBGE")
                continue
            lat, lon = municipio_centroid(code)
            con.execute("INSERT INTO poc_municipio VALUES (?,?,?,?,?)",
                        (code, name, uf, lat, lon))
            con.commit()
            row = (code, lat, lon)
        out.append({"code": row[0], "name": name, "uf": uf, "lat": row[1], "lon": row[2]})
    return out


def main():
    con = db.connect()
    muns = resolve_municipios(con)
    print(f"{len(muns)} municipalities")

    print("PAM ...", ibge.fetch_pam(con, [m["code"] for m in muns], products=(config.PRODUCT,)))
    for m in muns:
        n = climate.fetch_nasa_power(con, f"mun{m['code']}", m["lat"], m["lon"],
                                     start=date(1990, 1, 1))
        print(f"POWER {m['name']}: {n}")
    print("ONI ...", climate.fetch_oni(con))
    print("BCB ...", bcb.fetch(con, start=date(1999, 1, 1)))
    print("Pink Sheet ...", prices.fetch_pink_sheet(con))


if __name__ == "__main__":
    main()
