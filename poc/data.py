"""Load PoC data from the project database into pandas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ingest import db

from . import config


def connect():
    return db.connect()


def municipios(con) -> pd.DataFrame:
    df = pd.read_sql("SELECT code, name, uf, lat, lon FROM poc_municipio", con)
    keep = set(config.MUNICIPIOS)
    return df[[(n, u) in keep for n, u in zip(df.name, df.uf)]].reset_index(drop=True)


def _series(con, sql_like: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT series_id, ref_date, value FROM observation_latest WHERE series_id LIKE ?",
        con, params=(sql_like,))
    df["ref_date"] = pd.to_datetime(df["ref_date"])
    return df


def yields(con, muns: pd.DataFrame) -> pd.DataFrame:
    """Columns: code, year, y (sacas/ha). PAM gives kg/ha; 1 saca = 60 kg."""
    df = _series(con, f"pam:{config.PRODUCT}:rendimento:%")
    df["code"] = df.series_id.str.rsplit(":", n=1).str[1].astype(int)
    df = df[df.code.isin(muns.code) & df.value.notna() & (df.value > 0)]
    df = df.assign(year=df.ref_date.dt.year, y=df.value / 60.0)
    return df[df.year >= config.FIRST_YEAR][["code", "year", "y"]].reset_index(drop=True)


def weather(con, muns: pd.DataFrame) -> pd.DataFrame:
    """Daily weather per municipality: code, date, tmax, tmin, rain."""
    df = _series(con, "power:mun%")
    parts = df.series_id.str.split(":")
    df["code"] = parts.str[1].str.removeprefix("mun").astype(int)
    df["var"] = parts.str[2]
    df = df[df.code.isin(muns.code)]
    w = df.pivot_table(index=["code", "ref_date"], columns="var", values="value").reset_index()
    return w.rename(columns={"ref_date": "date", "T2M_MAX": "tmax", "T2M_MIN": "tmin",
                             "PRECTOTCORR": "rain"})


def lspa_vintages(con, uf: int = 32) -> pd.DataFrame:
    """Every published LSPA estimate for coffee in a state (default 32 = ES).

    Columns: product, var, ref_year, available_at (UTC), value. Read from
    `observation`, not `observation_latest`: the model must pick the vintage
    that was public at its forecast date.
    """
    df = pd.read_sql("SELECT series_id, ref_date, available_at, value FROM observation "
                     "WHERE series_id LIKE ?", con, params=(f"lspa:cafe_%:%:{uf}",))
    parts = df.series_id.str.split(":")
    df["product"], df["var"] = parts.str[1], parts.str[2]
    df["ref_year"] = pd.to_datetime(df.ref_date).dt.year
    df["available_at"] = pd.to_datetime(df.available_at, utc=True)
    return df[["product", "var", "ref_year", "available_at", "value"]]


def lspa_total_yield(vint: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Coffee-total yield (kg/ha) per year, as published before `cutoff`.

    LSPA reports "café total" for ES only up to 2017 and arabica/canephora
    separately from 2012, so later years are rebuilt the way PAM defines the
    total: (arabica + canephora production) / (arabica + canephora area).
    """
    known = vint[vint.available_at < cutoff].sort_values("available_at")
    last = known.groupby(["product", "var", "ref_year"]).value.last().unstack(["product", "var"])

    def col(p, v):
        return last[(p, v)] if (p, v) in last.columns else pd.Series(np.nan, index=last.index)

    total = col("cafe_total", "rendimento")
    prod = col("cafe_arabica", "producao").fillna(0) + col("cafe_canephora", "producao")
    area = col("cafe_arabica", "area_colhida").fillna(0) + col("cafe_canephora", "area_colhida")
    rebuilt = prod * 1000 / area      # production in tonnes, area in ha -> kg/ha
    return total.fillna(rebuilt).dropna()


def oni(con) -> pd.Series:
    df = _series(con, "noaa:oni")
    return df.set_index("ref_date")["value"].sort_index()


def price_brl_saca(con) -> pd.Series:
    """Monthly robusta price in BRL per 60 kg bag: Pink Sheet USD/kg x 60 x mean PTAX.

    The Pink Sheet robusta price is an international reference (ICO), not the
    price a conilon farmer in Espírito Santo receives; CEPEA's conilon
    indicator would be the better series once loaded.
    """
    usd = _series(con, config.PRICE_SERIES).set_index("ref_date")["value"].sort_index()
    fx = _series(con, "bcb:ptax_venda").set_index("ref_date")["value"].sort_index()
    fx_m = fx.resample("MS").mean()
    p = (usd * 60 * fx_m).dropna()
    return p[p.index >= config.PRICE_START]
