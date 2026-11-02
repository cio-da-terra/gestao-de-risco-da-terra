"""Break-even: combine the yield and price forecasts into P(revenue < cost).

Monte Carlo over the conformal (out-of-sample) error distributions, drawing
yield and price independently. Independence is a deliberate simplification
that also reflects the glossary's "no natural hedge" point: a local drought
cuts the farmer's yield without moving the international price.

The cost per hectare is an ILLUSTRATIVE input, not Conab data: a base value
in 2025 reais, deflated by IPCA to each year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HARVEST_MONTH = 8       # forecast made in Feb, price taken 6 months later (Aug)


def deflator(ipca_monthly: pd.Series, base_year: int) -> pd.Series:
    """Price level per year (Dec), relative to base_year = 1."""
    idx = (1 + ipca_monthly / 100).cumprod()
    dec = idx[idx.index.month == 12]
    level = pd.Series(dec.values, index=dec.index.year)
    return level / level.loc[base_year]


def simulate_year(yres, pres, code: int, year: int, cost_ha: float,
                  ymodel: str, pmodel: str, n: int = 20000, seed: int = 0) -> dict | None:
    rng = np.random.default_rng(seed + year)
    yrow = yres[(yres.model == ymodel) & (yres.code == code) & (yres.year == year)]
    origin = pd.Timestamp(year=year, month=2, day=1)
    prow = pres[(pres.model == pmodel) & (pres.origin == origin)]
    if yrow.empty or prow.empty:
        return None
    yrow, prow = yrow.iloc[0], prow.iloc[0]

    y_err = yres[(yres.model == ymodel) & (yres.year <= year - 2)].err.to_numpy()
    p_all = pres[pres.model == pmodel]
    p_err = p_all[p_all.target_date <= origin].err.to_numpy()
    if len(y_err) < 10 or len(p_err) < 24:
        return None

    y = np.exp(yrow.pred_log + rng.choice(y_err, n))        # sacas/ha
    p = prow.p0 * np.exp(prow.pred + rng.choice(p_err, n))  # R$/saca
    revenue = y * p
    locked = y * prow.p0                                     # price fixed today
    be_price = cost_ha / np.median(y)
    return {
        "year": year,
        "cost_ha": cost_ha,
        "yield_p10": np.quantile(y, 0.1), "yield_p50": np.median(y), "yield_p90": np.quantile(y, 0.9),
        "price_feb": prow.p0, "price_p50": np.median(p),
        "breakeven_price": be_price,
        "p_loss": (revenue < cost_ha).mean(),
        "p_loss_price_locked": (locked < cost_ha).mean(),
        "actual_yield": yrow.y,
        "actual_price": prow.p_actual,
        "actual_loss": yrow.y * prow.p_actual < cost_ha,
    }


def run(yres, pres, ipca, code: int, cost_2025: float, ymodel: str, pmodel: str) -> pd.DataFrame:
    defl = deflator(ipca, 2025)
    rows = []
    for year in sorted(yres.year.unique()):
        if year not in defl.index:
            continue
        r = simulate_year(yres, pres, code, year, cost_2025 * defl.loc[year], ymodel, pmodel)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)
