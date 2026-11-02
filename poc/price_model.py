"""Robusta price in BRL/saca, 6 months ahead: rolling backtest with conformal intervals.

Target: log(P[t+h] / P[t]). The random walk (no change) is the benchmark to
beat; the "simple neural network" required by the challenge is the MLP.
Models are refit every January on origins whose outcome was known by then.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .conformal import ACI, conformal_quantile

H = config.PRICE_HORIZON
N_LAGS = 12
REPORT_START = "2010-01-01"
MIN_SCORES = 24


def build(price: pd.Series, oni: pd.Series) -> pd.DataFrame:
    lp = np.log(price)
    r = lp.diff()
    df = pd.DataFrame({"logp": lp})
    for k in range(N_LAGS):
        df[f"r{k}"] = r.shift(k)
    df["msin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["mcos"] = np.cos(2 * np.pi * df.index.month / 12)
    # ONI for the season centred two months back is the latest one published
    df["oni"] = oni.reindex(df.index - pd.DateOffset(months=2)).to_numpy()
    df["target"] = lp.shift(-H) - lp
    return df.dropna(subset=[f"r{k}" for k in range(N_LAGS)] + ["oni"])


FEATURES = [f"r{k}" for k in range(N_LAGS)] + ["msin", "mcos", "oni"]


def make_models() -> dict:
    return {
        "random_walk": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "mlp": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0,
                                          max_iter=5000, random_state=0)),
    }


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    # Start early enough that every reported origin has >= MIN_SCORES past errors
    origins = df.index[(df.index >= "2005-01-01") & df.target.notna()]
    models = make_models()
    fitted_year = None
    out = []
    for t in origins:
        if t.year != fitted_year:           # refit each January
            train = df[(df.index <= t - pd.DateOffset(months=H)) & df.target.notna()]
            for m in models.values():
                if m is not None:
                    m.fit(train[FEATURES], train.target)
            fitted_year = t.year
        x = df.loc[[t], FEATURES]
        for name, m in models.items():
            pred = 0.0 if m is None else float(m.predict(x)[0])
            out.append({"model": name, "origin": t, "target_date": t + pd.DateOffset(months=H),
                        "p0": float(np.exp(df.at[t, "logp"])), "pred": pred,
                        "actual": float(df.at[t, "target"])})
    return pd.DataFrame(out)


def add_intervals(preds: pd.DataFrame, alpha: float = config.ALPHA) -> pd.DataFrame:
    """Scores and ACI feedback only from forecasts whose target date has passed."""
    preds = preds.copy()
    preds["err"] = preds.actual - preds.pred
    rows = []
    for name, g in preds.groupby("model"):
        g = g.sort_values("origin").reset_index(drop=True)
        aci = ACI(alpha, gamma=0.02)
        fed = 0
        q_fix, q_aci = [], []
        for i, row in g.iterrows():
            known = g[g.target_date <= row.origin]
            # feed ACI the outcomes that became known since the last step
            for j in range(fed, len(known)):
                if np.isfinite(q_aci[j]):
                    lo, hi = g.pred[j] - q_aci[j], g.pred[j] + q_aci[j]
                    aci.update(0.0 if lo <= g.actual[j] <= hi else 1.0)
            fed = len(known)
            scores = np.abs(known.err.to_numpy())
            ok = len(scores) >= MIN_SCORES
            q_fix.append(conformal_quantile(scores, alpha) if ok else np.inf)
            q_aci.append(conformal_quantile(scores, aci.alpha) if ok else np.inf)
        g["q_fix"], g["q_aci"] = q_fix, q_aci
        for tag in ("fix", "aci"):
            g[f"cov_{tag}"] = (g.err.abs() <= g[f"q_{tag}"])
            g[f"lo_{tag}"] = g.p0 * np.exp(g.pred - g[f"q_{tag}"])
            g[f"hi_{tag}"] = g.p0 * np.exp(g.pred + g[f"q_{tag}"])
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["p_pred"] = out.p0 * np.exp(out.pred)
    out["p_actual"] = out.p0 * np.exp(out.actual)
    return out


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    r = res[res.origin >= REPORT_START]
    g = r.groupby("model")
    s = pd.DataFrame({
        "MAE (R$/saca)": g.apply(lambda d: (d.p_pred - d.p_actual).abs().mean(), include_groups=False),
        "MAPE (%)": g.apply(lambda d: ((d.p_pred - d.p_actual).abs() / d.p_actual).mean() * 100,
                            include_groups=False),
        "Coverage 80% (fixed)": g.cov_fix.mean() * 100,
        "Coverage 80% (ACI)": g.cov_aci.mean() * 100,
        "Mean width ACI (% of price)": g.apply(lambda d: ((d.hi_aci - d.lo_aci) / d.p0).mean() * 100,
                                               include_groups=False),
    })
    return s.reindex(["random_walk", "ridge", "mlp"]).round(1)
