"""Conilon yield: rolling-origin backtest with conformal intervals.

Forecast is issued at the end of February of harvest year Y (flowering done,
grain filling under way). Point-in-time rules:
  - PAM for year Y-1 is only published around Sept of Y, so models are trained
    on years <= Y-2 and conformal scores come from years <= Y-2.
  - Weather is used up to Feb of Y; ONI up to its Sep-Dec (Y-1) seasons.

Structure (in the spirit of Stracqualursi, 2026): log yield = municipality
level + common linear trend + anomaly. The trend absorbs clonal varieties and
irrigation; the models explain the anomaly from weather, ENSO and the
biennial cycle (anomaly two years before).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .conformal import ACI, conformal_quantile

FEATURES = ["rain_veg", "rain_flower", "rain_fill", "tmax_fill", "hot_fill", "oni", "lag2_anom"]
FEATURE_LABELS = {
    "rain_veg": "Rain Mar–Jun (Y-1), branch growth",
    "rain_flower": "Rain Jul–Oct (Y-1), flowering",
    "rain_fill": "Rain Nov (Y-1)–Feb (Y), grain filling",
    "tmax_fill": "Mean max temp, grain filling",
    "hot_fill": "Days ≥ 34 °C, grain filling",
    "oni": "ONI (ENSO), Sep–Dec (Y-1)",
    "lag2_anom": "Yield anomaly two years before (biennial)",
}
BACKTEST_START = 2009   # first forecast year; earlier years only train
REPORT_START = 2012     # first year with enough past residuals for intervals
HOT_DAY = 34.0
LAG_CLIP = 0.3          # log points


def build_features(yld: pd.DataFrame, wx: pd.DataFrame, oni: pd.Series) -> pd.DataFrame:
    w = wx.copy()
    w["m"] = w.date.dt.month
    w["yr"] = w.date.dt.year

    def window(mask, harvest_year, aggs):
        g = w[mask].assign(year=harvest_year[mask]).groupby(["code", "year"])
        return g.agg(**aggs)

    veg = window(w.m.between(3, 6), w.yr + 1, {"rain_veg": ("rain", "sum")})
    flower = window(w.m.between(7, 10), w.yr + 1, {"rain_flower": ("rain", "sum")})
    fill_mask = (w.m >= 11) | (w.m <= 2)
    fill_year = np.where(w.m >= 11, w.yr + 1, w.yr)
    fill = window(fill_mask, pd.Series(fill_year, index=w.index), {
        "rain_fill": ("rain", "sum"),
        "tmax_fill": ("tmax", "mean"),
        "hot_fill": ("tmax", lambda s: (s >= HOT_DAY).sum()),
    })
    o = oni[oni.index.month.isin([9, 10, 11, 12])]
    oni_y = o.groupby(o.index.year + 1).mean().rename("oni")

    df = yld.set_index(["code", "year"]).join([veg, flower, fill]).reset_index()
    df = df.merge(oni_y, left_on="year", right_index=True, how="left")
    return df.dropna(subset=["rain_veg", "rain_flower", "rain_fill", "oni"]).reset_index(drop=True)


def make_models() -> dict:
    return {
        "trend": None,  # trend + municipality level only
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "gbm": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                             min_samples_leaf=15, random_state=0),
        "mlp": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(16,), alpha=0.1,
                                          max_iter=5000, random_state=0)),
    }


def _fit_trend(known: pd.DataFrame):
    codes = sorted(known.code.unique())

    def design(d):
        X = pd.get_dummies(pd.Categorical(d.code, categories=codes)).to_numpy(float)
        return np.column_stack([X, d.year.to_numpy(float) - 2000])

    lr = LinearRegression(fit_intercept=False).fit(design(known), np.log(known.y))
    return lambda d: lr.predict(design(d))


def backtest(df: pd.DataFrame, models: list[str] | None = None, lag: int = 2,
             features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Out-of-sample predictions (log scale) for every model and forecast year.

    lag: latest PAM year known at forecast time is Y - lag (2 = realistic,
    1 = the leaky setup used in experiment 5). features: subset for ablations.
    """
    features = features or FEATURES
    out = []
    ridge_coefs = None
    for Y in range(BACKTEST_START, df.year.max() + 1):
        known = df[df.year <= Y - lag].copy()
        test = df[df.year == Y].copy()
        if test.empty:
            continue
        trend = _fit_trend(known)
        known["anom"] = np.log(known.y) - trend(known)
        anom = known.set_index(["code", "year"])["anom"]

        def lag2(d):
            # Clipped: a drought crash two years back (e.g. 2017) is not a
            # biennial "on/off" signal and would otherwise dominate the forecast.
            idx = list(zip(d.code, d.year - 2))
            return anom.reindex(idx).fillna(0.0).clip(-LAG_CLIP, LAG_CLIP).to_numpy()

        known["lag2_anom"] = lag2(known)
        test["lag2_anom"] = lag2(test)
        base = trend(test)
        for name, model in make_models().items():
            if models and name not in models:
                continue
            if model is None:
                pred = base
            else:
                model.fit(known[features], known.anom)
                pred = base + model.predict(test[features])
                if name == "ridge" and Y == df.year.max():
                    ridge_coefs = pd.Series(model[-1].coef_, index=features)
            out.append(pd.DataFrame({"model": name, "code": test.code.values,
                                     "year": Y, "y": test.y.values, "pred_log": pred}))
    return pd.concat(out, ignore_index=True), ridge_coefs


def add_intervals(preds: pd.DataFrame, alpha: float = config.ALPHA, lag: int = 2) -> pd.DataFrame:
    """Online conformal intervals; scores and ACI feedback only from years <= Y - lag."""
    preds = preds.copy()
    preds["err"] = np.log(preds.y) - preds.pred_log
    rows = []
    for name, g in preds.groupby("model"):
        aci = ACI(alpha)
        done: dict[int, pd.DataFrame] = {}   # forecast year -> its rows, once made
        applied: set[int] = set()
        for Y in sorted(g.year.unique()):
            # ACI feedback: only for years whose outcomes are published by now
            for past in sorted(p for p in done if p <= Y - lag and p not in applied):
                if np.isfinite(done[past].q_aci.iloc[0]):
                    aci.update(1 - done[past].cov_aci.mean())
                applied.add(past)
            scores = np.abs(g[g.year <= Y - lag].err.to_numpy())
            cur = g[g.year == Y].copy()
            q_fix = conformal_quantile(scores, alpha)
            q_aci = conformal_quantile(scores, aci.alpha)
            for tag, q in (("fix", q_fix), ("aci", q_aci)):
                cur[f"q_{tag}"] = q
                cur[f"lo_{tag}"] = np.exp(cur.pred_log - q)
                cur[f"hi_{tag}"] = np.exp(cur.pred_log + q)
                cur[f"cov_{tag}"] = (cur.y >= cur[f"lo_{tag}"]) & (cur.y <= cur[f"hi_{tag}"])
            cur["alpha_aci"] = aci.alpha
            done[Y] = cur
            rows.append(cur)
    out = pd.concat(rows, ignore_index=True)
    out["pred"] = np.exp(out.pred_log)
    return out


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    r = res[res.year >= REPORT_START]
    g = r.groupby("model")
    s = pd.DataFrame({
        "MAE (sc/ha)": g.apply(lambda d: (d.pred - d.y).abs().mean(), include_groups=False),
        "MAPE (%)": g.apply(lambda d: ((d.pred - d.y).abs() / d.y).mean() * 100, include_groups=False),
        "Coverage 80% (fixed)": g.cov_fix.mean() * 100,
        "Coverage 80% (ACI)": g.cov_aci.mean() * 100,
        "Mean width ACI (sc/ha)": g.apply(lambda d: (d.hi_aci - d.lo_aci).mean(), include_groups=False),
        "Coverage in 2016 (ACI)": r[r.year == 2016].groupby("model").cov_aci.mean() * 100,
    })
    order = ["trend", "ridge", "gbm", "mlp"]
    return s.reindex([m for m in order if m in s.index]).round(1)
