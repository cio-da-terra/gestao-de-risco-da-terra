"""Causal step 1: dose–response of conilon yield to heat during grain filling.

    .venv/bin/python -m poc.causal_heat     # writes poc/results/causal_heat.md

Question: holding place, trend, rain and ENSO fixed, how much yield does
each extra hot day (Tmax >= 34 °C, Nov of Y-1 to Feb of Y) cost?

Identification: conditional on municipality, a common trend and the other
weather of the season, heat is as good as random (weather is not chosen by
farmers). The main threat is drought: hot spells and dry spells come
together (r ≈ -0.7 with grain-filling rain), so rain by stage is controlled.
Controls deliberately exclude mean temperature (the treatment itself, in
another unit) and the LSPA estimates (they may already reflect the heat:
"bad controls").

Estimators:
  1. Fixed-effects regression (municipality + linear trend), errors
     clustered by year (experiment 4: errors share a year shock).
  2. Double ML, partially linear model (Chernozhukov et al., 2018): flexible
     ML predicts yield and heat from the controls, the effect is estimated on
     the residuals; cross-fitting by year, errors clustered by year.
Both are run with a linear dose and with dose bins (the dose–response).
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from . import data  # noqa: E402
from . import yield_model as ym  # noqa: E402
from .__main__ import ACCENT, ALERT, BAND, INK, MUTED, OUT, _style  # noqa: E402

RAIN = ["rain_veg", "rain_flower", "rain_fill"]
CONTROLS = RAIN + ["oni"]
BINS = [(0, 0, "0 days"), (1, 10, "1–10"), (11, 25, "11–25"), (26, 45, "26–45"), (46, 999, "46+")]
N_FOLDS = 6
N_REPS = 5


# --- data -----------------------------------------------------------------

def heat_exposure(wx: pd.DataFrame, threshold: float, shift_years: int = 0) -> pd.DataFrame:
    """Hot days and degree-days above `threshold` in Nov (Y-1)–Feb (Y).

    shift_years=1 gives the placebo: the same window one season *later*
    (after the harvest), which cannot cause this year's yield.
    """
    w = wx[(wx.date.dt.month >= 11) | (wx.date.dt.month <= 2)].copy()
    w["year"] = np.where(w.date.dt.month >= 11, w.date.dt.year + 1, w.date.dt.year) - shift_years
    g = w.groupby(["code", "year"]).tmax
    return pd.DataFrame({"days": g.apply(lambda s: (s >= threshold).sum()),
                         "dd": g.apply(lambda s: np.clip(s - threshold, 0, None).sum())})


def build_panel(con) -> pd.DataFrame:
    muns = data.municipios(con)
    wx = data.weather(con, muns)
    df = ym.build_features(data.yields(con, muns), wx, data.oni(con))
    for thr in (32, 34, 35):
        h = heat_exposure(wx, thr)
        df = df.merge(h.rename(columns={"days": f"hot{thr}", "dd": f"edd{thr}"}),
                      left_on=["code", "year"], right_index=True, how="left")
    plc = heat_exposure(wx, 34, shift_years=1).rename(columns={"days": "hot34_next", "dd": "_"})
    df = df.merge(plc[["hot34_next"]], left_on=["code", "year"], right_index=True, how="left")
    df["log_y"] = np.log(df.y)
    df["t"] = df.year - 2000
    return df.reset_index(drop=True)


def bin_dummies(d: pd.Series) -> pd.DataFrame:
    """Dose bins, 0 days is the reference."""
    return pd.DataFrame({lab: ((d >= lo) & (d <= hi)).astype(float)
                         for lo, hi, lab in BINS[1:]}, index=d.index)


# --- inference ------------------------------------------------------------

def cluster_cov(V: np.ndarray, e: np.ndarray, groups: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """Cluster-robust (CR1) covariance: bread @ meat @ bread."""
    k = V.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        s = V[groups == g].T @ e[groups == g]
        meat += np.outer(s, s)
    G, n = len(np.unique(groups)), len(e)
    adj = G / (G - 1) * (n - 1) / max(n - k, 1)
    return adj * bread @ meat @ bread


def ols_fe(df: pd.DataFrame, treat: list[str], year_fe: bool = False, controls=None):
    """log_y ~ treat + controls + municipality FE + (trend | year FE), CR1 by year."""
    controls = CONTROLS if controls is None else controls
    X = [df[treat].to_numpy(float), df[controls].to_numpy(float),
         pd.get_dummies(df.code, drop_first=False).to_numpy(float)]
    if year_fe:
        X.append(pd.get_dummies(df.year, drop_first=True).to_numpy(float))
    else:
        X.append(df[["t"]].to_numpy(float))
    X = np.column_stack(X)
    y = df.log_y.to_numpy()
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    cov = cluster_cov(X, y - X @ beta, df.year.to_numpy(), bread)
    k = len(treat)
    return beta[:k], np.sqrt(np.diag(cov)[:k])


def dml_plr(df: pd.DataFrame, treat: list[str], seed: int = 0):
    """Partially linear double ML with cross-fitting by year, repeated N_REPS times."""
    W = np.column_stack([df[CONTROLS].to_numpy(float), df[["t"]].to_numpy(float),
                         pd.get_dummies(df.code).to_numpy(float)])
    y = df.log_y.to_numpy()
    D = df[treat].to_numpy(float)
    years = df.year.to_numpy()
    thetas, covs = [], []
    for r in range(N_REPS):
        # shuffle which years share a fold, so results don't hinge on one split
        uy = np.random.default_rng(seed + r).permutation(np.unique(years))
        fold_of = {yr: i % N_FOLDS for i, yr in enumerate(uy)}
        groups = np.array([fold_of[yr] for yr in years])
        ry, rD = np.zeros_like(y), np.zeros_like(D)
        for tr, te in GroupKFold(N_FOLDS).split(W, y, groups):
            def learner():
                return GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                 learning_rate=0.05, subsample=0.8,
                                                 random_state=seed + r)
            ry[te] = y[te] - learner().fit(W[tr], y[tr]).predict(W[te])
            for j in range(D.shape[1]):
                rD[te, j] = D[te, j] - learner().fit(W[tr], D[tr, j]).predict(W[te])
        bread = np.linalg.pinv(rD.T @ rD)
        theta = bread @ rD.T @ ry
        thetas.append(theta)
        covs.append(cluster_cov(rD, ry - rD @ theta, years, bread))
    thetas = np.array(thetas)
    theta = np.median(thetas, 0)
    # median aggregation of repeated cross-fitting (Chernozhukov et al., 2018)
    var = np.median([np.diag(c) + (t - theta) ** 2 for c, t in zip(covs, thetas)], 0)
    return theta, np.sqrt(var)


def pct(b):
    return 100 * (np.exp(b) - 1)


def row(label, b, se, n_years):
    tcrit = stats.t.ppf(0.975, n_years - 1)   # few clusters: t with G-1 df
    return {"Specification": label, "Effect per hot day (%)": pct(b),
            "95% IC low": pct(b - tcrit * se), "95% IC high": pct(b + tcrit * se),
            "p-value": 2 * stats.t.sf(abs(b / se), n_years - 1)}


# --- report ---------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    con = data.connect()
    df = build_panel(con)
    G = df.year.nunique()

    # Linear dose: main estimates
    main = []
    b, se = ols_fe(df, ["hot34"])
    main.append(row("Fixed effects (municipality + trend)", b[0], se[0], G))
    b, se = dml_plr(df, ["hot34"])
    main.append(row("Double ML (gradient boosting nuisances)", b[0], se[0], G))
    main = pd.DataFrame(main)

    # Robustness, all with fixed effects (fast) unless noted
    rob = []
    b, se = ols_fe(df, ["hot34"], controls=["oni"])
    rob.append(row("No rain controls (confounded by drought)", b[0], se[0], G))
    for thr in (32, 35):
        b, se = ols_fe(df, [f"hot{thr}"])
        rob.append(row(f"Threshold {thr} °C", b[0], se[0], G))
    b, se = ols_fe(df, ["edd34"])
    r = row("Degree-days above 34 °C (per °C·day)", b[0], se[0], G)
    rob.append(r)
    b, se = ols_fe(df, ["hot34"], year_fe=True)
    rob.append(row("Year fixed effects instead of trend", b[0], se[0], G))
    for drop, lab in (([2016], "Without 2016"), ([2015, 2016, 2017], "Without 2015–2017 drought"),
                      ([2024], "Without 2024")):
        d = df[~df.year.isin(drop)]
        b, se = ols_fe(d, ["hot34"])
        rob.append(row(lab, b[0], se[0], d.year.nunique()))
    d = df.dropna(subset=["hot34_next"])
    b, se = ols_fe(d, ["hot34", "hot34_next"])
    rob.append(row("Placebo: heat in the season *after* harvest", b[1], se[1], d.year.nunique()))
    rob.append(row("  …same model, real heat", b[0], se[0], d.year.nunique()))
    rob = pd.DataFrame(rob)

    # Dose–response by bins
    bins = bin_dummies(df.hot34)
    dfb = pd.concat([df, bins], axis=1)
    labs = list(bins.columns)
    counts = [int((df.hot34 == 0).sum())] + [int(bins[c].sum()) for c in labs]
    b_fe, se_fe = ols_fe(dfb, labs)
    b_ml, se_ml = dml_plr(dfb, labs)
    tcrit = stats.t.ppf(0.975, G - 1)
    dose = pd.DataFrame({
        "Hot days": ["0 days"] + labs, "Observations": counts,
        "FE effect (%)": [0.0] + list(pct(b_fe)),
        "FE 95% IC": ["reference"] + [f"[{pct(b - tcrit * s):.0f}, {pct(b + tcrit * s):.0f}]"
                                      for b, s in zip(b_fe, se_fe)],
        "DML effect (%)": [0.0] + list(pct(b_ml)),
        "DML 95% IC": ["reference"] + [f"[{pct(b - tcrit * s):.0f}, {pct(b + tcrit * s):.0f}]"
                                       for b, s in zip(b_ml, se_ml)],
    })

    # Plot
    x = np.arange(len(labs) + 1)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axhline(0, color="#D8D4CA", lw=1)
    for off, b, s, color, lab in ((-0.12, b_fe, se_fe, INK, "Fixed effects"),
                                   (0.12, b_ml, se_ml, ACCENT, "Double ML")):
        est = np.r_[0, pct(b)]
        lo = np.r_[0, pct(b - tcrit * s)]
        hi = np.r_[0, pct(b + tcrit * s)]
        ax.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="o", color=color, ms=5,
                    capsize=3, lw=1.4, label=f"{lab} (95% IC)")
    ax.set_xticks(x, [f"{l}\n(n={n})" for l, n in zip(["0 days"] + labs, counts)])
    ax.set_xlabel("Days with Tmax ≥ 34 °C during grain filling (Nov–Feb)", color=MUTED, fontsize=9)
    ax.set_ylabel("Yield change vs 0 hot days (%)", color=MUTED, fontsize=9)
    _style(ax, "Heat during grain filling and conilon yield (ES, 2002–2025)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "causal_heat_dose.png", dpi=150)
    plt.close(fig)

    fmt = {"Effect per hot day (%)": "{:.2f}", "95% IC low": "{:.2f}", "95% IC high": "{:.2f}",
           "p-value": "{:.3f}"}

    def show(t):
        t = t.copy()
        for c, f in fmt.items():
            if c in t:
                t[c] = t[c].map(f.format)
        return t.to_markdown(index=False)

    with open(OUT / "causal_heat.md", "w") as f:
        f.write("# Heat dose–response (causal step 1)\n\nGenerated by `python -m poc.causal_heat`. "
                f"{len(df)} municipality-years, {df.code.nunique()} municipalities, "
                f"{df.year.min()}–{df.year.max()}; 95% IC use errors clustered by year "
                f"(t with {G - 1} df).\n\n## Effect of one more hot day\n\n")
        f.write(show(main) + "\n\n## Dose–response\n\n" + dose.round(1).to_markdown(index=False))
        f.write("\n\n![Dose–response](causal_heat_dose.png)\n\n## Robustness (fixed effects)\n\n")
        f.write(show(rob) + "\n")
    print((OUT / "causal_heat.md").read_text())


if __name__ == "__main__":
    main()
