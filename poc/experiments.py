"""Experiments that test the project's main assumptions on the PoC data.

    .venv/bin/python -m poc.experiments      # writes poc/results/experiments.md

Each experiment states the assumption, the test and the verdict. Numbers are
computed, never typed in, so the report can be regenerated as data changes.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from . import data  # noqa: E402
from . import price_model as pm  # noqa: E402
from . import yield_model as ym  # noqa: E402
from .__main__ import ACCENT, ALERT, BAND, INK, MUTED, OUT, _style  # noqa: E402

N_PERM = 200
SEED = 0
WEATHER = ["rain_veg", "rain_flower", "rain_fill", "tmax_fill", "hot_fill", "oni"]
GROUPS = {
    "rain (3 stages)": ["rain_veg", "rain_flower", "rain_fill"],
    "heat": ["tmax_fill", "hot_fill"],
    "ENSO": ["oni"],
    "biennial lag": ["lag2_anom"],
    "LSPA (state estimates)": ["lspa_prev", "lspa_cur"],
}


def mae(preds: pd.DataFrame, model: str = "ridge") -> float:
    d = preds[(preds.model == model) & (preds.year >= ym.REPORT_START)]
    return float((np.exp(d.pred_log) - d.y).abs().mean())


def anomalies_all_years(feats: pd.DataFrame) -> pd.DataFrame:
    """Descriptive anomalies: trend fitted on all years (not for forecasting)."""
    trend = ym._fit_trend(feats)
    return feats.assign(anom=np.log(feats.y) - trend(feats))[["code", "year", "y", "anom"]]


def perm_pvalue(x, y, n=10000, seed=SEED) -> float:
    rng = np.random.default_rng(seed)
    r = abs(np.corrcoef(x, y)[0, 1])
    null = np.array([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(n)])
    return float((null >= r).mean())


# ---------------------------------------------------------------------------

def exp1_natural_hedge(anom, price_brl, price_usd):
    """Does a bad conilon year in ES come with a higher price?"""
    state = anom.groupby("year").anom.median()

    def harvest_change(p):
        h = p[p.index.month.isin([5, 6, 7, 8])]
        lvl = np.log(h.groupby(h.index.year).mean())
        return lvl.diff()

    rows, scatter = [], None
    for label, p in (("BRL/saca", price_brl), ("USD/kg", price_usd)):
        ch = harvest_change(p)
        d = pd.concat([state.rename("anom"), ch.rename("dp")], axis=1).dropna()
        r, _ = stats.pearsonr(d.anom, d.dp)
        rho, _ = stats.spearmanr(d.anom, d.dp)
        rows.append({"Price": label, "Years": len(d), "Pearson r": r, "Spearman ρ": rho,
                     "Permutation p": perm_pvalue(d.anom.values, d.dp.values)})
        if label == "USD/kg":
            scatter = d
    table = pd.DataFrame(rows).round(3)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.axhline(0, color="#D8D4CA", lw=1)
    ax.axvline(0, color="#D8D4CA", lw=1)
    ax.scatter(scatter.anom * 100, scatter.dp * 100, color=ACCENT, s=28, zorder=3)
    for yr, row in scatter.iterrows():
        if abs(row.anom) > 0.2 or abs(row.dp) > 0.3:
            ax.annotate(str(yr), (row.anom * 100, row.dp * 100), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("ES conilon yield anomaly vs trend (%, log)", color=MUTED, fontsize=9)
    ax.set_ylabel("Robusta price change, harvest to harvest (%, log, USD)", color=MUTED, fontsize=9)
    _style(ax, "A bad harvest in ES does not reliably come with a higher price")
    fig.tight_layout()
    fig.savefig(OUT / "exp1_natural_hedge.png", dpi=150)
    plt.close(fig)

    r_usd = table.loc[table.Price == "USD/kg", "Pearson r"].iloc[0]
    p_usd = table.loc[table.Price == "USD/kg", "Permutation p"].iloc[0]
    if p_usd >= 0.05:
        verdict = (f"**Supported.** No significant link (r = {r_usd:.2f}, p = {p_usd:.2f}): "
                   "a bad local harvest is not offset by a higher price.")
    elif r_usd < 0:
        verdict = (f"**Partly contradicted.** Bad ES years come with higher prices "
                   f"(r = {r_usd:.2f}, p = {p_usd:.2f}): there is some natural hedge.")
    else:
        verdict = f"**Contradicted in the other direction** (r = {r_usd:.2f}, p = {p_usd:.2f})."
    return table, verdict


def exp2_risk_split(yres, pres, be):
    """Which matters more for revenue: yield or price uncertainty?"""
    ye = yres[(yres.model == "ridge") & (yres.year >= ym.REPORT_START)].err
    pe = pres[(pres.model == "random_walk") & (pres.origin >= pm.REPORT_START)].err
    vy, vp = ye.var(), pe.var()
    table = pd.DataFrame({
        "Source": ["Yield (ridge, Feb forecast)", "Price (random walk, 6 months)"],
        "Std of log error": [ye.std(), pe.std()],
        "Share of revenue variance (independent)": [vy / (vy + vp), vp / (vy + vp)],
    }).round(3)
    extra = ""
    if be is not None and not be.empty:
        extra = (f" In the break-even demo, locking the price in February changed the chance of "
                 f"not covering costs by {((be.p_loss_price_locked - be.p_loss).abs().mean() * 100):.1f} "
                 "points on average.")
    share = vy / (vy + vp)
    verdict = (f"**{'Supported' if share > 0.5 else 'Not supported'}.** Yield accounts for "
               f"{share:.0%} of the revenue variance.{extra}")
    return table, verdict


def exp3_weather_skill(feats):
    """Is the ridge gain over the trend real, and which features carry it?"""
    base, _ = ym.backtest(feats, models=["trend", "ridge"])
    mae_trend, mae_ridge = mae(base, "trend"), mae(base, "ridge")

    rng = np.random.default_rng(SEED)
    years = np.sort(feats.year.unique())
    by_year = feats.set_index(["code", "year"])[WEATHER]
    perm_maes = []
    for _ in range(N_PERM):
        mapping = dict(zip(years, rng.permutation(years)))   # same shuffle for every municipality
        idx = list(zip(feats.code, feats.year.map(mapping)))
        shuffled = feats.copy()
        shuffled[WEATHER] = by_year.reindex(idx).to_numpy()
        shuffled = shuffled.dropna(subset=WEATHER)
        p, _ = ym.backtest(shuffled, models=["ridge"])
        perm_maes.append(mae(p, "ridge"))
    perm_maes = np.array(perm_maes)
    p_value = float((perm_maes <= mae_ridge).mean())

    abl = [{"Features removed": "none (full ridge)", "MAE (sc/ha)": mae_ridge}]
    for g, cols in GROUPS.items():
        keep = [f for f in ym.FEATURES if f not in cols]
        p, _ = ym.backtest(feats, models=["ridge"], features=keep)
        abl.append({"Features removed": g, "MAE (sc/ha)": mae(p, "ridge")})
    abl.append({"Features removed": "all (trend only)", "MAE (sc/ha)": mae_trend})
    abl = pd.DataFrame(abl)
    abl["Change vs full"] = abl["MAE (sc/ha)"] - mae_ridge
    abl = abl.round(2)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(perm_maes, bins=25, color=BAND, edgecolor="white")
    ax.axvline(mae_ridge, color=ACCENT, lw=2, label=f"Real weather: {mae_ridge:.2f}")
    ax.axvline(mae_trend, color=INK, lw=1.2, ls="--", label=f"Trend only: {mae_trend:.2f}")
    ax.set_xlabel("Ridge MAE with weather shuffled across years (sacas/ha)", color=MUTED, fontsize=9)
    _style(ax, f"Permutation test, {N_PERM} shuffles")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp3_permutation.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame([{"MAE trend": mae_trend, "MAE ridge": mae_ridge,
                             "Median MAE, shuffled weather": float(np.median(perm_maes)),
                             "Permutation p": p_value}]).round(3)
    verdict = (f"**{'Supported' if p_value < 0.05 else 'Not yet supported'}.** "
               f"Shuffled weather did as well as real weather in {p_value:.0%} of {N_PERM} runs.")
    return summary, abl, verdict


def exp4_exchangeability(yres):
    """Conformal assumes errors are exchangeable; are they, across municipalities?"""
    d = yres[(yres.model == "ridge") & (yres.year >= ym.REPORT_START)].copy()
    year_mean = d.groupby("year").err.transform("mean")
    ss_tot = ((d.err - d.err.mean()) ** 2).sum()
    ss_year = ((year_mean - d.err.mean()) ** 2).sum()
    share = ss_year / ss_tot

    per_year = d.groupby("year").agg(n=("cov_aci", "size"), covered=("cov_aci", "sum"))
    lo = stats.binom.ppf(0.05, per_year.n, 0.8)
    hi = stats.binom.ppf(0.95, per_year.n, 0.8)
    per_year["outside_band"] = (per_year.covered < lo) | (per_year.covered > hi)
    n_out, n_years = int(per_year.outside_band.sum()), len(per_year)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.fill_between(per_year.index, lo / per_year.n * 100, hi / per_year.n * 100, color=BAND,
                    step="mid", label="90% range if errors were independent")
    colors = [ALERT if o else ACCENT for o in per_year.outside_band]
    ax.bar(per_year.index, per_year.covered / per_year.n * 100, color=colors, width=0.5)
    ax.axhline(80, color=INK, lw=1, ls="--", label="Target 80%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of municipalities covered", color=MUTED, fontsize=9)
    _style(ax, "Coverage per year swings more than chance allows")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "exp4_coverage_by_year.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame([{"Error variance explained by the year": share,
                           "Years outside the 90% binomial range": f"{n_out} of {n_years}",
                           "Expected if independent": f"about {0.1 * n_years:.1f}"}])
    verdict = ("**Not supported.** " if share > 0.2 or n_out > 0.2 * n_years else "**Supported.** ") + (
        f"{share:.0%} of the error is a shared year shock, so the 18 municipalities in a year "
        "behave more like one draw than 18. Coverage holds on average, not year by year.")
    return table.round(3), verdict


def exp5_publication_lag(feats):
    """How much does ignoring PAM's publication lag flatter the results?"""
    rows = []
    for lag, label in ((2, "Realistic: train on ≤ Y-2"), (1, "Leaky: train on ≤ Y-1")):
        p, _ = ym.backtest(feats, models=["trend", "ridge"], lag=lag)
        res = ym.add_intervals(p, lag=lag)
        r = res[(res.model == "ridge") & (res.year >= ym.REPORT_START)]
        rows.append({"Setup": label, "MAE ridge (sc/ha)": mae(p, "ridge"),
                     "MAE trend (sc/ha)": mae(p, "trend"),
                     "Coverage 80% (fixed)": r.cov_fix.mean() * 100,
                     "Mean width (sc/ha)": (r.hi_fix - r.lo_fix).mean()})
    t = pd.DataFrame(rows).round(2)
    gain = t["MAE ridge (sc/ha)"].iloc[0] - t["MAE ridge (sc/ha)"].iloc[1]
    verdict = (f"**{'Supported' if gain > 0.2 else 'Weakly supported'}.** The leaky setup looks "
               f"{gain:.2f} sacas/ha better, an improvement a real February forecast could not have.")
    return t, verdict


def exp6_biennial(anom):
    a = anom.sort_values(["code", "year"]).copy()
    g = a.groupby("code").anom
    a["lag1"], a["lag2"] = g.shift(1), g.shift(2)
    rows = []
    for k in ("lag1", "lag2"):
        d = a.dropna(subset=[k])
        r, p = stats.pearsonr(d.anom, d[k])
        rows.append({"Lag": k, "Pairs": len(d), "Correlation": r, "p-value": p})
    t = pd.DataFrame(rows).round(3)
    r1, r2 = t.Correlation.iloc[0], t.Correlation.iloc[1]
    if r1 < -0.1 and r2 > 0.1:
        verdict = "**Biennial pattern present**: anomalies alternate (negative at lag 1, positive at lag 2)."
    else:
        verdict = (f"**Supported: biennial bearing is weak.** Lag-1 r = {r1:.2f}, lag-2 r = {r2:.2f}. "
                   "Anomalies persist rather than alternate (droughts last more than a year). "
                   "PAM for Y-1 is not published by February, so the model takes this signal from "
                   "IBGE LSPA instead (state estimates for Y-1 and the first estimate for Y; see the "
                   "LSPA row of experiment 3's ablation). Part of the persistence may also come "
                   "from PAM values repeated year to year (experiment 7).")
    return t, verdict


def exp7_pam_quality(feats):
    d = feats.sort_values(["code", "year"])
    rep = d.groupby("code").y.apply(lambda s: (s.diff() == 0).iloc[1:].mean())
    rounded = (d.y * 60 % 60 == 0).mean()  # yields that are whole sacas/ha
    t = pd.DataFrame([{"Years equal to the previous year (median municipality)": rep.median(),
                       "Worst municipality": rep.max(),
                       "Yields that are a whole number of sacas/ha": rounded}]).round(3)
    verdict = (f"**{'Concern confirmed' if rep.median() > 0.1 else 'Minor concern'}.** In the typical "
               f"municipality, {rep.median():.0%} of years repeat the previous value; "
               f"{rounded:.0%} of values are whole sacas/ha. Part of the 'error' is measurement, not model.")
    return t, verdict


# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    con = data.connect()
    muns = data.municipios(con)
    feats = ym.build_features(data.yields(con, muns), data.weather(con, muns), data.oni(con),
                              data.lspa_vintages(con))
    anom = anomalies_all_years(feats)

    price_brl = data.price_brl_saca(con)
    usd = data._series(con, "wb:coffee_robusta").set_index("ref_date")["value"].sort_index()
    usd = usd[usd.index >= price_brl.index.min()]

    yp, _ = ym.backtest(feats)
    yres = ym.add_intervals(yp)
    pres = pm.add_intervals(pm.backtest(pm.build(price_brl, data.oni(con))))
    be_path = OUT / "breakeven.csv"
    be = pd.read_csv(be_path) if be_path.exists() else None

    sections = []

    def add(n, title, assumption, test, tables, verdict, fig=None):
        s = [f"## {n}. {title}\n", f"**Assumption.** {assumption}\n", f"**Test.** {test}\n"]
        s += [t.to_markdown(index=False) + "\n" for t in tables]
        if fig:
            s.append(f"![{title}]({fig})\n")
        s.append(f"**Verdict.** {verdict}\n")
        sections.append("\n".join(s))

    t, v = exp1_natural_hedge(anom, price_brl, usd)
    add(1, "No natural hedge",
        "A local yield loss is not compensated by a higher price, so farmers need yield protection.",
        "Correlate the median ES yield anomaly with the robusta price change between harvest "
        "seasons (May–Aug), in BRL and in USD. p-values from 10,000 permutations.",
        [t], v, "exp1_natural_hedge.png")

    t, v = exp2_risk_split(yres, pres, be)
    add(2, "Yield risk outweighs price risk",
        "For a February decision, the harvest is more uncertain than the price.",
        "Compare the out-of-sample log errors of the best yield model and of the 6-month random "
        "walk for price; revenue variance split assuming independence.",
        [t], v)

    s, a, v = exp3_weather_skill(feats)
    add(3, "Weather adds real skill",
        "Climate by crop stage improves the yield forecast beyond the trend.",
        f"Permutation test: shuffle the weather and ENSO features across years ({N_PERM} times, same "
        "shuffle for every municipality) and re-run the ridge backtest. Ablation: drop one feature "
        "group at a time.",
        [s, a], v, "exp3_permutation.png")

    t, v = exp4_exchangeability(yres)
    add(4, "Errors are exchangeable across municipalities",
        "Conformal intervals assume errors are exchangeable; pooling 18 municipalities per year "
        "treats them as 18 independent draws.",
        "Share of the ridge error variance explained by the year (one-way ANOVA), and per-year "
        "coverage against the 90% range of a Binomial(18, 0.8).",
        [t], v, "exp4_coverage_by_year.png")

    t, v = exp5_publication_lag(feats)
    add(5, "Respecting the PAM publication lag matters",
        "Using yield data that would not be published yet makes results look better than they are.",
        "Re-run the backtest training on years ≤ Y-1 (leaky) instead of ≤ Y-2 (realistic).",
        [t], v)

    t, v = exp6_biennial(anom)
    add(6, "Biennial bearing is weak for conilon",
        "Unlike arabica, conilon yields do not alternate strongly between high and low years.",
        "Within-municipality correlation of yield anomalies (vs trend) at lags 1 and 2.",
        [t], v)

    t, v = exp7_pam_quality(feats)
    add(7, "PAM municipal yields are informative",
        "Yearly municipal yields from PAM measure what happened each year.",
        "Share of years whose yield equals the previous year's, and share of whole-number values.",
        [t], v)

    header = ("# Experiments: testing the main assumptions\n\n"
              "Generated by `python -m poc.experiments` on the PoC data (18 conilon municipalities "
              "in Espírito Santo, forecast years "
              f"{ym.REPORT_START}–{feats.year.max()}). Verdicts are computed from the numbers.\n\n")
    (OUT / "experiments.md").write_text(header + "\n".join(sections))
    print((OUT / "experiments.md").read_text())


if __name__ == "__main__":
    main()
