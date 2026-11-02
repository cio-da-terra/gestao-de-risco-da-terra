# Proof of concept: conilon yield, price and break-even

A first end-to-end test of the modelling idea on real data: forecast a
conilon farmer's yield and price with **calibrated 80% intervals**, then turn
them into the **chance of not covering costs**. Everything is backtested as if
forecasting in real time.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r poc/requirements.txt
.venv/bin/python -m poc.fetch     # once: fills data/cio_da_terra.sqlite (~5 min)
.venv/bin/python -m poc           # backtests + writes poc/results/
.venv/bin/python -m poc --municipio Linhares --cost 25000
```

## Setup

| | |
|---|---|
| Crop and region | Conilon, 18 municipalities in northern Espírito Santo (`config.py`) |
| Yield data | IBGE PAM "café total" yield, 2002–2025 (mostly conilon there) |
| Weather | NASA POWER daily rain and temperature at each municipality's centroid |
| Price | World Bank Pink Sheet robusta (USD/kg) × 60 × monthly PTAX → R$/saca |
| Forecast moment | End of February of harvest year Y (after flowering) |
| Intervals | Online split conformal + adaptive conformal inference (ACI), 80% |

**Point-in-time rules.** PAM for year Y-1 is only published around September
of Y, so a February forecast trains on years ≤ Y-2 and calibrates on errors
from years ≤ Y-2. Price models only learn from forecasts whose outcome month
has passed. This makes the scores worse than an in-sample fit, and honest.

**Yield model.** log yield = municipality level + common linear trend
(clonal varieties, irrigation) + anomaly. The anomaly is modelled from rain
by crop stage, heat during grain filling, ENSO and the anomaly two years
before (clipped). Models: trend only, ridge, gradient boosting, small MLP.

**Price model.** 6-month-ahead log change from the last 12 monthly returns,
season and ENSO. Models: random walk, ridge, small MLP.

**Break-even.** Monte Carlo over the out-of-sample errors of the best yield
and price models, drawn independently, against an **illustrative** cost of
R$ 20,000/ha in 2025 reais deflated by IPCA. It also runs a "price locked in
February" scenario.

## Results

Full tables in [`results/summary.md`](results/summary.md).

**Yield**, 18 municipalities, forecast years 2012–2025:

| Model | MAE (sacas/ha) | Coverage of 80% interval (ACI) | Mean width (sacas/ha) |
|---|---|---|---|
| Trend only | 9.1 | 77.8% | 36.7 |
| Ridge | **8.8** | 77.4% | 31.8 |
| Gradient boosting | 9.1 | 76.6% | 31.4 |
| MLP (neural net) | 9.7 | 82.5% | 39.6 |

![Yield backtest, Jaguaré](results/yield_backtest.png)

**Price**, 6 months ahead, origins 2010–2025:

| Model | MAE (R$/saca) | Coverage (ACI) | Mean width (% of price) |
|---|---|---|---|
| Random walk | 91.7 | 76.9% | 38.8 |
| Ridge | **89.7** | 76.9% | 41.3 |
| MLP (neural net) | 118.1 | 79.6% | 56.9 |

![Price backtest](results/price_backtest.png)

**Break-even**, Jaguaré:

![Chance of not covering costs](results/breakeven.png)

## What we learned

1. **Weather adds little to the yield forecast so far.** Ridge beats the
   trend baseline by only 0.3 sacas/ha. The signs make sense (El Niño and
   heat lower yield, rain at flowering raises it), but February weather
   explains a small part of the anomaly.
2. **The 2015–2017 drought is the failure case.** Nothing like it was in the
   training years, and it hit every municipality at once: coverage in 2016 was
   0–17%. Errors are correlated across municipalities within a year, so 80%
   coverage holds on average across years, not in each year.
3. **Price six months ahead is close to a random walk.** Ridge ties it, and
   the MLP is worse. The value is in the interval (about ±20%), not in the
   point forecast.
4. **Yield risk dominates the farmer's risk.** Locking the price in February
   barely changes the chance of not covering costs. That supports focusing
   the tool on yield protection (Proagro, insurance, Garantia-Safra) over
   price hedging, consistent with the "no natural hedge" argument.
5. **The break-even signal is not yet reliable.** For Jaguaré it gave 13–24%
   in the 2013–2017 loss years and 80% in 2019, a good year. The yield model
   is the weak link.
6. **The neural network meets the challenge requirement but loses** to
   simpler models on both tasks, as expected with about 400 yield rows.

## Limitations

- **PAM municipal yields are coarse:** values are often repeated across years
  (Jaguaré: 23, 23, 30, 30, 32, 32 … 45, 45, 45), which caps any model's
  accuracy.
- **Price is not the farmer's price:** Pink Sheet robusta is an international
  reference. CEPEA's conilon indicator, with local basis, is the right series.
- **The cost is illustrative.** Replace it with Conab production costs.
- No irrigation data, no water balance (rain minus evapotranspiration), and
  NASA POWER's grid is coarse (~50 km).
- Yield and price are drawn independently; FX risk is folded into the price.

## Next steps

- Water-balance features (ERA5 or CHIRPS + evapotranspiration) and a share of
  irrigated area per municipality.
- A common "year shock" term (or a state-level in-season estimate from IBGE
  LSPA / Conab) to handle the correlated errors behind the 2016 miss.
- Use Conab or LSPA estimates for year Y-1 to close the two-year PAM gap.
- CQR with quantile models, so intervals can be wider where the model is less sure.
- CEPEA conilon prices and Conab costs, then re-run the break-even.
