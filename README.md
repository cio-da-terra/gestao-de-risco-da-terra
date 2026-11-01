# cio-da-terra

AI4Good project: helping small family farmers understand the price and
yield risk on what they grow, and compare the ways they can protect their
income. The tool estimates a farmer's break-even and the chance that
harvest-time income falls below it, with calibrated uncertainty. It is
educational and never gives trading advice.

The design is not tied to one crop. **Coffee is the first target**: it has
a large smallholder base (arábica and conilon), long public price and yield
series, and a futures market to compare against. The current fetchers
therefore focus on coffee; other crops are added by extending the product
lists and price sources.

This README covers the project timeline and the **data ingestion** layer:
the code that fetches public data on a schedule and stores it so models can
be trained and backtested honestly.

## Timeline

The plan has 6 phases of roughly one week each. The final deadline is not set
yet, so weeks are relative. Status as of **2026-11-01**: phase 1 is mostly
done and **phase 2 (data) is in progress**.

| # | Phase | Status |
|---|---|---|
| 1 | Definition: challenge statement, roles, first crop, repo | mostly done |
| 2 | Data: ingestion, storage, knowledge base | **in progress** |
| 3 | Models: yield and price, calibrated uncertainty | not started |
| 4 | Agents: protection simulator and knowledge agent | not started |
| 5 | Integration: orchestrator, interface, MVP flow | not started |
| 6 | Tests, ethics review, pitch and rehearsal | not started |

### 1. Definition

- [x] Challenge statement and essential question ("ampliar, não substituir")
- [x] First target crop: coffee (arábica and conilon)
- [x] Repository, ingestion skeleton, glossary
- [ ] Pick the study municipalities (replace the examples in `LOCATIONS`)
- [ ] Define the environmental scope: which climate risks and which
      sustainable adaptation practices the tool covers
- [ ] Set the final deadline and turn these phases into dates

### 2. Data

- [x] Point-in-time SQLite store with raw snapshots
- [x] Fetchers: BCB (PTAX, Selic, IPCA), NASA POWER, NOAA ONI, IBGE PAM,
      World Bank Pink Sheet
- [x] GitHub Actions workflow and DVC remote (disabled stub)
- [ ] Conab production costs (needed for break-even)
- [ ] Conab producer prices (local price / basis)
- [ ] CEPEA: download sample files and test the parser; confirm licensing
- [ ] Update the Pink Sheet URL to the current edition
- [ ] CHIRPS rainfall and ERA5 temperature, aggregated per municipality
- [ ] ZARC planting windows for the chosen municipalities
- [ ] Decide on storage (Google Drive security concern), then enable the
      scheduled runs
- [ ] Knowledge base documents for the RAG: MCR (Pronaf, Proagro), CPR law,
      PNAE/PAA rules, Garantia-Safra, PGPM, ZARC, plus sources on
      sustainable adaptation practices; each with version and date

### 3. Models

- [ ] Features: GDD/EDD (heat), rainfall by crop stage, ENSO, FX
- [ ] Yield model: quantile LightGBM per municipality, against a trend baseline
- [ ] Price model: small neural network (MLP or LSTM) against a seasonal
      baseline. This covers the challenge's "simple neural network" constraint
- [ ] Calibrated intervals: CQR, with adaptive conformal (ACI) over time
- [ ] Backtests with `db.as_of()`, reporting coverage, MAPE and RMSE
- [ ] Track runs in MLflow, pointing to the DVC data version

### 4. Agents

- [ ] Protection agent: simulate income under each strategy (spot sale,
      PNAE/PAA, cooperative, CPR, insurance/Proagro, diversification)
      and report break-even and the probability of falling below it
- [ ] Knowledge agent: RAG over the programme documents, with citations and
      a check that answers come from the retrieved text
- [ ] Eligibility checks (CAF, Pronaf tier, ZARC window)

### 5. Integration

- [ ] Orchestrator (LLM) that coordinates the agents and explains results
      in plain language, never recommending a trade
- [ ] Referral path to ATER/Emater or a cooperative for out-of-scope cases
- [ ] Web interface with the full demo flow for one crop and one region

### 6. Tests and pitch

- [ ] Unit tests for agents and fetchers; model evaluation in CI
- [ ] Ethics review: uncertainty shown everywhere, no individual advice
      (CVM), no personal data stored (LGPD)
- [ ] Pitch script: problem (income and climate risk), solution, live demo,
      ethics, essential question
- [ ] Rehearsal

## Quick start

```bash
pip install -r requirements.txt        # only openpyxl, for Excel sources
python -m ingest daily                 # BCB + NASA POWER
python -m ingest monthly               # + ONI, Pink Sheet, IPCA, CEPEA files
python -m ingest yearly --only pam     # IBGE PAM yield
```

Data goes to `data/` (git-ignored): `data/cio_da_terra.sqlite` plus raw
downloads under `data/raw/`. The first run backfills history, later runs
fetch only what is new.

## Sources

| Source | Data | Cadence | Module | Status |
|---|---|---|---|---|
| BCB SGS | USD/BRL PTAX, Selic, IPCA | daily / monthly | `sources/bcb.py` | working |
| NASA POWER | daily max/min temperature, rain, humidity per location | daily | `sources/climate.py` | working |
| NOAA CPC | ONI (ENSO index) | monthly | `sources/climate.py` | working |
| World Bank Pink Sheet | arabica and robusta prices, USD/kg | monthly | `sources/prices.py` | working, URL needs updating (see below) |
| CEPEA/ESALQ | arábica and conilon indicators, BRL/saca | daily | `sources/prices.py` | parser for manual files, untested |
| IBGE PAM (SIDRA 1613) | coffee area, production, yield per municipality | yearly | `sources/ibge.py` | working |
| CHIRPS, ERA5 | gridded rain and temperature | daily | `sources/climate.py` | stub |
| Conab | producer prices, production costs | monthly / yearly | `sources/prices.py` | stub |

A longer cadence also runs the shorter ones: `monthly` includes `daily`,
`yearly` includes both. Use `--only <source> ...` to run specific sources
(names are the keys of `JOBS` in `ingest/__main__.py`).

Locations are set in `LOCATIONS` in `ingest/__main__.py`. The current
entries (Patrocínio/MG, Jaguaré/ES) are examples until the team picks the
study municipalities.

## How data is stored

The design goal is **point-in-time correctness**: a backtest for date D must
only see what was actually known on D. Otherwise the model looks better than
it is and the calibrated intervals are not really calibrated.

Every value in `observation` carries two dates:

- `ref_date`: the period the value describes (a day, a month, a harvest year)
- `available_at`: when it could first have been known

Rules:

- **Revisions are inserted, never overwritten.** If a source changes a past
  value, a new row is added with a later `available_at`.
- **Backfills get realistic dates.** On a point's first appearance,
  `available_at = min(fetched_at, ref_date + publication lag)`. The lag is
  set per source, e.g. PAM for year Y counts as known around September of
  Y+1, not the day it was downloaded.
- **Unchanged values are skipped**, so reruns are safe and add nothing.

Tables and views:

| Name | Purpose |
|---|---|
| `series` | one row per series: id, source, description, unit, frequency |
| `observation` | all values with `ref_date`, `available_at`, `fetched_at` |
| `observation_latest` | view: newest known value per point (what the app shows) |
| `raw_snapshot` | every untouched download, its hash and file path |
| `fetch_log` | one row per job run: ok/error, new rows, error message |

Series ids look like `source:name[:qualifiers]`, e.g. `bcb:ptax_venda`,
`power:patrocinio_mg:T2M_MAX`, `pam:cafe_arabica:rendimento:3148103`.

Reading data:

```python
from datetime import datetime, timezone
from ingest import db

con = db.connect()
# Exactly what was knowable on 1 March 2024, for a backtest
ptax = db.as_of(con, "bcb:ptax_venda", datetime(2024, 3, 1, tzinfo=timezone.utc))

# Latest values, for the app
con.execute("SELECT ref_date, value FROM observation_latest "
            "WHERE series_id = ? ORDER BY ref_date", ("noaa:oni",)).fetchall()
```

## Adding a source

1. Create a function `fetch_x(con) -> int` in `ingest/sources/`.
2. Download with `http.get()`, then keep the raw bytes:
   `snap = db.save_raw(con, "x", url, payload, ".json")`.
3. Register each series with `db.register_series(...)`.
4. Parse into `db.Obs(series_id, ref_date, value)` and store with
   `db.upsert_observations(con, rows, publication_lag, snap)`.
   Choose `publication_lag` from when the source actually publishes.
5. Add it to `JOBS` in `ingest/__main__.py` under the right cadence.

Each source runs independently: if one fails, it is logged in `fetch_log`,
the others still run, and the process exits with code 1.

## Adding a crop

- **Yield:** add the crop to `PRODUCTS` in `sources/ibge.py`, using its name
  as it appears in SIDRA (the id is looked up automatically). Coffee is in
  table 1613 (permanent crops); annual crops like corn, beans or cassava
  are in table 1612, so `PAM_TABLE` would need to become per-product.
- **Price:** add the commodity's column to `PINK_COLUMNS` if the Pink Sheet
  covers it, or a new price source for crops it doesn't (e.g. Conab Prohort
  for vegetables sold at Ceasas).

## Scheduled runs

`.github/workflows/ingest.yml` runs the jobs on GitHub Actions (weekdays for
daily, day 5 for monthly) and keeps the database in Google Drive through
DVC; git only holds the `data.dvc` pointer. **It is off by default.**
[`docs/ingest-setup.md`](docs/ingest-setup.md) explains how to turn it on.

## Known issues and open items

- **CEPEA licensing.** Its policy restricts redistributing price series.
  The files are downloaded by hand into `data/manual/cepea/`
  (`arabica.xlsx`, `conilon.xlsx`) and are never published raw. The Pink
  Sheet (CC BY 4.0) is the fallback.
- **Pink Sheet URL.** The World Bank changes the file link with each
  edition; the one in `prices.py` stops at Dec 2025. Take the current link
  from the World Bank commodity markets page.
- **Conab production costs** are needed for break-even and are not
  implemented yet.
- **Storage security.** Keeping data on Google Drive is under discussion
  and should be settled before scheduled runs are turned on.
- SIDRA variable and category ids are looked up by name from the table
  metadata; BCB series codes are hard-coded in `bcb.py`.

## References

### Methods

- Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under
  distribution shift. *NeurIPS* 34. [arXiv:2106.00170](https://arxiv.org/abs/2106.00170).
  The basis for keeping interval coverage valid as climate and prices shift.
- Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized quantile
  regression. *NeurIPS* 32. [arXiv:1905.03222](https://arxiv.org/abs/1905.03222).
- Angelopoulos, A. N. & Bates, S. (2021). A gentle introduction to conformal
  prediction and distribution-free uncertainty quantification.
  [arXiv:2107.07511](https://arxiv.org/abs/2107.07511).
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal* 21(1),
  C1–C68. [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097).
- Wager, S. & Athey, S. (2018). Estimation and inference of heterogeneous
  treatment effects using random forests. *JASA* 113(523), 1228–1242.
  [doi:10.1080/01621459.2017.1319839](https://doi.org/10.1080/01621459.2017.1319839).

### Applications

- Odion, D. et al. (2025). [Towards improving farmers livelihood in Nigeria
  using food price forecasting](https://www.research-collection.ethz.ch/entities/publication/166363fb-ddab-4df1-9d7e-a3109666acdb).
  *Journal of Agriculture and Food Research* 24, 102365. Main resource for
  the challenge's "resource from the last 2 years" requirement.
- Zhao, C. et al. (2025). [A Vegetable-Price Forecasting Method Based on
  Mixture of Experts](https://doaj.org/article/f8126c607ff5463586fb461e3522bac1).
  *Agriculture* 15(2), 162.
- Runyiri, J. (2025). [A Predictive model for hedging futures contracts to
  stabilize Kenyan coffee farmers' income](https://su-plus.strathmore.edu/items/78ca6dd5-17d3-4583-a579-36b146128803).
  Dissertation, Strathmore University.
- [Prototype of AI-Integrated Chatbot for Shallot Price Forecasting and
  Advisory Support to Assist Farmer Decision Making](https://journal.ipb.ac.id/jtep/article/view/69862).
  *Jurnal Keteknikan Pertanian* (IPB). Authors and year still to confirm.
- Stracqualursi, L. (2026). [Forecasting Crop Yield Anomalies on Panel Data
  via Spatially Demeaned Ensembles](https://doi.org/10.1007/s13253-026-00743-8).
  *Journal of Agricultural, Biological and Environmental Statistics*. Open
  access (CC BY 4.0). Main methodological reference for yield: forecasts
  anomalies after removing each region's fixed level (municipality fixed
  effects, in PAM terms), with a light RF/XGBoost/SVR ensemble and split
  conformal intervals. Country-level global panel (1990–2013), not
  municipalities.
- de Freitas, C. H., Coelho, R. D., Costa, J. O. & Sentelhas, P. C. (2025).
  [Equationing Arabica coffee: Adaptation, calibration, and application of an
  agrometeorological model for yield estimation](https://doi.org/10.1016/j.agsy.2024.104181).
  *Agricultural Systems* 223, 104181. Domain reference for arábica in
  Brazil: process-based baseline and which climate variables matter at each
  crop stage.
- Quantile regression forest for corn yield across 1,260 Chinese counties,
  1984–2019. *International Food and Agribusiness Management Review* 28(2),
  2025. [doi:10.22434/ifamr1134](https://doi.org/10.22434/ifamr1134). The
  closest setup to ours: a sub-national panel with probabilistic intervals.
  Authors still to confirm.
- Farag, M., Emam, A., Leonhardt, J. & Roscher, R. (2025). Enhancing decision
  support in crop production: Analyzing conformal prediction for uncertainty
  quantification. *Computers and Electronics in Agriculture* 237, 110559.
  [doi:10.1016/j.compag.2025.110559](https://doi.org/10.1016/j.compag.2025.110559).
  Conformal prediction in agriculture in a strong venue; probably not yield
  forecasting, so check the abstract before citing.
- Namdeo, D. & Baraskar, R. (2026). Climate-Adaptive Transformer for Crop
  Yield Prediction with Conformal Uncertainty Quantification. *IJERT*
  15(04), IJERTV15IS040092. Same pipeline idea (crop-stage climate features +
  conformal + ACI), cited only as an example: low-rigour venue, inconsistent
  dataset counts, and its satellite features are proxies derived from climate
  data. Prefer the two references above.

### Context: farmer wellbeing and risk in Brazil

These support the motivation, not the method.

- Lameiras, M. M. (2025). [The cost of stress: How financial planning could
  save lives on farms](https://fieldreport.caes.uga.edu/news/financial-planning-relieves-farm-stress/).
  *CAES Field Report*, University of Georgia, 18 Dec 2025. Reports a survey of
  310 US farmers by A. Scheyett et al.: disaster assistance and government
  programmes were rated the most helpful topics for reducing stress. The
  survey measured perceived helpfulness, not a reduction in stress.
- Norco & Black Dog Institute (2026). [2026 Norco National Farmer Wellbeing
  Report](https://drive.google.com/file/d/1wzMNIvjTvrTnIjXAX0QRdJBcdsGZyFy_/view).
  Survey of 1,017 Australian producers, Oct 2025 – Mar 2026. Industry
  survey, not peer-reviewed.
- Nichi, J. (2026). [Comida cara: o que o recorde da safra não
  mostra?](https://outraspalavras.net/crise-brasileira/comida-cara-o-que-o-recorde-da-safra-nao-mostra/)
  *Outras Palavras*, 14 Aug 2026. Insurance gap (subsidised cover under 3% of
  planted area in 2026/27) and the limits of credit-only policy for family
  farmers. For its numbers, cite Conab, IBGE and MAPA directly.

### Still to find

- The meta-analysis behind the 0.06 SMD effect of price-information
  interventions (cited in the project glossary without a source).
- Abstracts and full text of de Freitas et al. (2025) and Farag et al.
  (2025): only their metadata has been checked so far.
