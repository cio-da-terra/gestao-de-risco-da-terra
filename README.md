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

This README covers the **data ingestion** layer: the code that fetches
public data on a schedule and stores it so models can be trained and
backtested honestly.

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
