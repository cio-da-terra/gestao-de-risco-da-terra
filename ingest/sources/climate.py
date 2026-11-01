"""Weather and climate.

- NASA POWER: daily point data, no API key. Good for the prototype.
- NOAA CPC ONI: monthly ENSO index.
- CHIRPS / ERA5: stubs. They are gridded (bigger downloads, extra libs:
  rasterio/xarray, and cdsapi + a free Copernicus account for ERA5).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from .. import db
from ..http import get

POWER_PARAMS = {
    "T2M_MAX": ("Temp. máxima 2 m", "°C"),
    "T2M_MIN": ("Temp. mínima 2 m", "°C"),
    "PRECTOTCORR": ("Precipitação corrigida", "mm/dia"),
    "RH2M": ("Umidade relativa 2 m", "%"),
}
POWER_LAG = timedelta(days=4)  # POWER's near-real-time data lags a few days


def fetch_nasa_power(con, name: str, lat: float, lon: float,
                     start: date | None = None, end: date | None = None) -> int:
    end = end or date.today() - timedelta(days=1)
    probe = f"power:{name}:T2M_MAX"
    last = db.last_ref_date(con, probe)
    start = start or (last - timedelta(days=15) if last else date(1990, 1, 1))
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters={','.join(POWER_PARAMS)}&community=AG"
           f"&latitude={lat}&longitude={lon}"
           f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON")
    payload = get(url, timeout=300)
    snap = db.save_raw(con, "nasa_power", url, payload, ".json")
    params = json.loads(payload)["properties"]["parameter"]
    total = 0
    for p, (desc, unit) in POWER_PARAMS.items():
        sid = f"power:{name}:{p}"
        db.register_series(con, sid, "nasa_power", f"{desc} ({name})", unit, "daily")
        rows = [db.Obs(sid, datetime.strptime(k, "%Y%m%d").date(), None if v == -999 else v)
                for k, v in params.get(p, {}).items()]
        total += db.upsert_observations(con, rows, POWER_LAG, snap)
    return total


ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
# 3-month season -> its centre month
_SEASON_MID = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
               "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def fetch_oni(con) -> int:
    payload = get(ONI_URL)
    snap = db.save_raw(con, "noaa_oni", ONI_URL, payload, ".txt")
    sid = "noaa:oni"
    db.register_series(con, sid, "noaa_cpc", "Oceanic Niño Index (3-mo running anomaly)", "°C", "monthly")
    rows = []
    for line in payload.decode().splitlines()[1:]:
        parts = line.split()
        if len(parts) != 4:
            continue
        seas, yr, _total, anom = parts
        rows.append(db.Obs(sid, date(int(yr), _SEASON_MID[seas], 1), float(anom)))
    # ONI for a season is published ~early in the month after it ends
    return db.upsert_observations(con, rows, timedelta(days=70), snap)


def fetch_chirps(con, *args, **kwargs) -> int:
    """TODO: download monthly/daily CHIRPS GeoTIFFs from
    https://data.chc.ucsb.edu/products/CHIRPS-2.0/ and average over each
    municipality polygon (geobr + rasterio). Preliminary data is revised
    ~3 weeks later: store both, the db keeps the revision history."""
    raise NotImplementedError


def fetch_era5(con, *args, **kwargs) -> int:
    """TODO: ERA5-Land via cdsapi (needs ~/.cdsapirc). Pull t2m hourly or
    daily stats for a bounding box, aggregate to municipality, then derive
    GDD/EDD in a features step (not here: ingestion stores raw variables)."""
    raise NotImplementedError
