"""Step 1 (Maharashtra) - download REAL daily climate data, 2001-2025, for every district.

Source : NASA POWER daily API (MERRA-2 / IMERG reanalysis+satellite, free, no key)
         https://power.larc.nasa.gov
Output : data/raw/mh/<district_id>.csv   [date, tmax, tmin, rain, soil_wet]
           tmax/tmin  = T2M_MAX / T2M_MIN  (deg C)
           rain       = PRECTOTCORR         (mm/day)
           soil_wet   = GWETROOT            (root-zone soil wetness, 0-1)  -> water stress

Run from project root:   python ml/mh_fetch.py
Resumable: finished districts are skipped. Delete data/raw/mh to start fresh.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from mh_districts import DISTRICTS  # noqa: E402

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
OUT = Path("data/raw/mh")
CHUNKS = [(2001, 2009), (2010, 2018), (2019, 2025)]
PARAMS_FULL = "T2M_MAX,T2M_MIN,PRECTOTCORR,GWETROOT"
PARAMS_BASIC = "T2M_MAX,T2M_MIN,PRECTOTCORR"


def parse(j):
    p = j["properties"]["parameter"]
    df = pd.DataFrame({
        "date": pd.to_datetime(list(p["T2M_MAX"].keys()), format="%Y%m%d"),
        "tmax": list(p["T2M_MAX"].values()),
        "tmin": list(p["T2M_MIN"].values()),
        "rain": list(p["PRECTOTCORR"].values()),
        "soil_wet": list(p["GWETROOT"].values()) if "GWETROOT" in p else np.nan,
    })
    return df.replace(-999, np.nan).replace(-999.0, np.nan)


def get_chunk(d, y0, y1, params):
    q = {
        "parameters": params, "community": "AG",
        "latitude": d["lat"], "longitude": d["lon"],
        "start": f"{y0}0101", "end": f"{y1}1231", "format": "JSON",
    }
    last = None
    for attempt in range(4):
        try:
            r = requests.get(URL, params=q, timeout=180)
            if r.status_code == 422 and params == PARAMS_FULL:
                return None   # parameter not accepted -> caller retries without soil wetness
            r.raise_for_status()
            return parse(r.json())
        except (requests.RequestException, KeyError, ValueError) as e:
            last = e
            print(f"   retry {attempt + 1}: {e}")
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"NASA POWER download failed for {d['name']}: {last}")


def fetch_district(d):
    frames = []
    for y0, y1 in CHUNKS:
        df = get_chunk(d, y0, y1, PARAMS_FULL)
        if df is None:
            print("   soil wetness (GWETROOT) not available -> continuing without it")
            df = get_chunk(d, y0, y1, PARAMS_BASIC)
        frames.append(df)
        time.sleep(1.5)
    return pd.concat(frames, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(DISTRICTS, 1):
        path = OUT / f"{d['id']}.csv"
        if path.exists():
            print(f"[{i}/{len(DISTRICTS)}] {d['name']}: already downloaded")
            continue
        print(f"[{i}/{len(DISTRICTS)}] {d['name']} ...")
        df = fetch_district(d)
        df.to_csv(path, index=False)
    print("Done. Next: python ml/mh_build.py")


if __name__ == "__main__":
    main()
