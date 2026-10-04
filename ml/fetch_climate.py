"""Step 1 – download monthly climate (temperature + rainfall) for every state.

Source : NASA POWER monthly API (free, no API key)
Output : data/raw/climate_monthly.csv   [state, year, month, temp_c, precip_mm_day]

Run from the project root:   python ml/fetch_climate.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from states import STATES  # noqa: E402

START_YEAR = 1990
END_YEAR = 2025          # change if you want a different range
URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
CACHE = Path("data/raw/power_cache")
OUT = Path("data/raw/climate_monthly.csv")


def fetch_state(name, lat, lon):
    cache = CACHE / f"{name.replace(' ', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON",
    }
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data))
            return data
        except Exception as exc:  # network hiccup / throttling
            print(f"   retry {attempt + 1} for {name}: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not download {name}")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, (lat, lon, *_)) in enumerate(STATES.items(), 1):
        print(f"[{i}/{len(STATES)}] {name}")
        par = fetch_state(name, lat, lon)["properties"]["parameter"]
        for key, t in par["T2M"].items():
            year, month = int(key[:4]), int(key[4:])
            if month == 13:          # POWER stores the annual mean as month 13
                continue
            p = par["PRECTOTCORR"].get(key)
            rows.append((name, year, month,
                         None if t == -999 else t,
                         None if p is None or p == -999 else p))
        time.sleep(1)
    df = pd.DataFrame(rows, columns=["state", "year", "month", "temp_c", "precip_mm_day"])
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT}")
    print(df.groupby('state').size().describe()[["min", "max"]])


if __name__ == "__main__":
    main()
