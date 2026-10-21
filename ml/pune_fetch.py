"""Step 1 (Pune) - download REAL daily weather for every Pune area.

Output : data/raw/pune/<area_id>.csv  [date, tmax, tmin, rain, elevation]

Sources (tries in this order, automatically):
  1. Open-Meteo (ERA5 reanalysis, ~25 km)  - better, but its free API often rate-limits
  2. NASA POWER (MERRA-2/IMERG, ~50 km)    - fallback, you already use it in fetch_climate.py

Run from project root:
    python ml/pune_fetch.py          # auto (Open-Meteo, then NASA if blocked)
    python ml/pune_fetch.py nasa     # force NASA POWER
Resumable: finished areas are skipped. Delete data/raw/pune to start fresh.
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from pune_areas import AREAS  # noqa: E402

START_YEAR = 2000
END_DATE = date.today() - timedelta(days=7)   # archives lag a few days
OUT = Path("data/raw/pune")

OM_URL = "https://archive-api.open-meteo.com/v1/archive"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


class Blocked(Exception):
    pass


def clean(df):
    """Make the series gap-free (the model needs consecutive days)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    for c in ("tmax", "tmin"):
        df[c] = df[c].interpolate(limit=7).ffill().bfill()
    df["rain"] = df["rain"].fillna(0.0).clip(lower=0)
    df.index.name = "date"
    return df.reset_index()


# ── Open-Meteo ───────────────────────────────────────────────────

def om_json(params):
    for attempt in range(3):
        try:
            r = requests.get(OM_URL, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"   network error: {e}")
            time.sleep(5)
            continue
        if r.status_code == 429:
            try:
                reason = r.json().get("reason", "")
            except ValueError:
                reason = ""
            print(f"   Open-Meteo rate limit: {reason or '429'}")
            if attempt < 2:
                print("   waiting 65s ...")
                time.sleep(65)
            continue
        r.raise_for_status()
        return r.json()
    raise Blocked("Open-Meteo is rate-limiting you")


def fetch_open_meteo(a):
    frames, elevation = [], None
    for y in range(START_YEAR, END_DATE.year + 1, 5):
        start = date(y, 1, 1)
        end = min(date(min(y + 4, END_DATE.year), 12, 31), END_DATE)
        j = om_json({
            "latitude": a["lat"], "longitude": a["lon"],
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Kolkata",
        })
        d = j["daily"]
        frames.append(pd.DataFrame({
            "date": d["time"], "tmax": d["temperature_2m_max"],
            "tmin": d["temperature_2m_min"], "rain": d["precipitation_sum"],
        }))
        elevation = j.get("elevation", elevation)
        time.sleep(2)
    return pd.concat(frames, ignore_index=True), elevation


# ── NASA POWER ───────────────────────────────────────────────────

def parse_nasa(j):
    p = j["properties"]["parameter"]
    df = pd.DataFrame({
        "date": list(p["T2M_MAX"].keys()),
        "tmax": list(p["T2M_MAX"].values()),
        "tmin": list(p["T2M_MIN"].values()),
        "rain": list(p["PRECTOTCORR"].values()),
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.replace(-999, np.nan)
    elevation = j.get("geometry", {}).get("coordinates", [None, None, None])[2]
    return df, elevation


def fetch_nasa(a):
    frames, elevation = [], None
    for y in range(START_YEAR, END_DATE.year + 1, 10):
        end = min(date(min(y + 9, END_DATE.year), 12, 31), END_DATE)
        params = {
            "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR", "community": "RE",
            "latitude": a["lat"], "longitude": a["lon"],
            "start": f"{y}0101", "end": end.strftime("%Y%m%d"), "format": "JSON",
        }
        for attempt in range(4):
            try:
                r = requests.get(NASA_URL, params=params, timeout=120)
                r.raise_for_status()
                df, elev = parse_nasa(r.json())
                frames.append(df)
                elevation = elev if elev is not None else elevation
                break
            except (requests.RequestException, KeyError, ValueError) as e:
                print(f"   NASA retry {attempt + 1}: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError("NASA POWER download failed - check internet")
        time.sleep(2)
    return pd.concat(frames, ignore_index=True), elevation


# ── main ─────────────────────────────────────────────────────────

def main():
    source = "nasa" if (len(sys.argv) > 1 and sys.argv[1].lower() == "nasa") else "open-meteo"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Source: {source}")
    for i, a in enumerate(AREAS, 1):
        path = OUT / f"{a['id']}.csv"
        if path.exists():
            print(f"[{i}/{len(AREAS)}] {a['name']}: already downloaded")
            continue
        print(f"[{i}/{len(AREAS)}] {a['name']} ...")
        if source == "open-meteo":
            try:
                raw, elev = fetch_open_meteo(a)
            except Blocked as e:
                print(f"\n>>> {e}. Switching to NASA POWER for all areas.\n")
                source = "nasa"
        if source == "nasa":
            raw, elev = fetch_nasa(a)
        df = clean(raw)
        df["elevation"] = elev if elev is not None else 560.0
        df.to_csv(path, index=False)
    print(f"Done (source: {source}). Next: python ml/pune_build.py")


if __name__ == "__main__":
    main()