"""Step 0 (Pune) - compute flood-exposure index from REAL geographic data.

Inputs (all downloaded live, free, no API key):
  * OpenStreetMap waterways (rivers / streams / canals) via the Overpass API
  * SRTM 30 m elevation via OpenTopoData (falls back to Open-Meteo elevation)

For every area in pune_areas.py it measures
  river_dist_km        distance to the nearest river/stream/canal
  height_above_river_m area elevation minus elevation of that nearest waterway point
and combines them into one 0-1 index (1 = low and right next to water):
  exposure = 0.5 * (1 - dist/2 km) + 0.5 * (1 - height/25 m)      (each part clipped to 0-1)

This is a physical susceptibility index built from measured inputs. It is NOT an
official flood-inundation map. The 2 km / 25 m / 50-50 weights are modelling choices.

Run from project root:   python ml/pune_exposure.py
Output: ml/pune_exposure.json  (pune_areas.py picks it up automatically)
Then:   python ml/pune_build.py
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
from pune_areas import AREAS  # noqa: E402

OUT = Path(__file__).with_name("pune_exposure.json")
S, W, N, E = 18.35, 73.65, 18.75, 74.05           # bounding box around Pune + PCMC
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "pune-climate-early-warning-student-project/1.0"}
CACHE = Path("data/raw/pune_waterways.json")      # saved once, reused on re-runs
QUERY = (f'[out:json][timeout:120];way["waterway"~"^(river|stream|canal)$"]'
         f"({S},{W},{N},{E});out geom;")
TOPO = "https://api.opentopodata.org/v1/srtm30m"
OM_ELEV = "https://api.open-meteo.com/v1/elevation"
DIST_SCALE_KM = 2.0
HEIGHT_SCALE_M = 25.0


def hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def fetch_waterways():
    if CACHE.exists():
        print(f"  using saved file {CACHE}")
        return json.loads(CACHE.read_text())["elements"]
    last = None
    for url in OVERPASS:
        for attempt in range(2):
            try:
                r = requests.post(url, data={"data": QUERY}, headers=HEADERS, timeout=150)
                if r.status_code in (429, 504):
                    last = f"HTTP {r.status_code} from {url.split('/')[2]}"
                    print(f"   {last} - waiting 30s ...")
                    time.sleep(30)
                    continue
                r.raise_for_status()
                data = r.json()
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(data))
                return data["elements"]
            except (requests.RequestException, ValueError, KeyError) as e:
                last = f"{e.__class__.__name__} from {url.split('/')[2]}"
                print(f"   {last} - trying next ...")
                time.sleep(5)
    sys.exit(
        f"\nCould not download waterways ({last}). The servers are blocking your network.\n"
        "MANUAL FIX (2 minutes):\n"
        "  1. Open https://overpass-turbo.eu in your browser (phone hotspot also works)\n"
        "  2. Paste this query, click Run:\n"
        f"     {QUERY}\n"
        "  3. Export > Data > 'download as raw OSM data' (.json)\n"
        f"  4. Save that file as {CACHE} (create the data/raw folder if needed)\n"
        "  5. Run this script again.\n")


def river_points(elements, step_m=50):
    """Waterway nodes, densified to roughly one point every 50 m."""
    pts = []
    for el in elements:
        g = el.get("geometry") or []
        for p, q in zip(g, g[1:]):
            d_m = hav_km(p["lat"], p["lon"], q["lat"], q["lon"]) * 1000
            n = max(1, int(d_m // step_m))
            for t in np.linspace(0, 1, n, endpoint=False):
                pts.append((p["lat"] + (q["lat"] - p["lat"]) * t,
                            p["lon"] + (q["lon"] - p["lon"]) * t))
        if g:
            pts.append((g[-1]["lat"], g[-1]["lon"]))
    return np.array(pts)


def elev_chunk(chunk):
    loc = "|".join(f"{la:.5f},{lo:.5f}" for la, lo in chunk)
    for attempt in range(3):
        try:
            r = requests.get(TOPO, params={"locations": loc}, headers=HEADERS, timeout=60)
            r.raise_for_status()
            vals = [x["elevation"] for x in r.json()["results"]]
            if all(v is not None for v in vals):
                return [float(v) for v in vals]
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(3 * (attempt + 1))
    try:                                           # fallback source
        r = requests.get(OM_ELEV, params={
            "latitude": ",".join(f"{la:.5f}" for la, _ in chunk),
            "longitude": ",".join(f"{lo:.5f}" for _, lo in chunk)}, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return [float(v) for v in r.json()["elevation"]]
    except (requests.RequestException, ValueError, KeyError):
        sys.exit("Could not download elevation data. Try again in a few minutes.")


def elevations(points):
    out = []
    for i in range(0, len(points), 100):
        out += elev_chunk(points[i:i + 100])
        time.sleep(1.2)
    return out


def index(dist_km, height_m):
    prox = min(max(1 - dist_km / DIST_SCALE_KM, 0.0), 1.0)
    low = min(max(1 - height_m / HEIGHT_SCALE_M, 0.0), 1.0)
    return round(0.5 * prox + 0.5 * low, 2)


def nearest(areas, rpts):
    res = []
    for a in areas:
        d = hav_km(a["lat"], a["lon"], rpts[:, 0], rpts[:, 1])
        i = int(np.argmin(d))
        res.append((float(d[i]), (float(rpts[i, 0]), float(rpts[i, 1]))))
    return res


def build(areas, near, area_elev, river_elev):
    out = {}
    for a, (dist, _), ea, er in zip(areas, near, area_elev, river_elev):
        h = max(0.0, ea - er)
        out[a["id"]] = {"exposure": index(dist, h), "river_dist_km": round(dist, 2),
                        "height_above_river_m": round(h, 1)}
    return out


def main():
    print("Downloading OpenStreetMap waterways ...")
    rpts = river_points(fetch_waterways())
    if len(rpts) == 0:
        sys.exit("No waterways returned - try again later.")
    print(f"  {len(rpts):,} waterway points")
    near = nearest(AREAS, rpts)
    print("Downloading SRTM elevation ...")
    pts = [(a["lat"], a["lon"]) for a in AREAS] + [p for _, p in near]
    el = elevations(pts)
    res = build(AREAS, near, el[:len(AREAS)], el[len(AREAS):])
    res["_meta"] = {
        "method": ("Index from distance to nearest river/stream/canal (OpenStreetMap) and "
                   "height above it (SRTM 30 m). Not an official flood map."),
        "computed_on": date.today().isoformat(),
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(f"{'area':20s} dist_km  height_m  exposure")
    for a in AREAS:
        r = res[a["id"]]
        print(f"{a['name']:20s} {r['river_dist_km']:7.2f} {r['height_above_river_m']:9.1f} {r['exposure']:9.2f}")
    print(f"Wrote {OUT}. Next: python ml/pune_build.py")


if __name__ == "__main__":
    main()