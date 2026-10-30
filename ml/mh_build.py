"""Step 2 (Maharashtra) - climate indicators, trends and census migration.

Input : data/raw/mh/<district>.csv                 (from mh_fetch.py, NASA POWER daily 2001-2025)
        data/raw/census/migration_district.csv     (Census 2001 + 2011, you fill it - see README)  [optional]
        data/raw/events.csv                        (EM-DAT, from convert_emdat.py)                  [optional]
Output: public/data/maharashtra.json               (read by src/MaharashtraClimate.tsx)

Indicators per district and year (nothing is typed in by hand - all computed from the downloaded data)
  temperature  : mean / max / min temperature
  rainfall     : annual, monsoon (Jun-Sep), monsoon departure from the district's own 2001-2025 mean
  heatwave     : IMD-style - Tmax >= 40 C (37 C coastal) AND >= 4.5 C above that calendar day's normal,
                 or Tmax >= 45 C; only spells of >= 2 consecutive days are counted
  flood proxy  : days with rain >= 64.5 mm (IMD 'heavy') and >= 115.6 mm ('very heavy'), Rx1day, Rx5day
  drought      : monsoon rain <= -25 % of mean = drought year, <= -50 % = severe (IMD meteorological drought);
                 longest dry spell in the monsoon (days with < 2.5 mm)
  water stress : pre-monsoon (Mar-May) root-zone soil wetness, % of days below the district's own
                 20th-percentile wetness, and a composite 0-100 index (low monsoon rain + dry soil + high Tmax,
                 each ranked within the district's own 25 years)
Trends: Theil-Sen slope per decade + Mann-Kendall (Kendall tau) p-value over 2001-2025,
        and mean of 2001-2010 vs mean of 2016-2025.

Run from project root:   python ml/mh_build.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from mh_districts import DISTRICTS, census_lookup, norm  # noqa: E402

RAW = Path("data/raw/mh")
CENSUS = Path("data/raw/census/migration_district.csv")
EVENTS = Path("data/raw/events.csv")
OUT = Path("public/data/maharashtra.json")

Y0, Y1 = 2001, 2025
MONSOON = (6, 7, 8, 9)
WET_DAY, HEAVY, VHEAVY = 2.5, 64.5, 115.6
HW_ANOM, HW_PLAINS, HW_COAST, HW_SEVERE = 4.5, 40.0, 37.0, 45.0
DROUGHT_DEP, SEVERE_DEP = -25.0, -50.0
MIN_COVERAGE = 0.98

TREND_METRICS = {
    "tmean": ("Mean temperature", "°C"),
    "tmax": ("Mean max temperature", "°C"),
    "tmin": ("Mean min temperature", "°C"),
    "rain": ("Annual rainfall", "mm"),
    "monsoon_rain": ("Monsoon rainfall", "mm"),
    "heatwave_days": ("Heatwave days", "days"),
    "heavy_days": ("Heavy-rain days (≥64.5 mm)", "days"),
    "max_dry_spell": ("Longest monsoon dry spell", "days"),
    "wsi": ("Water-stress index", "0–100"),
}


# ── loading ───────────────────────────────────────────────────────
def load_district(d):
    p = RAW / f"{d['id']}.csv"
    if not p.exists():
        sys.exit(f"Missing {p}\nRun first:  python ml/mh_fetch.py")
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    df = df[(df.index.year >= Y0) & (df.index.year <= Y1)]
    full = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D")
    cover = df[["tmax", "tmin", "rain"]].dropna().shape[0] / len(full)
    if cover < MIN_COVERAGE:
        sys.exit(f"{d['name']}: only {cover:.1%} of days {Y0}-{Y1} have data. Delete {p} and re-run mh_fetch.py")
    df = df.reindex(full)
    for c in ("tmax", "tmin", "soil_wet"):
        if c in df:
            df[c] = df[c].interpolate(limit=7)
    df["rain"] = df["rain"].fillna(0.0).clip(lower=0)
    return df


def tmax_normal(df):
    """Calendar-day normal of Tmax (2001-2025 mean, 15-day circular smoothing)."""
    doy = np.minimum(df.index.dayofyear, 365)
    clim = df["tmax"].groupby(doy).mean().reindex(range(1, 366)).interpolate()
    a, k = clim.to_numpy(), 7
    ext = np.concatenate([a[-k:], a, a[:k]])
    sm = np.convolve(ext, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return pd.Series(sm, index=range(1, 366)), doy


def longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


# ── indicators ────────────────────────────────────────────────────
def annual_table(d, df):
    yr = df.index.year
    normal, doy = tmax_normal(df)
    anom = df["tmax"] - normal.reindex(doy).to_numpy()
    thr = HW_COAST if d["coastal"] else HW_PLAINS
    cond = (((df["tmax"] >= thr) & (anom >= HW_ANOM)) | (df["tmax"] >= HW_SEVERE)).fillna(False)
    grp = (cond != cond.shift()).cumsum()
    hw = cond & (cond.groupby(grp).transform("size") >= 2)
    spells = hw & ~hw.shift(fill_value=False)

    a = pd.DataFrame(index=range(Y0, Y1 + 1))
    a["tmean"] = ((df["tmax"] + df["tmin"]) / 2).groupby(yr).mean()
    a["tmax"] = df["tmax"].groupby(yr).mean()
    a["tmin"] = df["tmin"].groupby(yr).mean()
    a["rain"] = df["rain"].groupby(yr).sum()
    mon = df[df.index.month.isin(MONSOON)]
    a["monsoon_rain"] = mon["rain"].groupby(mon.index.year).sum()
    a["monsoon_dep"] = (a["monsoon_rain"] - a["monsoon_rain"].mean()) / a["monsoon_rain"].mean() * 100
    a["heatwave_days"] = hw.groupby(yr).sum()
    a["heatwave_spells"] = spells.groupby(yr).sum()
    a["tmax_peak"] = df["tmax"].groupby(yr).max()
    a["heavy_days"] = (df["rain"] >= HEAVY).groupby(yr).sum()
    a["vheavy_days"] = (df["rain"] >= VHEAVY).groupby(yr).sum()
    a["rx1"] = df["rain"].groupby(yr).max()
    a["rx5"] = df["rain"].rolling(5).sum().groupby(yr).max()
    a["max_dry_spell"] = pd.Series(
        {y: longest_run((g["rain"] < WET_DAY).to_numpy()) for y, g in mon.groupby(mon.index.year)}
    )
    dep = a["monsoon_dep"]
    a["monsoon_class"] = np.select(
        [dep <= SEVERE_DEP, dep <= DROUGHT_DEP, dep >= 25.0],
        ["severe drought", "drought", "excess"], "normal",
    )

    comps = [(-a["monsoon_rain"]).rank(pct=True), a["tmax"].rank(pct=True)]
    has_soil = "soil_wet" in df and df["soil_wet"].notna().mean() > 0.9
    if has_soil:
        pre = df[df.index.month.isin((3, 4, 5))]
        a["soil_premonsoon"] = pre["soil_wet"].groupby(pre.index.year).mean()
        p20 = df["soil_wet"].quantile(0.20)
        a["dry_soil_pct"] = (df["soil_wet"] < p20).groupby(yr).mean() * 100
        comps.append((-a["soil_premonsoon"]).rank(pct=True))
    a["wsi"] = pd.concat(comps, axis=1).mean(axis=1) * 100
    return a, has_soil


def monthly_climatology(df):
    m = df.index.month
    out = []
    for k in range(1, 13):
        s = df[m == k]
        yearly_rain = s["rain"].groupby(s.index.year).sum()
        out.append({
            "m": k,
            "tmax": s["tmax"].mean(), "tmin": s["tmin"].mean(),
            "rain": yearly_rain.mean(),
        })
    return out


def trend(series: pd.Series):
    s = series.dropna()
    if len(s) < 10 or s.nunique() < 3:
        return None
    x, y = s.index.to_numpy(dtype=float), s.to_numpy(dtype=float)
    slope = stats.theilslopes(y, x)[0]
    _, p = stats.kendalltau(x, y)
    first = s[(s.index >= Y0) & (s.index <= Y0 + 9)].mean()
    last = s[(s.index >= Y1 - 9) & (s.index <= Y1)].mean()
    return {
        "slope_decade": slope * 10, "p": None if np.isnan(p) else p,
        "first": first, "last": last, "change": last - first,
    }


def trends_of(a):
    return {k: trend(a[k]) for k in TREND_METRICS if k in a}


# ── disasters (EM-DAT, optional) ──────────────────────────────────
def load_events():
    if not EVENTS.exists():
        return None
    ev = pd.read_csv(EVENTS)
    ev = ev[ev["state"].astype(str).str.lower().str.contains("maharashtra")]
    ev = ev[(ev.year >= Y0) & (ev.year <= Y1)]
    if ev.empty:
        return None
    g = ev.groupby(["year", "hazard"]).agg(
        events=("hazard", "size"), deaths=("deaths", "sum"), affected=("affected", "sum")
    ).reset_index()
    return g.to_dict("records")


# ── migration (Census 2001 + 2011) ────────────────────────────────
REQUIRED = ["district", "year", "total_population", "total_migrants"]
OPTIONAL = ["migrants_intra_district", "migrants_inter_district", "migrants_inter_state", "migrants_outside_india"]


def load_migration(district_out):
    if not CENSUS.exists():
        return None, f"{CENSUS} not found"
    df = pd.read_csv(CENSUS)
    df.columns = [c.strip().lower() for c in df.columns]
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"{CENSUS}: missing columns {miss}")
    df = df.dropna(subset=["district", "year"])
    if df.empty:
        return None, f"{CENSUS} has no data rows yet"
    lut = census_lookup()
    df["cid"] = df["district"].map(lambda s: lut.get(norm(s)))
    bad = df[df["cid"].isna()]["district"].unique()
    if len(bad):
        sys.exit(f"{CENSUS}: unknown district names: {list(bad)}\nUse Census spellings (e.g. Aurangabad, Osmanabad, Buldana, Gondiya, Raigarh).")
    df = df[df["year"].isin([2001, 2011])]
    for c in REQUIRED[2:] + [c for c in OPTIONAL if c in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.duplicated(["cid", "year"]).any():
        sys.exit(f"{CENSUS}: duplicate district+year rows")

    names = {d["id"]: d["name"] for d in DISTRICTS}
    names["thane"] = "Thane (incl. Palghar)"
    rows = []
    for cid, g in df.groupby("cid"):
        g = g.set_index("year")
        if not {2001, 2011} <= set(g.index):
            print(f"  note: {names[cid]} lacks 2001 or 2011 row - skipped")
            continue
        r = {"id": cid, "name": names[cid]}
        for y in (2001, 2011):
            pop, mig = g.loc[y, "total_population"], g.loc[y, "total_migrants"]
            r[f"pop_{y}"] = pop
            r[f"migrants_{y}"] = mig
            r[f"share_{y}"] = mig / pop * 100
            for c in OPTIONAL:
                if c in g and pd.notna(g.loc[y, c]) and mig:
                    r[f"{c[10:]}_pct_{y}"] = g.loc[y, c] / mig * 100
        r["share_change_pp"] = r["share_2011"] - r["share_2001"]
        r["pop_growth_pct"] = (r["pop_2011"] / r["pop_2001"] - 1) * 100
        r["migrant_growth_pct"] = (r["migrants_2011"] / r["migrants_2001"] - 1) * 100
        rows.append(r)
    if not rows:
        return None, f"{CENSUS}: no district has both 2001 and 2011"

    # climate stress in the decade BEFORE the 2011 census (2001-2010) vs. migration change
    by_id = {d["id"]: d for d in district_out}
    for r in rows:
        a = pd.DataFrame(by_id[r["id"]]["annual"]).set_index("year").loc[2001:2010]
        r["drought_years_2001_10"] = int(a["monsoon_class"].isin(["drought", "severe drought"]).sum())
        r["wsi_2001_10"] = a["wsi"].mean()
        r["heatwave_days_2001_10"] = a["heatwave_days"].mean()
    tab = pd.DataFrame(rows)
    corr = []
    for x, lab in [("drought_years_2001_10", "Drought years 2001–10"),
                   ("wsi_2001_10", "Mean water-stress index 2001–10"),
                   ("heatwave_days_2001_10", "Mean heatwave days/yr 2001–10")]:
        for y, ylab in [("share_change_pp", "Change in migrant share 2001→2011 (pp)"),
                        ("pop_growth_pct", "Population growth 2001→2011 (%)")]:
            if len(tab) >= 10 and tab[x].nunique() > 2:
                rho, p = stats.spearmanr(tab[x], tab[y])
                corr.append({"x": lab, "y": ylab, "rho": rho, "p": p, "n": len(tab)})
    state = {}
    for y in (2001, 2011):
        state[f"pop_{y}"] = tab[f"pop_{y}"].sum()
        state[f"migrants_{y}"] = tab[f"migrants_{y}"].sum()
        state[f"share_{y}"] = state[f"migrants_{y}"] / state[f"pop_{y}"] * 100
    return {"rows": rows, "correlations": corr, "state": state,
            "optional_columns": [c for c in OPTIONAL if c in df.columns]}, None


# ── json helpers ──────────────────────────────────────────────────
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 3)
    return o


def records(a):
    t = a.copy()
    t.index.name = "year"
    return t.reset_index().to_dict("records")


def main():
    districts, annuals, soil_flags = [], {}, []
    for d in DISTRICTS:
        df = load_district(d)
        a, has_soil = annual_table(d, df)
        soil_flags.append(has_soil)
        annuals[d["id"]] = a
        districts.append({
            "id": d["id"], "name": d["name"], "region": d["region"],
            "lat": d["lat"], "lon": d["lon"], "coastal": d["coastal"],
            "annual": records(a), "monthly": monthly_climatology(df),
            "trends": trends_of(a),
            "drought_years": int(a["monsoon_class"].isin(["drought", "severe drought"]).sum()),
        })
        print(f"  {d['name']}: ok")

    num_cols = [c for c in annuals["pune"].columns if c != "monsoon_class"]
    state_a = sum(annuals[i][num_cols] for i in annuals) / len(annuals)
    state_a["drought_districts"] = sum(
        (annuals[i]["monsoon_class"].isin(["drought", "severe drought"])).astype(int) for i in annuals
    )

    migration, mig_msg = load_migration(districts)
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "period": [Y0, Y1],
        "n_districts": len(districts),
        "climate_source": "NASA POWER daily (MERRA-2 / IMERG), district-HQ grid cell, 2001–2025",
        "soil_wetness": all(soil_flags),
        "trend_method": "Theil–Sen slope per decade; Mann–Kendall (Kendall tau) p-value; 2001–10 vs 2016–25 mean",
        "metrics": {k: {"label": v[0], "unit": v[1]} for k, v in TREND_METRICS.items()},
        "census_source": "Census of India 2001 & 2011, migration tables D-series (user-supplied CSV)" if migration else None,
        "migration_message": mig_msg,
    }
    out = {
        "meta": meta, "districts": districts,
        "state": {"annual": records(state_a), "trends": trends_of(state_a)},
        "disasters": load_events(), "migration": migration,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(out), separators=(",", ":")))
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    if mig_msg:
        print(f"Migration: {mig_msg} -> the Migration tab will show setup steps.")
    else:
        print(f"Migration: {len(migration['rows'])} districts loaded.")


if __name__ == "__main__":
    main()
