"""Step 3 (Maharashtra) - district migration analysis, climate-migration risk score, 2030-2050 projections.

Input : public/data/maharashtra.json              (from mh_build.py: real NASA POWER indicators, 2001-2025)
        data/raw/census/migration_district.csv    (Census 2001 + 2011, same file mh_build.py uses)  [optional]
Output: public/data/mh_analysis.json              (read by src/MaharashtraAnalysis.tsx)

Nothing here touches the India section (train.py / risk.json / App.tsx).

-- 5. District migration (needs the Census CSV; without it this part stays empty, nothing is invented)
   inflow   = Census "migrants" (people enumerated in the district whose last residence was elsewhere),
              2001 and 2011, as a stock and as % of the district population.
   outflow  = The Census does NOT publish out-migration per district. If you supply a column `out_migrants`
              (persons who left the district) it is used. Without it we show a transparent proxy:
              "net movement vs state" = district population growth 2001-11 minus the state's growth
              (positive = gaining people relative to Maharashtra, negative = losing). No birth/death data needed.

-- 6. Climate migration risk score (0-100 + Low / Moderate / High / Very High)
   Five factors, each the average of two indicators of the district's last 10 years (2016-25):
     heat      heatwave days, mean max temperature
     drought   share of drought years 2001-25, worst monsoon deficit
     flood     heavy-rain days, heaviest 5-day rain
     rainfall  low monsoon rainfall, year-to-year variability (CV)
     water     low pre-monsoon root-zone soil wetness, longest monsoon dry spell
   Each indicator is turned into a 0-100 score = its percentile among the 36 districts' 2016-25 values
   (higher = worse). Composite = weighted mean. Level cut-offs = quartiles of the 2016-25 composite, so the
   levels are RELATIVE to Maharashtra, not an absolute danger scale. The cut-offs are frozen and reused for
   the future years.

-- 7. Projections 2030 / 2035 / 2040 / 2050
   Statistical trend extrapolation, NOT a climate model (no CMIP6 / SSP scenarios).
   For each district and indicator: Theil-Sen trend over 2001-25. If the Mann-Kendall p < 0.10 the trend is
   applied starting from the 2016-25 mean; otherwise the projection stays at the 2016-25 mean.
   The band (low / high) combines the 90 % Theil-Sen slope range with the 10-90 % spread of the yearly
   residuals, so a non-significant trend gets a wide band.
   The risk score is recomputed for each future year from the projected indicators, using the 2016-25
   district values as the fixed reference (so an overall rise shows up as a rise).

Run from project root:   python ml/mh_analysis.py
Options:                 --src public/data/maharashtra.json --census data/raw/census/migration_district.csv
                         --out public/data/mh_analysis.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from mh_districts import DISTRICTS, census_lookup, norm  # noqa: E402

BASE0, BASE1 = 2016, 2025            # baseline decade
FUTURE = [2030, 2035, 2040, 2050]
P_APPLY = 0.10                       # apply a trend only if Mann-Kendall p < this
NEAR_STATE_PP = 2.0                  # |growth - state growth| below this = "near state average"
LEVELS = ["Low", "Moderate", "High", "Very High"]

WEIGHTS = {"heat": 0.20, "drought": 0.25, "flood": 0.20, "rainfall": 0.15, "water": 0.20}
FACTOR_LABELS = {"heat": "Heat", "drought": "Drought", "flood": "Flood / heavy rain",
                 "rainfall": "Rainfall scarcity & variability", "water": "Water stress"}
# factor -> [(indicator, sign)]   sign -1: a LOWER value means higher risk
FACTORS = {
    "heat": [("heatwave_days", 1), ("tmax", 1)],
    "drought": [("drought_freq", 1), ("worst_deficit", 1)],
    "flood": [("heavy_days", 1), ("rx5", 1)],
    "rainfall": [("monsoon_rain", -1), ("cv_rain", 1)],
    "water": [("soil_premonsoon", -1), ("max_dry_spell", 1)],
}
INDICATOR_LABELS = {
    "heatwave_days": "Heatwave days / yr", "tmax": "Mean max temp (°C)",
    "drought_freq": "Drought years (% of 2001–25)", "worst_deficit": "Worst monsoon deficit (%)",
    "heavy_days": "Heavy-rain days / yr", "rx5": "Heaviest 5-day rain (mm)",
    "monsoon_rain": "Monsoon rainfall (mm)", "cv_rain": "Monsoon rain variability (CV %)",
    "soil_premonsoon": "Pre-monsoon soil wetness (0–1)", "max_dry_spell": "Longest monsoon dry spell (days)",
}
# indicators projected forward (the others in FACTORS are held at their 2001-25 value)
PROJECTED = ["heatwave_days", "tmax", "heavy_days", "rx5", "monsoon_rain", "soil_premonsoon", "max_dry_spell"]
# metrics shown on the projection chart: key -> (label, unit, lower clamp, upper clamp)
PROJ_METRICS = {
    "tmean": ("Mean temperature", "°C", None, None),
    "tmax": ("Mean max temperature", "°C", None, None),
    "rain": ("Annual rainfall", "mm", 0, None),
    "monsoon_rain": ("Monsoon rainfall (Jun–Sep)", "mm", 0, None),
    "heatwave_days": ("Heatwave days", "days/yr", 0, 365),
    "heavy_days": ("Heavy-rain days (≥64.5 mm)", "days/yr", 0, 365),
    "rx5": ("Heaviest 5-day rain", "mm", 0, None),
    "max_dry_spell": ("Longest monsoon dry spell", "days", 0, 122),
    "soil_premonsoon": ("Pre-monsoon soil wetness", "0–1", 0, 1),
}


# ── helpers ───────────────────────────────────────────────────────
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 3)
    return o


def pct_rank(ref: np.ndarray, v: float) -> float:
    """Mid-rank percentile (0-100) of v inside the reference values (ties count half)."""
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0 or not np.isfinite(v):
        return float("nan")
    less = (ref < v - 1e-9).sum()
    equal = (np.abs(ref - v) <= 1e-9).sum()
    return (less + 0.5 * equal) / len(ref) * 100


def level_of(score: float, cuts) -> str:
    if not np.isfinite(score):
        return "n/a"
    return LEVELS[int(np.searchsorted(cuts, score, side="right"))]


# ── climate-migration risk ────────────────────────────────────────
def baseline_inputs(a: pd.DataFrame) -> dict:
    w = a.loc[BASE0:BASE1]
    drought = a["monsoon_class"].isin(["drought", "severe drought"])
    soil = w["soil_premonsoon"].mean() if "soil_premonsoon" in w else np.nan
    return {
        "heatwave_days": w["heatwave_days"].mean(), "tmax": w["tmax"].mean(),
        "heavy_days": w["heavy_days"].mean(), "rx5": w["rx5"].mean(),
        "monsoon_rain": w["monsoon_rain"].mean(), "soil_premonsoon": soil,
        "max_dry_spell": w["max_dry_spell"].mean(),
        # whole 2001-25 record (too noisy for a 10-year window, and not projected)
        "drought_freq": drought.mean() * 100,
        "worst_deficit": -a["monsoon_dep"].min(),
        "cv_rain": a["monsoon_rain"].std() / a["monsoon_rain"].mean() * 100,
    }


def score_one(inputs: dict, ref: dict) -> dict:
    """Factor scores + composite for one district, measured against the baseline reference arrays."""
    subs = {}
    for f, items in FACTORS.items():
        vals = []
        for k, sign in items:
            r = pct_rank(sign * ref[k], sign * inputs[k])
            if np.isfinite(r):
                vals.append(r)
        subs[f] = float(np.mean(vals)) if vals else float("nan")
    ok = {f: w for f, w in WEIGHTS.items() if np.isfinite(subs[f])}
    score = sum(subs[f] * w for f, w in ok.items()) / sum(ok.values()) if ok else float("nan")
    return {"factors": subs, "score": score}


# ── projections ───────────────────────────────────────────────────
def project_series(s: pd.Series, lo_clamp, hi_clamp):
    s = s.dropna()
    if len(s) < 15 or s.nunique() < 3:
        return None
    x, y = s.index.to_numpy(float), s.to_numpy(float)
    base = float(s.loc[BASE0:BASE1].mean())
    slope, icpt, s_lo, s_hi = stats.theilslopes(y, x, alpha=0.90)
    _, p = stats.kendalltau(x, y)
    p = None if np.isnan(p) else float(p)
    q10, q90 = np.percentile(y - (icpt + slope * x), [10, 90])
    applied = p is not None and p < P_APPLY
    centre = (BASE0 + BASE1) / 2

    def clamp(v):
        if lo_clamp is not None:
            v = max(v, lo_clamp)
        if hi_clamp is not None:
            v = min(v, hi_clamp)
        return v

    vals = {}
    for yr in FUTURE:
        dx = yr - centre
        mid = base + (slope * dx if applied else 0.0)
        lo = base + min(s_lo * dx, s_hi * dx) + q10
        hi = base + max(s_lo * dx, s_hi * dx) + q90
        vals[yr] = {"mid": clamp(mid), "low": clamp(min(lo, mid)), "high": clamp(max(hi, mid))}
    return {"baseline": base, "slope_decade": float(slope * 10), "p": p, "applied": bool(applied), "values": vals}


# ── migration (Census) ────────────────────────────────────────────
def load_census(path: Path):
    if not path.exists():
        return None, f"{path} not found"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    miss = [c for c in ["district", "year", "total_population", "total_migrants"] if c not in df.columns]
    if miss:
        return None, f"{path}: missing columns {miss}"
    df = df.dropna(subset=["district", "year"])
    if df.empty:
        return None, f"{path} has no data rows yet"
    lut = census_lookup()
    df["cid"] = df["district"].map(lambda s: lut.get(norm(s)))
    bad = sorted(df[df["cid"].isna()]["district"].astype(str).unique())
    if bad:
        return None, f"{path}: unknown district names {bad} (use Census spellings, e.g. Aurangabad, Osmanabad, Buldana, Gondiya, Raigarh)"
    df = df[df["year"].isin([2001, 2011])].copy()
    num = ["total_population", "total_migrants", "migrants_intra_district", "migrants_inter_district",
           "migrants_inter_state", "migrants_outside_india", "out_migrants"]
    for c in num:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.duplicated(["cid", "year"]).any():
        return None, f"{path}: duplicate district+year rows"
    return df, None


def migration_part(df: pd.DataFrame, risk_by_id: dict):
    names = {d["id"]: d["name"] for d in DISTRICTS}
    names["thane"] = "Thane (incl. Palghar)"
    streams = [("intra_district", "migrants_intra_district"), ("inter_district", "migrants_inter_district"),
               ("inter_state", "migrants_inter_state"), ("outside_india", "migrants_outside_india")]
    rows = []
    for cid, g in df.groupby("cid"):
        g = g.set_index("year")
        if not {2001, 2011} <= set(g.index):
            print(f"  note: {names[cid]} lacks a 2001 or 2011 row - skipped")
            continue
        r = {"id": cid, "name": names[cid]}
        for y in (2001, 2011):
            pop, mig = g.loc[y, "total_population"], g.loc[y, "total_migrants"]
            r[f"pop_{y}"], r[f"migrants_{y}"] = pop, mig
            r[f"inflow_share_{y}"] = mig / pop * 100
            for key, col in streams:
                if col in g and pd.notna(g.loc[y, col]) and mig:
                    r[f"{key}_pct_{y}"] = g.loc[y, col] / mig * 100
            if "out_migrants" in g and pd.notna(g.loc[y, "out_migrants"]):
                r[f"out_{y}"] = g.loc[y, "out_migrants"]
                r[f"out_share_{y}"] = g.loc[y, "out_migrants"] / pop * 100
                r[f"net_flow_{y}"] = mig - g.loc[y, "out_migrants"]
        r["inflow_change"] = r["migrants_2011"] - r["migrants_2001"]
        r["inflow_share_change_pp"] = r["inflow_share_2011"] - r["inflow_share_2001"]
        r["pop_growth_pct"] = (r["pop_2011"] / r["pop_2001"] - 1) * 100
        rows.append(r)
    if not rows:
        return None, "no district has both 2001 and 2011 rows"

    tab = pd.DataFrame(rows)
    st_growth = tab["pop_2011"].sum() / tab["pop_2001"].sum() - 1
    for r in rows:
        r["net_vs_state_pp"] = r["pop_growth_pct"] - st_growth * 100
        r["net_vs_state_persons"] = r["pop_2011"] - r["pop_2001"] * (1 + st_growth)
        r["movement"] = ("Gaining" if r["net_vs_state_pp"] > NEAR_STATE_PP
                         else "Losing" if r["net_vs_state_pp"] < -NEAR_STATE_PP else "Near state average")
        rk = risk_by_id.get(r["id"])
        r["risk_score"] = rk["score"] if rk else None
        r["risk_level"] = rk["level"] if rk else None
    has_out = all("out_2011" in r and "out_2001" in r for r in rows)

    corr = []
    t2 = pd.DataFrame(rows).dropna(subset=["risk_score"])
    if len(t2) >= 10:
        for y, lab in [("net_vs_state_pp", "Population growth 2001→11 vs state (pp)"),
                       ("inflow_share_change_pp", "Change in migrant share 2001→11 (pp)")]:
            rho, p = stats.spearmanr(t2["risk_score"], t2[y])
            corr.append({"x": "Climate risk score (2016–25)", "y": lab, "rho": rho, "p": p, "n": len(t2)})
    state = {"growth_pct": st_growth * 100}
    for y in (2001, 2011):
        state[f"pop_{y}"] = tab[f"pop_{y}"].sum()
        state[f"migrants_{y}"] = tab[f"migrants_{y}"].sum()
        state[f"inflow_share_{y}"] = state[f"migrants_{y}"] / state[f"pop_{y}"] * 100
        if has_out:
            state[f"out_{y}"] = sum(r[f"out_{y}"] for r in rows)
    stream_cols = [k for k, c in streams if any(f"{k}_pct_2011" in r for r in rows)]
    return {"rows": rows, "state": state, "correlations": corr, "has_outflow": has_out,
            "streams": stream_cols, "near_state_pp": NEAR_STATE_PP}, None


# ── main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="public/data/maharashtra.json")
    ap.add_argument("--census", default="data/raw/census/migration_district.csv")
    ap.add_argument("--out", default="public/data/mh_analysis.json")
    args = ap.parse_args()
    src, census_path, out_path = Path(args.src), Path(args.census), Path(args.out)

    if not src.exists():
        sys.exit(f"{src} not found. Run first:  python ml/mh_fetch.py  &&  python ml/mh_build.py")
    data = json.loads(src.read_text(encoding="utf-8"))
    dist = data["districts"]
    ann = {d["id"]: pd.DataFrame(d["annual"]).set_index("year") for d in dist}
    info = {d["id"]: d for d in dist}
    ids = [d["id"] for d in dist]
    print(f"Loaded {len(ids)} districts, {data['meta']['period'][0]}-{data['meta']['period'][1]}")

    # ---- 6. baseline risk --------------------------------------------------
    base_in = {i: baseline_inputs(ann[i]) for i in ids}
    keys = {k for f in FACTORS.values() for k, _ in f}
    ref = {k: np.array([base_in[i][k] for i in ids], dtype=float) for k in keys}
    base = {i: score_one(base_in[i], ref) for i in ids}
    cuts = np.percentile([base[i]["score"] for i in ids], [25, 50, 75])
    for i in ids:
        base[i]["level"] = level_of(base[i]["score"], cuts)

    # ---- 7. projections ------------------------------------------------------
    proj, proj_in = {}, {i: {y: dict(base_in[i]) for y in FUTURE} for i in ids}
    for i in ids:
        proj[i] = {}
        for m, (_, _, lo, hi) in PROJ_METRICS.items():
            if m in ann[i]:
                p = project_series(ann[i][m], lo, hi)
                if p:
                    proj[i][m] = p
        for y in FUTURE:
            for k in PROJECTED:
                if k in proj[i]:
                    proj_in[i][y][k] = proj[i][k]["values"][y]["mid"]
    future_risk = {
        i: {y: score_one(proj_in[i][y], ref) for y in FUTURE} for i in ids
    }
    for i in ids:
        for y in FUTURE:
            future_risk[i][y]["level"] = level_of(future_risk[i][y]["score"], cuts)

    # ---- state level (equal-weight average of districts, like the rest of the app) ----
    state_proj = {}
    for m in PROJ_METRICS:
        have = [i for i in ids if m in proj[i]]
        if not have:
            continue
        hist = pd.concat([ann[i][m] for i in have], axis=1).mean(axis=1)
        state_proj[m] = {
            "history": [{"year": int(y), "v": float(v)} for y, v in hist.items()],
            "values": {y: {k: float(np.mean([proj[i][m]["values"][y][k] for i in have])) for k in ("mid", "low", "high")}
                       for y in FUTURE},
            "applied_districts": int(sum(proj[i][m]["applied"] for i in have)), "n": len(have),
        }
    state_levels = {y: {lv: sum(future_risk[i][y]["level"] == lv for i in ids) for lv in LEVELS} for y in FUTURE}
    state_levels[BASE1] = {lv: sum(base[i]["level"] == lv for i in ids) for lv in LEVELS}
    state_mean_score = {y: float(np.mean([future_risk[i][y]["score"] for i in ids])) for y in FUTURE}
    state_mean_score[BASE1] = float(np.mean([base[i]["score"] for i in ids]))

    # ---- 5. migration ------------------------------------------------------------
    cdf, mig_msg = load_census(census_path)
    census_of = {d["id"]: d["census_id"] for d in DISTRICTS}
    risk_by_census = {}
    for i in ids:
        if i == "palghar":
            continue
        risk_by_census[census_of[i]] = {"score": base[i]["score"], "level": base[i]["level"]}
    migration = None
    if cdf is not None:
        migration, mig_msg = migration_part(cdf, risk_by_census)

    # ---- assemble ----------------------------------------------------------------
    districts_out = []
    for i in ids:
        d = info[i]
        districts_out.append({
            "id": i, "name": d["name"], "region": d["region"], "census_id": census_of[i],
            "risk": {
                "inputs": base_in[i], "factors": base[i]["factors"],
                "score": base[i]["score"], "level": base[i]["level"],
                "rank": 0,
            },
            "future_risk": {str(y): {"score": future_risk[i][y]["score"], "level": future_risk[i][y]["level"],
                                     "factors": future_risk[i][y]["factors"]} for y in FUTURE},
            "projection": {
                m: {"baseline": p["baseline"], "slope_decade": p["slope_decade"], "p": p["p"], "applied": p["applied"],
                    "history": [{"year": int(y), "v": float(v)} for y, v in ann[i][m].items()],
                    "values": p["values"]}
                for m, p in proj[i].items()
            },
        })
    order = sorted(districts_out, key=lambda r: -r["risk"]["score"])
    for rank, r in enumerate(order, 1):
        r["risk"]["rank"] = rank

    out = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "source": str(src).replace("\\", "/"),
            "baseline": [BASE0, BASE1], "future_years": FUTURE,
            "weights": WEIGHTS, "factor_labels": FACTOR_LABELS, "indicator_labels": INDICATOR_LABELS,
            "factor_indicators": {f: [k for k, _ in v] for f, v in FACTORS.items()},
            "levels": LEVELS, "level_cutoffs": [float(c) for c in cuts],
            "projected_indicators": PROJECTED, "p_apply": P_APPLY,
            "proj_metrics": {k: {"label": v[0], "unit": v[1]} for k, v in PROJ_METRICS.items()},
            "migration_message": mig_msg,
        },
        "migration": migration,
        "risk_summary": {"levels": state_levels, "mean_score": state_mean_score},
        "districts": districts_out,
        "state_projection": state_proj,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean(out), separators=(",", ":")), encoding="utf-8")
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # ---- console summary -----------------------------------------------------------
    print(f"\nRisk cut-offs (quartiles of 2016-25 composite): {', '.join(f'{c:.1f}' for c in cuts)}")
    print("Top 8 baseline risk:")
    for r in order[:8]:
        print(f"  {r['risk']['rank']:>2}. {r['name']:<34} {r['risk']['score']:5.1f}  {r['risk']['level']}")
    print("\nDistricts per level:  " + "   ".join(f"{y}: " + "/".join(str(state_levels[y][lv]) for lv in LEVELS)
                                                for y in [BASE1] + FUTURE) + f"   ({'/'.join(LEVELS)})")
    print("Mean risk score:      " + "   ".join(f"{y}: {state_mean_score[y]:.1f}" for y in [BASE1] + FUTURE))
    if migration:
        print(f"\nMigration: {len(migration['rows'])} census districts, outflow column: {'yes' if migration['has_outflow'] else 'no'}")
    else:
        print(f"\nMigration: NOT built - {mig_msg}\n  -> the dashboard tab will show the setup steps. Fill the Census CSV, then re-run this script.")


if __name__ == "__main__":
    main()
