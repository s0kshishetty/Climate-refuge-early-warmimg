"""Step 3 – merge climate + disaster events (+ optional static data) into one panel.

Inputs  : data/raw/climate_monthly.csv   (from fetch_climate.py)
          data/raw/events.csv            (from convert_emdat.py or hand-made)
          data/raw/static_extra.csv      (OPTIONAL: state + any numeric columns,
                                          e.g. Census-2011 out-migration rate, poverty %)
Output  : data/processed/panel.csv       (one row per state x month)

Run:  python ml/prepare_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from states import STATES, normalize_state  # noqa: E402

# ───────────────────────── settings you can tune ─────────────────────────
HORIZON = 3                 # predict "high-impact event in the next 3 months"
BASELINE = (2000, 2020)     # climate normal period (WMO standard)
DISPLACED_FROM_AFFECTED = 0.05   # ASSUMPTION: 5% of 'affected' people are displaced
HI_QUANTILE = 0.5           # event is "high impact" if displaced proxy >= this quantile
HAZARDS = ["flood", "cyclone", "drought", "heat"]
# ─────────────────────────────────────────────────────────────────────────

RAW, OUT = Path("data/raw"), Path("data/processed")


def load_events():
    ev = pd.read_csv(RAW / "events.csv")
    ev["state"] = ev["state"].map(normalize_state)
    ev = ev.dropna(subset=["state"])
    for c in ("deaths", "affected", "homeless", "displaced"):
        if c not in ev:
            ev[c] = np.nan
    ev["month"] = ev["month"].fillna(1).clip(1, 12).astype(int)
    proxy = ev["homeless"].fillna(0) + DISPLACED_FROM_AFFECTED * ev["affected"].fillna(0)
    ev["displaced_est"] = ev["displaced"].where(ev["displaced"].notna(), proxy)
    pos = ev.loc[ev.displaced_est > 0, "displaced_est"]
    thr = pos.quantile(HI_QUANTILE) if len(pos) else 0
    ev["high_impact"] = (ev.displaced_est >= thr) & (ev.displaced_est > 0)
    print(f"events: {len(ev)}  | high-impact threshold = {thr:,.0f} displaced (est.) | "
          f"high-impact events: {int(ev.high_impact.sum())}")
    return ev, float(thr)


def climate_features():
    c = pd.read_csv(RAW / "climate_monthly.csv")
    c = c[c.state.isin(STATES)].copy()
    days = pd.to_datetime(dict(year=c.year, month=c.month, day=1)).dt.days_in_month
    c["rain_mm"] = c.precip_mm_day * days
    c["date"] = pd.to_datetime(dict(year=c.year, month=c.month, day=1))
    b = c[(c.year >= BASELINE[0]) & (c.year <= BASELINE[1])]
    base = b.groupby(["state", "month"]).agg(
        r_mu=("rain_mm", "mean"), r_sd=("rain_mm", "std"), t_mu=("temp_c", "mean")).reset_index()
    c = c.merge(base, on=["state", "month"], how="left")
    c["rain_z"] = ((c.rain_mm - c.r_mu) / c.r_sd.clip(lower=5)).clip(-4, 4)
    c["temp_anom"] = c.temp_c - c.t_mu
    c = c.sort_values(["state", "date"])
    g = c.groupby("state")
    c["rain_z_1"] = c.rain_z
    c["rain_z_3"] = g.rain_z.transform(lambda s: s.rolling(3, min_periods=3).mean())
    c["rain_z_6"] = g.rain_z.transform(lambda s: s.rolling(6, min_periods=6).mean())
    c["temp_anom_1"] = c.temp_anom
    c["temp_anom_3"] = g.temp_anom.transform(lambda s: s.rolling(3, min_periods=3).mean())
    c["dry_months_6"] = g.rain_z.transform(lambda s: (s < -1).astype(float).rolling(6, min_periods=6).sum())
    c["wet_months_3"] = g.rain_z.transform(lambda s: (s > 1).astype(float).rolling(3, min_periods=3).sum())
    keep = ["state", "date", "year", "month", "rain_z_1", "rain_z_3", "rain_z_6",
            "temp_anom_1", "temp_anom_3", "dry_months_6", "wet_months_3"]
    return c[keep]


def main():
    clim = climate_features()
    ev, thr = load_events()

    ev_end = pd.Timestamp(year=int(ev.year.max()), month=12, day=1)
    end = min(clim.date.max(), ev_end)
    clim = clim[clim.date <= end]
    print(f"panel covers {clim.date.min():%Y-%m} → {end:%Y-%m}")

    # monthly event table
    ev["date"] = pd.to_datetime(dict(year=ev.year, month=ev.month, day=1))
    m = ev.groupby(["state", "date"]).agg(
        n_events=("hazard", "size"), n_hi=("high_impact", "sum"),
        displaced=("displaced_est", "sum"), affected=("affected", "sum"),
        deaths=("deaths", "sum")).reset_index()
    for h in HAZARDS:
        mh = ev[ev.hazard == h].groupby(["state", "date"]).size().rename(f"n_{h}").reset_index()
        m = m.merge(mh, on=["state", "date"], how="left")

    p = clim.merge(m, on=["state", "date"], how="left").sort_values(["state", "date"])
    for c in ["n_events", "n_hi", "displaced", "affected", "deaths"] + [f"n_{h}" for h in HAZARDS]:
        p[c] = p[c].fillna(0)

    g = p.groupby("state")
    # history features (use only information up to month t)
    p["events_12m"] = g.n_events.transform(lambda s: s.rolling(12, min_periods=1).sum())
    p["hi_events_36m"] = g.n_hi.transform(lambda s: s.rolling(36, min_periods=1).sum())
    p["displaced_12m_log"] = np.log1p(g.displaced.transform(lambda s: s.rolling(12, min_periods=1).sum()))
    for h in HAZARDS:
        p[f"{h}_12m"] = g[f"n_{h}"].transform(lambda s: s.rolling(12, min_periods=1).sum())

    def months_since(s):
        out, last = [], None
        for i, v in enumerate(s.values):
            if v > 0:
                last = i
            out.append(60 if last is None else min(i - last, 60))
        return pd.Series(out, index=s.index)
    p["months_since_event"] = g.n_events.transform(months_since)

    # seasonality + static
    p["month_sin"] = np.sin(2 * np.pi * p.month / 12)
    p["month_cos"] = np.cos(2 * np.pi * p.month / 12)
    st = pd.DataFrame(STATES, index=["lat", "lon", "coastal", "himalayan"]).T.reset_index(names="state")
    p = p.merge(st, on="state", how="left")
    extra = RAW / "static_extra.csv"
    if extra.exists():
        ex = pd.read_csv(extra)
        ex["state"] = ex["state"].map(normalize_state)
        p = p.merge(ex.dropna(subset=["state"]), on="state", how="left")
        print("static_extra columns merged:", [c for c in ex.columns if c != "state"])

    # target: high-impact event in the next HORIZON months
    def future_any(s):
        fut = sum(s.shift(-k) for k in range(1, HORIZON + 1))
        return (fut > 0).astype(float).where(fut.notna())
    p["target"] = p.groupby("state")["n_hi"].transform(future_any)

    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT / "panel.csv", index=False)
    lab = p.dropna(subset=["target"])
    print(f"\nSaved {len(p)} rows -> {OUT/'panel.csv'}")
    print(f"labelled rows: {len(lab)} | positive rate: {lab.target.mean():.1%}")
    meta = pd.Series({"hi_threshold": thr, "horizon": HORIZON})
    meta.to_json(OUT / "meta.json")


if __name__ == "__main__":
    main()
