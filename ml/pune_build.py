"""Step 2 (Pune) - features, model training, and today's risk per area.

Input  : data/raw/pune/*.csv            (from pune_fetch.py)
Output : public/data/pune.json          (read by src/PuneRisk.tsx)

What it does
  1. Builds features from past weather (rain windows, dry spell, temp anomaly, season...)
  2. Targets (next 3 days):  rain  = >= 50 mm rain       heat = Tmax >= 38 C
  3. Time-based split: train <= 2016, validation 2017-2020, test 2021+
  4. HistGradientBoosting + isotonic calibration, compared with a
     "just use the month's average" baseline
  5. Pulls the last ~75 days (Open-Meteo forecast API, observed part) so the
     risk is "as of yesterday". If offline, it uses the end of the archive.

Run from project root:   python ml/pune_build.py
Re-run it any time to refresh today's risk (no need to re-download history).
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from pune_areas import AREAS  # noqa: E402

RAW = Path("data/raw/pune")
OUT = Path("public/data/pune.json")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

H = 3                      # warning horizon (days)
TRAIN_END, VAL_END = 2016, 2020
RAIN_MM = 50.0             # heavy-rain event: total rain in next H days
HEAT_C = 38.0              # heat-stress event: any day Tmax >= this
# risk level = how many times higher than the normal (training) frequency
LEVELS = [(8.0, "CRITICAL"), (4.0, "HIGH"), (2.0, "MODERATE")]
RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

HAZARDS = {
    "rain": {"label": "Heavy rain / flood", "rule": f">= {RAIN_MM:.0f} mm rain in the next {H} days"},
    "heat": {"label": "Heat stress", "rule": f"Tmax >= {HEAT_C:.0f} C on any of the next {H} days"},
}
LABELS = {
    "rain_1d": "Rain today", "rain_3d": "Rain last 3 days", "rain_7d": "Rain last 7 days",
    "rain_14d": "Rain last 14 days", "rain_30d": "Rain last 30 days", "rain_60d": "Rain last 60 days",
    "wet_days_14": "Wet days in last 14", "days_since_heavy": "Days since heavy rain",
    "tmax": "Max temp today", "tmin": "Min temp today", "dtr": "Day-night temp range",
    "tmax_3d": "3-day avg max temp", "tmax_7d": "7-day avg max temp",
    "tmax_anom": "Max temp vs normal", "rain30_anom": "30-day rain vs normal",
    "season_sin": "Season (sin)", "season_cos": "Season (cos)",
    "elevation": "Elevation", "lat": "Latitude", "lon": "Longitude",
}
FEATURES = list(LABELS)


# ── data + features ──────────────────────────────────────────────

def load_area(a):
    p = RAW / f"{a['id']}.csv"
    if not p.exists():
        sys.exit(f"Missing {p}\nRun first:  python ml/pune_fetch.py")
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return df


def climatology(df):
    """Monthly 'normal' values from training years only (no leakage)."""
    tr = df[df.index.year <= TRAIN_END]
    r30 = df["rain"].rolling(30).sum().loc[tr.index]
    return {
        "tmax": tr["tmax"].groupby(tr.index.month).mean(),
        "rain30": r30.groupby(tr.index.month).mean(),
    }


def features(df, clim, a, elev):
    r, tx, tn = df["rain"], df["tmax"], df["tmin"]
    f = pd.DataFrame(index=df.index)
    for w in (1, 3, 7, 14, 30, 60):
        f[f"rain_{w}d"] = r.rolling(w).sum()
    f["wet_days_14"] = (r >= 2.5).astype(float).rolling(14).sum()
    pos = pd.Series(np.arange(len(r), dtype=float), index=r.index)
    last = pos.where(r >= 10).ffill()
    f["days_since_heavy"] = (pos - last).fillna(90).clip(upper=90)
    f["tmax"], f["tmin"], f["dtr"] = tx, tn, tx - tn
    f["tmax_3d"] = tx.rolling(3).mean()
    f["tmax_7d"] = tx.rolling(7).mean()
    month = df.index.month
    f["tmax_anom"] = f["tmax_3d"] - clim["tmax"].reindex(month).to_numpy()
    f["rain30_anom"] = f["rain_30d"] - clim["rain30"].reindex(month).to_numpy()
    doy = df.index.dayofyear
    f["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["elevation"], f["lat"], f["lon"] = elev, a["lat"], a["lon"]
    return f[FEATURES]


def targets(df):
    fr = df["rain"].rolling(H).sum().shift(-H)      # rain over t+1 .. t+H
    ft = df["tmax"].rolling(H).max().shift(-H)      # hottest day over t+1 .. t+H
    return pd.DataFrame({
        "rain": (fr >= RAIN_MM).astype(float).where(fr.notna()),
        "heat": (ft >= HEAT_C).astype(float).where(ft.notna()),
    }, index=df.index)


# ── training ─────────────────────────────────────────────────────

def safe(fn, y, p):
    try:
        return float(fn(y, p))
    except ValueError:
        return float("nan")


def train_hazard(panel, key):
    d = panel.dropna(subset=[key])
    y = d[key].astype(int)
    yr = d.index.year
    tr, va, te = yr <= TRAIN_END, (yr > TRAIN_END) & (yr <= VAL_END), yr > VAL_END
    X = d[FEATURES]

    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=250,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0,
    ).fit(X[tr], y[tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        clf.predict_proba(X[va])[:, 1], y[va])
    p_te = iso.predict(clf.predict_proba(X[te])[:, 1])

    base_rate = float(y[tr].mean())
    month_rate = y[tr].groupby(d.index[tr].month).mean()
    b_te = pd.Series(d.index[te].month).map(month_rate).fillna(base_rate).to_numpy()

    metrics = {
        "roc_auc": safe(roc_auc_score, y[te], p_te),
        "pr_auc": safe(average_precision_score, y[te], p_te),
        "brier": safe(brier_score_loss, y[te], p_te),
        "baseline_roc_auc": safe(roc_auc_score, y[te], b_te),
        "baseline_pr_auc": safe(average_precision_score, y[te], b_te),
        "test_positive_rate": float(y[te].mean()),
    }

    importance = []
    try:
        rng = np.random.RandomState(0)
        idx = rng.choice(np.flatnonzero(te), size=min(15000, int(te.sum())), replace=False)
        pi = permutation_importance(clf, X.iloc[idx], y.iloc[idx], scoring="roc_auc",
                                    n_repeats=3, random_state=0)
        order = np.argsort(pi.importances_mean)[::-1][:6]
        top = max(pi.importances_mean[order[0]], 1e-9)
        importance = [{"label": LABELS[FEATURES[i]],
                       "score": round(float(max(pi.importances_mean[i], 0) / top), 3)}
                      for i in order if pi.importances_mean[i] > 0]
    except ValueError:
        pass

    print(f"  {key}: base rate {base_rate:.3%} | test ROC-AUC {metrics['roc_auc']:.3f} "
          f"(month-baseline {metrics['baseline_roc_auc']:.3f}) | PR-AUC {metrics['pr_auc']:.3f} "
          f"(baseline {metrics['baseline_pr_auc']:.3f})")
    return {"clf": clf, "iso": iso, "base_rate": base_rate, "metrics": metrics,
            "importance": importance, "years": (int(yr.min()), int(yr.max()))}


# ── "today" ──────────────────────────────────────────────────────

def recent_weather(a, hist):
    """Last ~75 days. Live from Open-Meteo; falls back to end of archive."""
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": a["lat"], "longitude": a["lon"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "past_days": 75, "forecast_days": 1, "timezone": "Asia/Kolkata",
        }, timeout=30)
        r.raise_for_status()
        j = r.json()["daily"]
        cur = pd.DataFrame({"rain": j["precipitation_sum"], "tmax": j["temperature_2m_max"],
                            "tmin": j["temperature_2m_min"]},
                           index=pd.to_datetime(j["time"]))
        cur = cur[cur.index < pd.Timestamp(date.today())].dropna()
        if len(cur) >= 61:
            return cur, True
    except Exception as e:  # noqa: BLE001
        print(f"   (live fetch failed for {a['name']}: {e.__class__.__name__}; using archive end)")
    return hist[["rain", "tmax", "tmin"]].tail(75), False


def level_of(p, base):
    lift = p / base if base > 0 else 0.0
    for t, name in LEVELS:
        if lift >= t:
            return name, lift
    return "LOW", lift


def main():
    print("Loading data ...")
    hist, clims, elevs, frames = {}, {}, {}, []
    for a in AREAS:
        df = load_area(a)
        hist[a["id"]] = df
        elevs[a["id"]] = float(df["elevation"].iloc[0])
        clims[a["id"]] = climatology(df)
        p = pd.concat([features(df, clims[a["id"]], a, elevs[a["id"]]), targets(df)], axis=1)
        frames.append(p)
    panel = pd.concat(frames).dropna(subset=FEATURES)
    print(f"  {len(panel):,} area-days, {len(AREAS)} areas")

    print("Training ...")
    models = {k: train_hazard(panel, k) for k in HAZARDS}

    print("Scoring today ...")
    areas_out, live_count, as_of = [], 0, None
    for a in AREAS:
        cur, live = recent_weather(a, hist[a["id"]])
        live_count += int(live)
        f = features(cur, clims[a["id"]], a, elevs[a["id"]])
        row = f.iloc[-1]
        x = f.iloc[[-1]]
        day = f.index[-1]
        as_of = day if as_of is None or day > as_of else as_of

        out = {"id": a["id"], "name": a["name"], "lat": a["lat"], "lon": a["lon"],
               "elevation": round(elevs[a["id"]]), "exposure": a["exposure"]}
        for k, m in models.items():
            p = float(m["iso"].predict(m["clf"].predict_proba(x)[:, 1])[0])
            if k == "rain":                      # wetter-risk areas matter more
                p = min(1.0, p * (0.7 + 0.6 * a["exposure"]))
            lvl, lift = level_of(p, m["base_rate"])
            out[k] = {"prob": round(p, 4), "level": lvl, "lift": round(lift, 2)}
        out["overall"] = max((out["rain"]["level"], out["heat"]["level"]), key=RANK.get)

        mon = day.month
        out["recent"] = {k: round(float(row[c]), 1) for k, c in [
            ("rain_3d", "rain_3d"), ("rain_7d", "rain_7d"), ("rain_30d", "rain_30d"),
            ("tmax", "tmax"), ("tmin", "tmin")]}
        dsh = int(row["days_since_heavy"])
        out["drivers"] = [
            {"label": "30-day rainfall",
             "detail": f"{row['rain_30d']:.0f} mm vs about {clims[a['id']]['rain30'][mon]:.0f} mm normal for this month"},
            {"label": "Last heavy rain (>= 10 mm)",
             "detail": f"{dsh} days ago" if dsh < 90 else "none in the last 2+ months"},
            {"label": "3-day average max temp",
             "detail": f"{row['tmax_3d']:.1f} C ({row['tmax_anom']:+.1f} C vs normal)"},
        ]
        tail = cur.tail(30)
        out["history"] = [{"d": i.strftime("%d %b"), "rain": round(float(r.rain), 1),
                           "tmax": round(float(r.tmax), 1)} for i, r in tail.iterrows()]
        areas_out.append(out)

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "source": "Open-Meteo / ERA5 reanalysis, daily, per-area coordinates",
        "live_areas": live_count,
        "horizon_days": H,
        "train_years": [models["rain"]["years"][0], TRAIN_END],
        "val_years": [TRAIN_END + 1, VAL_END],
        "test_years": [VAL_END + 1, models["rain"]["years"][1]],
        "n_rows": int(len(panel)),
        "level_rule": ("Level = how many times higher than the normal frequency of the event "
                       "(x2 MODERATE, x4 HIGH, x8 CRITICAL). Rain risk is scaled by the area's "
                       "flood-exposure estimate."),
        "hazards": {k: {**HAZARDS[k], "base_rate": round(models[k]["base_rate"], 4),
                        "metrics": {mk: (None if np.isnan(mv) else round(mv, 4))
                                    for mk, mv in models[k]["metrics"].items()},
                        "importance": models[k]["importance"]} for k in HAZARDS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "areas": areas_out}, ensure_ascii=False))
    print(f"Wrote {OUT}  (as of {meta['as_of']}, live data for {live_count}/{len(AREAS)} areas)")


if __name__ == "__main__":
    main()
