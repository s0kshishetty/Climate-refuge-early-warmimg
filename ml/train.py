"""Step 4 – train the early-warning model and export public/data/risk.json

Model   : HistGradientBoosting (main) vs Logistic Regression & 'persistence' baselines
Split   : strictly by time (train → validation → test), no random shuffling
Output  : public/data/risk.json  (the React app reads this file)

Run:  python ml/train.py
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PANEL = Path("data/processed/panel.csv")
META = Path("data/processed/meta.json")
OUT = Path("public/data/risk.json")
TEST_YEARS, VAL_YEARS = 5, 4        # last 5 labelled years = test, 4 before = validation
SEED = 42

LABELS = {
    "rain_z_1": "Last-month rainfall anomaly", "rain_z_3": "3-month rainfall anomaly",
    "rain_z_6": "6-month rainfall anomaly", "temp_anom_1": "Last-month temperature anomaly",
    "temp_anom_3": "3-month temperature anomaly", "dry_months_6": "Dry months (last 6)",
    "wet_months_3": "Very wet months (last 3)", "events_12m": "Disasters in last 12 months",
    "hi_events_36m": "High-impact disasters (3 yrs)", "displaced_12m_log": "Displacement (12-month, log)",
    "flood_12m": "Floods in last 12 months", "cyclone_12m": "Cyclones in last 12 months",
    "drought_12m": "Droughts in last 12 months", "heat_12m": "Extreme-temperature events (12 mo)",
    "months_since_event": "Months since last disaster", "month_sin": "Season (sin)",
    "month_cos": "Season (cos)", "lat": "Latitude", "lon": "Longitude",
    "coastal": "Coastal state", "himalayan": "Himalayan state",
}
CORE = list(LABELS)


def level_of(p, th):
    if p >= th["critical"]:
        return "CRITICAL"
    if p >= th["high"]:
        return "HIGH"
    if p >= th["moderate"]:
        return "MODERATE"
    return "LOW"


def main():
    df = pd.read_csv(PANEL, parse_dates=["date"])
    meta = json.loads(META.read_text())
    extra = [c for c in df.columns if c not in CORE + [
        "state", "date", "year", "month", "target", "n_events", "n_hi", "displaced", "affected",
        "deaths", "n_flood", "n_cyclone", "n_drought", "n_heat"] and df[c].dtype != object]
    feats = CORE + extra
    for c in extra:
        LABELS[c] = c.replace("_", " ").capitalize()
    df = df.dropna(subset=["rain_z_6", "temp_anom_3"])           # warm-up rows
    lab = df.dropna(subset=["target"]).copy()
    lab = lab[lab.date <= "2025-09-01"].copy()

    last_year = int(lab.year.max())
    test_start = last_year - TEST_YEARS + 1
    val_start = test_start - VAL_YEARS
    tr, va, te = (lab[lab.year < val_start], lab[(lab.year >= val_start) & (lab.year < test_start)],
                  lab[lab.year >= test_start])
    print(f"train {tr.year.min()}-{tr.year.max()} ({len(tr)}) | val {val_start}-{test_start-1} ({len(va)}) "
          f"| test {test_start}-{last_year} ({len(te)})")
    print(f"positive rate  train {tr.target.mean():.1%} | val {va.target.mean():.1%} | test {te.target.mean():.1%}")

    X = lambda d: d[feats].astype(float)
    pos_w = (1 - tr.target.mean()) / max(tr.target.mean(), 1e-3)
    w = np.where(tr.target == 1, min(pos_w, 8), 1.0)

    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                        l2_regularization=1.0, random_state=SEED)
    gb.fit(X(tr), tr.target, sample_weight=w)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(X(tr).fillna(0), tr.target)

    # probability calibration on the validation years
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(gb.predict_proba(X(va))[:, 1], va.target)
    predict = lambda d: iso.predict(gb.predict_proba(X(d))[:, 1])

    p_te = predict(te)
    p_va = predict(va)
    thr = {
        "moderate": float(np.quantile(p_va, 0.60)),
        "high": float(np.quantile(p_va, 0.80)),
        "critical": float(np.quantile(p_va, 0.93)),
    }

    def metrics(y, p):
        return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))}
    m_gb = metrics(te.target, p_te)
    m_lr = metrics(te.target, lr.predict_proba(X(te).fillna(0))[:, 1])
    m_pers = {"roc_auc": float(roc_auc_score(te.target, te.events_12m)),
              "pr_auc": float(average_precision_score(te.target, te.events_12m))}
    hi = p_te >= thr["high"]
    tp = float(((hi) & (te.target == 1)).sum())
    m_gb["recall_at_high"] = tp / max(float((te.target == 1).sum()), 1)
    m_gb["precision_at_high"] = tp / max(float(hi.sum()), 1)
    m_gb["base_rate"] = float(te.target.mean())
    print("\nTEST  GB:", {k: round(v, 3) for k, v in m_gb.items()})
    print("TEST  LR:", {k: round(v, 3) for k, v in m_lr.items()})
    print("TEST  persistence:", {k: round(v, 3) for k, v in m_pers.items()})

    # global feature importance (permutation, on test years)
    pi = permutation_importance(gb, X(te), te.target, scoring="roc_auc", n_repeats=5,
                                random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=feats).clip(lower=0)
    imp_n = imp / imp.sum() if imp.sum() > 0 else imp
    importance = [{"feature": f, "label": LABELS[f], "importance": float(v)}
                  for f, v in imp_n.sort_values(ascending=False).head(10).items()]

    # forecast for the latest month in the panel + 10-month trend (all out-of-sample)
    allp = df.copy()
    allp["prob"] = predict(allp)
    latest_date = allp.date.max()
    mu, sd = allp[feats].mean(), allp[feats].std().replace(0, 1)
    ev_tot = pd.read_csv(PANEL).groupby("state").agg(
        events=("n_events", "sum"), affected=("affected", "sum"), displaced=("displaced", "sum"))
    haz_cols = ["n_flood", "n_cyclone", "n_drought", "n_heat"]
    haz_tot = pd.read_csv(PANEL).groupby("state")[haz_cols].sum()

    states = []
    for s, g in allp.groupby("state"):
        g = g.sort_values("date")
        cur = g.iloc[-1]
        prev = g.iloc[-2]["prob"] if len(g) > 1 else cur.prob
        z = ((cur[feats].astype(float) - mu) / sd)
        contrib = (z.clip(lower=0) * imp_n).drop(
            labels=["month_sin", "month_cos", "lat", "lon", "coastal", "himalayan"], errors="ignore"
        ).sort_values(ascending=False)
        
        drivers = [{"label": LABELS[f], "z": float(z[f])} for f, v in contrib.head(3).items() if v > 0]
        hz = haz_tot.loc[s]
        dominant = hz.idxmax().replace("n_", "") if hz.sum() > 0 else "none"
        trend = [{"m": d.strftime("%b %y"), "p": round(float(p) * 100, 1)}
                 for d, p in zip(g.date.tail(10), g.prob.tail(10))]
        states.append({
            "name": s, "lat": float(cur.lat), "lon": float(cur.lon),
            "prob": round(float(cur.prob) * 100, 1), "level": level_of(cur.prob, thr),
            "delta": round((float(cur.prob) - float(prev)) * 100, 1),
            "events": int(ev_tot.loc[s, "events"]), "affected": float(ev_tot.loc[s, "affected"]),
            "displaced_est": float(ev_tot.loc[s, "displaced"]), "dominant_hazard": dominant,
            "drivers": drivers, "trend": trend,
        })
    states.sort(key=lambda r: -r["prob"])

    alerts = []
    for r in states:
        if r["level"] in ("CRITICAL", "HIGH"):
            why = "; ".join(f"{d['label'].lower()} ({d['z']:+.1f}σ)" for d in r["drivers"]) or "multiple weak signals"
            alerts.append({"state": r["name"], "level": r["level"], "prob": r["prob"],
                           "msg": f"{r['prob']:.0f}% chance of a high-impact disaster in the next "
                                  f"{meta['horizon']} months. Signals: {why}."})

    out = {
        "meta": {
            "generated": date.today().isoformat(), "as_of": latest_date.strftime("%Y-%m"),
            "horizon_months": int(meta["horizon"]), "n_states": len(states),
            "train_years": [int(tr.year.min()), int(tr.year.max())], "val_years": [val_start, test_start - 1],
            "test_years": [test_start, last_year], "n_rows": int(len(lab)),
            "positive_rate": float(lab.target.mean()), "hi_threshold_displaced": meta["hi_threshold"],
            "thresholds": thr, "model": "Gradient boosting (calibrated)",
            "metrics": {"gradient_boosting": m_gb, "logistic_regression": m_lr, "persistence": m_pers},
        },
        "importance": importance, "states": states, "alerts": alerts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nSaved {OUT}  |  as-of {out['meta']['as_of']}  |  alerts: {len(alerts)}")


if __name__ == "__main__":
    main()
