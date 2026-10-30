"""Step 4 (Maharashtra) - ML model: climate -> in-migration, forecast 2030 / 2035 / 2040 / 2050.

Inputs (all real, nothing typed in):
  public/data/mh_analysis.json            climate history 2001-2025 + projected climate (run ml/mh_analysis.py first)
  data/raw/census/migration_district.csv  Census 2001 + 2011 migrants + population
Output: public/data/mh_forecast.json      (read by src/MigrationForecast.tsx)

What is learned (35 districts, one row per district):
  target   = change 2001->2011 in "migrants from outside the district" as % of population (pp)
             (inter-district + inter-state + abroad; migration inside the district is left out)
  features = climate of 2001-2010 (heatwave days, Tmax, Rx5, monsoon rain, dry spell, soil wetness),
             coastal flag, migrant share in 2001
  models   = Ridge, small Random Forest; compared with "predict the average" by leave-one-district-out CV.
Forecast for year Y: share(Y) = share(2011) + model(climate_Y) * (Y-2011)/10,  climate_Y from mh_analysis projections.
"Climate effect" = model(climate_Y) - model(climate 2001-10): the part of the change that comes from climate alone.
Only TWO census years exist, so this is 35 samples: treat the output as a scenario with a wide band, not as a certain number.

Run from project root:  python ml/mh_predict.py
"""
import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from mh_districts import DISTRICTS, census_lookup, norm  # noqa: E402

FUT = ["2030", "2035", "2040", "2050"]
CLIM = ["heatwave_days", "tmax", "rx5", "monsoon_rain", "max_dry_spell", "soil_premonsoon"]
LABEL = {"heatwave_days": "Heatwave days", "tmax": "Max temperature", "heavy_days": "Heavy-rain days",
         "rx5": "Heaviest 5-day rain", "monsoon_rain": "Monsoon rainfall", "max_dry_spell": "Longest dry spell",
         "soil_premonsoon": "Pre-monsoon soil wetness", "coastal": "Coastal district", "share_start": "Starting migrant share"}
FEATS = CLIM + ["coastal", "share_start"]

ana = json.load(open("public/data/mh_analysis.json"))
cen = pd.read_csv("data/raw/census/migration_district.csv")
lut = census_lookup()
cen["cid"] = cen["district"].map(lambda s: lut[norm(s)])
for c in ["migrants_inter_district", "migrants_inter_state", "migrants_outside_india"]:
    cen[c] = pd.to_numeric(cen[c], errors="coerce").fillna(0)
cen["out_d"] = cen.migrants_inter_district + cen.migrants_inter_state + cen.migrants_outside_india
cen["share"] = cen.out_d / cen.total_population * 100
by = {(r.cid, r.year): r for r in cen.itertuples()}
meta = {d["id"]: d for d in DISTRICTS}
ann = {d["id"]: pd.DataFrame(d["annual"]).set_index("year") for d in json.load(open("public/data/maharashtra.json"))["districts"]}

rows, fut_x = [], {}
for d in ana["districts"]:
    cid = d["census_id"]
    if d["id"] == "palghar" or (cid, 2001) not in by or (cid, 2011) not in by:
        continue
    pr = d["projection"]
    a = ann[d["id"]]
    # an indicator without a projection (e.g. no variation) is read from the annual record and held at its 2016-25 mean
    hist = {k: (np.mean([h["v"] for h in pr[k]["history"] if 2001 <= h["year"] <= 2010]) if k in pr
                else a.loc[2001:2010, k].mean()) for k in CLIM}
    fv = lambda k, y: pr[k]["values"][y]["mid"] if k in pr else a.loc[2016:2025, k].mean()
    s0, s1 = by[(cid, 2001)], by[(cid, 2011)]
    base = {**hist, "coastal": float(meta[cid]["coastal"]), "share_start": s0.share}
    rows.append({"id": cid, "name": meta[cid]["name"], **base, "y": s1.share - s0.share,
                 "share_2001": s0.share, "share_2011": s1.share, "pop_2011": s1.total_population,
                 "cagr": (s1.total_population / s0.total_population) ** 0.1 - 1})
    fut_x[cid] = {y: {**{k: fv(k, y) for k in CLIM}, "coastal": base["coastal"],
                      "share_start": s1.share} for y in FUT}
df = pd.DataFrame(rows)
X, y = df[FEATS].values, df["y"].values
print(f"{len(df)} districts, target mean {y.mean():.2f} pp, sd {y.std():.2f} pp")

models = {
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 30))),
    "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=3, min_samples_leaf=3, random_state=0),
}
res = {}
loo_mean = np.array([np.delete(y, i).mean() for i in range(len(y))])
rm = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))
res["Average only (baseline)"] = {"rmse": rm(loo_mean), "r2": float(1 - np.sum((y - loo_mean) ** 2) / np.sum((y - y.mean()) ** 2))}
for n, m in models.items():
    p = cross_val_predict(m, X, y, cv=LeaveOneOut())
    res[n] = {"rmse": rm(p), "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}
best = min(models, key=lambda n: res[n]["rmse"])
skill = res[best]["rmse"] < res["Average only (baseline)"]["rmse"]
for n, v in res.items():
    print(f"  {n:26s} LOO RMSE {v['rmse']:.2f} pp   LOO R2 {v['r2']:+.2f}")
print(f"best model: {best}  | beats the average baseline: {skill}")
model = models[best].fit(X, y)
band = res[best]["rmse"]

# what the model uses
if best == "Ridge":
    co = model[-1].coef_
    imp = np.abs(co) / np.abs(co).sum()
    sign = np.sign(co)
else:
    imp = model.feature_importances_
    sign = np.sign([np.corrcoef(df[f], y)[0, 1] for f in FEATS])
importance = sorted([{"feature": f, "label": LABEL[f], "importance": float(i), "direction": int(s)}
                     for f, i, s in zip(FEATS, imp, sign)], key=lambda r: -r["importance"])

out = []
for r in rows:
    cid = r["id"]
    ref = np.array([[r[f] for f in FEATS]])
    pred_ref = float(model.predict(ref)[0])
    pred = {}
    for yy in FUT:
        x = np.array([[fut_x[cid][yy][f] for f in FEATS]])
        f_y = float(model.predict(x)[0])
        k = (int(yy) - 2011) / 10
        # climate-only effect: same district, same starting share, climate of the year vs climate of 2001-10
        x_ref = x.copy(); x_ref[0, :len(CLIM)] = ref[0, :len(CLIM)]
        eff = (f_y - float(model.predict(x_ref)[0])) * k
        share = max(0.0, r["share_2011"] + f_y * k)
        pop = r["pop_2011"] * (1 + r["cagr"]) ** (int(yy) - 2011)
        pred[yy] = {"share": share, "low": max(0.0, share - band * k), "high": share + band * k,
                    "climate_effect_pp": eff, "population": pop, "migrants": share / 100 * pop}
    out.append({"id": cid, "name": r["name"], "share_2001": r["share_2001"], "share_2011": r["share_2011"],
                "pop_2011": r["pop_2011"], "pop_cagr_pct": r["cagr"] * 100, "pred": pred})

state = {}
for yy in FUT:
    pop = sum(o["pred"][yy]["population"] for o in out)
    mig = sum(o["pred"][yy]["migrants"] for o in out)
    state[yy] = {"share": mig / pop * 100, "population": pop, "migrants": mig,
                 "climate_effect_pp": float(np.average([o["pred"][yy]["climate_effect_pp"] for o in out],
                                                       weights=[o["pop_2011"] for o in out]))}
p11 = sum(o["pop_2011"] for o in out)
state["2011"] = {"share": sum(o["share_2011"] * o["pop_2011"] for o in out) / p11}

Path("public/data").mkdir(parents=True, exist_ok=True)
json.dump({
    "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_districts": len(df), "years": FUT,
             "target": "In-migrants from outside the district (inter-district + inter-state + abroad), % of population",
             "best_model": best, "beats_baseline": bool(skill), "cv": res, "band_pp_per_decade": band,
             "climate_source": ana["meta"].get("source", "NASA POWER 2001-2025 + trend projection in mh_analysis.json"),
             "caveats": ["Only two Census years (2001, 2011) exist, so the model learns from 35 districts (one change each).",
                         "Leave-one-district-out cross-validation scores are shown; a model that does not beat the average baseline has no real skill.",
                         "Climate inputs for 2030-2050 are trend extrapolations (see Future projections), not CMIP6 scenarios.",
                         "Census 'migrant' = lifetime stock of people living outside their birthplace/last residence, not yearly flows; marriage, work and education dominate.",
                         "Scenario, not a prediction of certainty. Band = leave-one-out RMSE scaled by decades."]},
    "importance": importance, "state": state, "districts": out}, open("public/data/mh_forecast.json", "w"), indent=1)
print("written public/data/mh_forecast.json")
