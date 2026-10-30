# ClimateWatch India EWS — Climate Migration Early Warning System

Python ML pipeline (`ml/`) + React dashboard (`src/`). The dashboard has a sidebar with two pages (India state-wise + Maharashtra):

- **India** — state-wise climate-migration risk. Reads `public/data/risk.json`, produced by `ml/train.py`.
- **Maharashtra** — district-wise climate 2001–2025, trend analysis and Census 2001/2011 migration. Reads `public/data/maharashtra.json`, produced by `ml/mh_build.py` (see the Maharashtra section below).
- (Older) Pune area-wise early warning scripts `ml/pune_*.py` are still in the repo but no longer in the sidebar.

## Run order (from project root)
```
pip install -r requirements.txt
python ml/fetch_climate.py                      # -> data/raw/climate_monthly.csv   (NASA POWER, no key)
python ml/convert_emdat.py data/raw/emdat_india.xlsx   # -> data/raw/events.csv
python ml/prepare_data.py                       # -> data/processed/panel.csv
python ml/train.py                              # -> public/data/risk.json
pnpm install && pnpm dev                        # dashboard
```
Optional: put `data/raw/static_extra.csv` (column `state` + numeric columns such as Census-2011
out-migration rate, poverty %) — it is merged automatically as extra state-level features.

## Target (what the model predicts)
`target = 1` if the state has a high-impact disaster (estimated displaced people >= median of all events)
in any of the next 3 months. Displaced ≈ homeless + 0.05 × affected (assumption, see `prepare_data.py`).
Split by time: train → validation (calibration) → test (last 5 labelled years).

---

# Pune area-wise early warning

Search a Pune area (Kothrud, Hinjewadi, Kalyani Nagar, ...) and see its risk for the coming days,
predicted by models trained on real daily weather data.

## Run order (from project root)
```
python ml/pune_fetch.py nasa      # -> data/raw/pune/<area>.csv      daily weather per area (NASA POWER, no key)
python ml/pune_exposure.py        # -> ml/pune_exposure.json         flood-exposure index (OpenStreetMap + SRTM)
python ml/pune_build.py           # -> public/data/pune.json         train models + risk per area
npm install && npm run dev        # dashboard -> sidebar -> Pune
```
Re-run only `python ml/pune_build.py` to refresh the risk; the downloaded history is reused.
`pune_fetch.py` without `nasa` tries Open-Meteo (ERA5) first and falls back to NASA POWER if rate-limited.
Keep **one source for all areas** (delete `data/raw/pune` before switching).

If the OpenStreetMap server blocks your network, `pune_exposure.py` prints manual steps: export the
waterways from https://overpass-turbo.eu and save as `data/raw/pune_waterways.json`.

## What is predicted
| Hazard | Event the model predicts | Horizon |
|---|---|---|
| Heavy rain / flood | >= 50 mm rain in total | next 3 days |
| Heat stress | any day with Tmax >= 38 °C | next 3 days |
| Dry spell / drought | rain < 30 % of the month's normal (only months that normally get >= 20 mm / 30 days) | next 30 days |

Earthquake and cyclone are **not predicted** (they cannot be forecast from weather data). The Pune page shows
them as information cards only.

## How it works
1. **Data** — daily Tmax, Tmin and rainfall per area, 2000 -> today (real reanalysis data, not simulated).
2. **Features** — rain over the last 1/3/7/14/30/60 days, wet days, days since last heavy rain, temperature
   averages and anomaly vs the month's normal (normals from training years only), season (sin/cos),
   elevation, lat/lon.
3. **Targets** — the three events above, looked up from the actual future weather in the history.
4. **Model** — HistGradientBoosting, one per hazard, trained on all areas together. Time-based split:
   train <= 2016, validation 2017-2020 (probability calibration, isotonic), test 2021+.
   Each model is compared with a "just use the month's average" baseline.
5. **Today's risk** — the last ~75 days of weather -> features -> calibrated probability. Level = how many times
   higher than the normal frequency of the event (x2 MODERATE, x4 HIGH, x8 CRITICAL; also >= 50 % HIGH,
   >= 80 % CRITICAL). If the live API is unreachable, the end of the downloaded archive is used.
6. **Flood-exposure index (0-1)** — computed per area from the distance to the nearest river/stream/canal
   (OpenStreetMap) and the height above it (SRTM 30 m elevation). It scales the heavy-rain probability
   (x 0.7 ... x 1.3).

## Files
| File | Purpose |
|---|---|
| `ml/pune_areas.py` | List of Pune areas (id, name, lat, lon). Loads `pune_exposure.json` if present. |
| `ml/pune_fetch.py` | Downloads daily weather for every area. |
| `ml/pune_exposure.py` | Computes the flood-exposure index from OpenStreetMap + SRTM. |
| `ml/pune_build.py` | Features, training, evaluation, today's risk -> `public/data/pune.json`. |
| `src/Layout.tsx` | Sidebar (India / Pune) and page switch. |
| `src/PuneRisk.tsx` | Pune page: area search, risk tiles, charts, model card, info cards. |
| `src/main.tsx` | Entry point, renders `Layout`. |

## Limitations (please read)
- The weather grid is coarse (NASA POWER about 50 km), so neighbouring areas get similar weather. Differences
  between areas come mostly from elevation, location and the exposure index.
- Predictions use **past weather only** (no weather-forecast input). Skill beyond seasonality is modest; the
  model card on the Pune page shows the test scores next to the month-average baseline.
- "Heavy rain / flood" uses rainfall as a proxy for flood; there is no river-level or dam-discharge data.
- The exposure index is a physical susceptibility index from measured inputs. It is **not an official
  flood-inundation map** and is not verified by PMC, IMD or NDMA. Its scales (2 km, 25 m, 50-50 weights) are
  modelling choices. SRTM elevation has several metres of error.
- Heat risk is about 0 outside the hot season (about Mar-Jun); dry-spell risk is only judged in rainy months.
- Research prototype, not an official warning. Follow IMD / PMC alerts.

---

# Maharashtra: climate data, trends and migration

All numbers are computed from downloaded real data. Nothing is typed into the code.

## Run order (from project root)
```
pip install -r requirements.txt
python ml/mh_fetch.py        # -> data/raw/mh/<district>.csv   NASA POWER daily 2001-2025, 36 districts
python ml/mh_build.py        # -> public/data/maharashtra.json
npm install && npm run dev   # sidebar -> Maharashtra
```

## Census migration data (real, you supply it once)
```
mkdir -p data/raw/census
cp data/census_template/migration_district_TEMPLATE.csv data/raw/census/migration_district.csv
```
Fill one row per district per census year (2001, 2011) from the Census of India D-series migration tables
(censusindia.gov.in -> Census Tables -> Migration). Use the Census spellings (Aurangabad, Osmanabad, Buldana,
Gondiya, Raigarh ...). Palghar did not exist in 2001/2011: it is inside Thane.
Required columns: `district, year, total_population, total_migrants`.
Optional: `migrants_intra_district, migrants_inter_district, migrants_inter_state, migrants_outside_india`.
Re-run `python ml/mh_build.py`.

Optional disaster records: `python ml/convert_emdat.py data/raw/emdat_india.xlsx` (already part of the India pipeline)
- Maharashtra rows are then listed on the Climate tab.

## Indicators
Heatwave = IMD-style (Tmax >= 40 C, 37 C coastal, and >= 4.5 C above the calendar-day normal, or >= 45 C; 2+ day spells).
Flood proxy = days with >= 64.5 mm and >= 115.6 mm rain. Drought = monsoon rain <= -25 % of the district mean (IMD),
severe <= -50 %. Water stress = composite 0-100 of low monsoon rain, dry root-zone soil (NASA GWETROOT) and high Tmax,
ranked inside each district. Trends = Theil-Sen slope per decade + Mann-Kendall p-value.
