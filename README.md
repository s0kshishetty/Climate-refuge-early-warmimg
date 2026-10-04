# ClimateWatch India EWS — Climate Migration Early Warning System

Python ML pipeline (`ml/`) + React dashboard (`src/`). The dashboard only reads `public/data/risk.json`,
which is produced by `ml/train.py`.

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
