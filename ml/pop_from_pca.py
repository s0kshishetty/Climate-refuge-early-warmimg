"""Census PCA (Primary Census Abstract) files -> data/raw/census/population_district.csv
Takes the DISTRICT rows of Maharashtra (state code 27, Total area) and writes district, year, total_population.

Run from project root:
  python ml/pop_from_pca.py --pca-2001 <2001 PCA file that has Maharashtra DISTRICT rows> --pca-2011 <2011 all-India PCA file>
"""
import argparse
from pathlib import Path
import pandas as pd


def read(path, year):
    df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip().upper().replace("_", "") for c in df.columns]
    pick = lambda *n: next(c for c in df.columns if c in n)
    st, lvl, name, tru, pop = pick("STATE"), pick("LEVEL"), pick("NAME"), pick("TRU"), pick("TOTP")
    x = df[(df[st].str.strip().str.lstrip("0") == "27") & (df[tru].str.strip() == "Total")
           & (df[lvl].str.strip().str.upper() == "DISTRICT")]
    if x.empty:
        raise SystemExit(f"{path}: no Maharashtra DISTRICT rows. This is probably a state-level file; "
                         f"download the Maharashtra district PCA for {year} from censusindia.gov.in")
    out = pd.DataFrame({"district": x[name].str.strip(), "year": year, "total_population": pd.to_numeric(x[pop])})
    print(f"{year}: {len(out)} districts")
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--pca-2001", required=True)
ap.add_argument("--pca-2011", required=True)
ap.add_argument("--out", default="data/raw/census/population_district.csv")
a = ap.parse_args()
res = pd.concat([read(a.pca_2001, 2001), read(a.pca_2011, 2011)])
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
res.to_csv(a.out, index=False)
print("written", a.out)
