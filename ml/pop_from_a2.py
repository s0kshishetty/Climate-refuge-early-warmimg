"""Census A-2 (decadal variation, Maharashtra) -> data/raw/census/population_district.csv
A-2 has the total population of every district for every census year, so ONE file gives both 2001 and 2011.

Run from project root:
  python ml/pop_from_a2.py data/raw/census/27_A-2_Maharastra.xls
"""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("a2")
ap.add_argument("--out", default="data/raw/census/population_district.csv")
a = ap.parse_args()

df = pd.read_excel(a.a2, header=None, skiprows=5).iloc[:, :5]
df.columns = ["state", "dcode", "name", "year", "persons"]
df[["state", "dcode", "name"]] = df[["state", "dcode", "name"]].ffill()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["persons"] = pd.to_numeric(df["persons"], errors="coerce")
df = df.dropna(subset=["year", "persons"])
df = df[df["state"].astype(str).str.strip().isin(["27", "27.0"]) & df["year"].isin([2001, 2011])]
df = df[df["dcode"].astype(str).str.strip().str.lstrip("0").str.replace(".0", "", regex=False) != ""]  # drop state row
out = pd.DataFrame({"district": df["name"].astype(str).str.strip(), "year": df["year"].astype(int),
                    "total_population": df["persons"].astype(int)})
print(out.groupby("year").agg(districts=("district", "count"), total=("total_population", "sum")))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.sort_values(["year", "district"]).to_csv(a.out, index=False)
print("written", a.out)
