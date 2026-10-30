"""Census D-02 Excel files (2001 + 2011)  ->  data/raw/census/migration_district.csv   (the file mh_build.py / mh_analysis.py read)

Inputs
  --d02-2001  Census 2001 D-02 Maharashtra file (PC01_D02_27.xls or .xlsx)
  --d02-2011  Census 2011 D-02 Maharashtra file (DS-2700-D02-MDDS.XLSX)
  --pop       data/raw/census/population_district.csv   columns: district,year,total_population
              (Census PCA / A-1 table: total population of each district in 2001 and 2011.
               D-02 has NO population, so this one small file has to come from censusindia.gov.in.)
Nothing is typed into the code: every number comes from the files you give.

Run from project root:
  python ml/census_d02_to_csv.py --d02-2001 data/raw/census/PC01_D02_27.xls --d02-2011 data/raw/census/DS-2700-D02-MDDS.XLSX
"""
import argparse, re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mh_districts import DISTRICTS, census_lookup, norm  # noqa: E402

COLS = ["table", "state", "dcode", "area", "last", "plc", "ru", "tot", "m", "f"]
KINDS = {"total": "total_migrants", "same": "migrants_intra_district", "other_d": "migrants_inter_district",
         "other_s": "migrants_inter_state", "abroad": "migrants_outside_india"}


def kind(last):
    l = str(last).strip().lower()
    if l == "total": return "total"
    if l.startswith("elsewhere in the district"): return "same"
    if l.startswith("in other districts"): return "other_d"
    if l.startswith("states in india beyond"): return "other_s"
    if "outside india" in l: return "abroad"
    return None


def clean(a):
    a = re.sub(r"^district\s*-\s*", "", str(a).strip(), flags=re.I)
    return re.sub(r"\s*\d+\s*$", "", a).replace("*", "").strip()


def read_d02(path, year, lut):
    df = pd.read_excel(path, header=None, skiprows=5).iloc[:, :10]
    df.columns = COLS
    df = df[df.state.astype(str).str.strip().isin(["27", "27.0"])]
    rows = {}
    for _, r in df.iterrows():
        a = str(r.area).strip()
        if re.match(r"^state", a, re.I): continue
        if str(r.ru).strip() != "Total" or str(r.plc).strip() not in ("Total", "Total Population"): continue
        k = kind(r["last"])
        if not k: continue
        name = clean(a)
        cid = lut.get(norm(name))
        if cid is None:
            sys.exit(f"{path}: district '{name}' not in ml/mh_districts.py - add it as an alias there")
        rows.setdefault(cid, {"district": name, "year": year})[KINDS[k]] = float(r.tot)
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d02-2001", required=True)
    ap.add_argument("--d02-2011", required=True)
    ap.add_argument("--pop", default="data/raw/census/population_district.csv")
    ap.add_argument("--out", default="data/raw/census/migration_district.csv")
    a = ap.parse_args()
    lut = census_lookup()
    df = pd.DataFrame(read_d02(a.d02_2001, 2001, lut) + read_d02(a.d02_2011, 2011, lut))
    print(f"D-02 read: {df[df.year == 2001].shape[0]} districts (2001), {df[df.year == 2011].shape[0]} districts (2011)")

    pop_path = Path(a.pop)
    if not pop_path.exists():
        tpl = df[["district", "year"]].sort_values(["year", "district"]).assign(total_population="")
        tpl_path = pop_path.with_name("population_district_TEMPLATE.csv")
        tpl_path.parent.mkdir(parents=True, exist_ok=True)
        tpl.to_csv(tpl_path, index=False)
        sys.exit(f"\n{pop_path} not found.\nFill the empty total_population column of {tpl_path} from Census PCA "
                 f"(2001 + 2011 district population), save it as {pop_path} and run this again.")
    pop = pd.read_csv(pop_path)
    pop.columns = [c.strip().lower() for c in pop.columns]
    pop["cid"] = pop["district"].map(lambda s: lut.get(norm(s)))
    if pop.cid.isna().any():
        sys.exit(f"{pop_path}: unknown districts {sorted(pop[pop.cid.isna()].district.unique())}")
    pop["total_population"] = pd.to_numeric(pop["total_population"], errors="coerce")
    df["cid"] = df["district"].map(lambda s: lut.get(norm(s)))
    df = df.merge(pop[["cid", "year", "total_population"]], on=["cid", "year"], how="left")
    miss = df[df.total_population.isna()]
    if len(miss):
        sys.exit(f"total_population missing for: {miss[['district', 'year']].values.tolist()}")
    cols = ["district", "year", "total_population", "total_migrants", "migrants_intra_district",
            "migrants_inter_district", "migrants_inter_state", "migrants_outside_india"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["year", "district"])[cols].to_csv(a.out, index=False)
    print(f"written {a.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
