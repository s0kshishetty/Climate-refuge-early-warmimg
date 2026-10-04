"""Step 2 – turn an EM-DAT export (India) into data/raw/events.csv

Usage:  python ml/convert_emdat.py data/raw/emdat_india.xlsx
        (csv also works)

Output columns: state, year, month, hazard, deaths, affected, homeless
One EM-DAT event that touches N states becomes N rows (numbers split equally).
Events where no state can be found are saved to data/raw/events_unmatched.csv
so you can fix them by hand (or just ignore them).

If the script cannot find a column, it prints all your column names – paste
that list to me/Claude and the mapping below can be adjusted in 1 minute.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from states import extract_states, normalize_state  # noqa: E402

# keyword(s) searched (lower-case 'contains') in your column names
COLUMN_KEYS = {
    "year":     ["start year"],
    "month":    ["start month"],
    "type":     ["disaster type"],
    "subtype":  ["disaster subtype"],
    "location": ["location"],
    "admin":    ["admin units"],
    "deaths":   ["total deaths"],
    "affected": ["total affected", "no. affected"],
    "homeless": ["no. homeless", "homeless"],
}
REQUIRED = ["year", "type", "location"]


def find_col(columns, keys):
    low = {c: str(c).lower() for c in columns}
    for k in keys:
        for c, l in low.items():
            if k in l:
                return c
    return None


def hazard_of(dtype, subtype):
    t = f"{dtype} {subtype}".lower()
    if "flood" in t:
        return "flood"
    if "cyclone" in t or "storm" in t or "tropical" in t:
        return "cyclone"
    if "drought" in t:
        return "drought"
    if "temperature" in t or "heat" in t or "cold" in t:
        return "heat"
    if "landslide" in t or "mass movement" in t:
        return "landslide"
    return "other"


def states_of(row, col_admin, col_loc):
    found = set()
    if col_admin and isinstance(row[col_admin], str):
        for nm in re.findall(r'"adm1_name"\s*:\s*"([^"]+)"', row[col_admin]):
            s = normalize_state(nm)
            if s:
                found.add(s)
    if not found and col_loc:
        found = extract_states(row[col_loc])
    return found


def main(path):
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    cols = {k: find_col(df.columns, v) for k, v in COLUMN_KEYS.items()}
    missing = [k for k in REQUIRED if cols[k] is None]
    if missing:
        print("Could not find columns for:", missing)
        print("Your columns are:\n", list(df.columns))
        sys.exit(1)
    print("Column mapping:", {k: v for k, v in cols.items()})

    out, unmatched = [], []
    for _, r in df.iterrows():
        st = states_of(r, cols["admin"], cols["location"])
        month = r[cols["month"]] if cols["month"] else 1
        month = int(month) if pd.notna(month) else 1
        base = dict(
            year=int(r[cols["year"]]), month=month,
            hazard=hazard_of(r[cols["type"]], r[cols["subtype"]] if cols["subtype"] else ""),
        )
        num = {}
        for k in ("deaths", "affected", "homeless"):
            v = r[cols[k]] if cols[k] else 0
            num[k] = float(v) if pd.notna(v) else 0.0
        if not st:
            unmatched.append({**base, **num, "location": r[cols["location"]]})
            continue
        for s in sorted(st):
            out.append({"state": s, **base, **{k: v / len(st) for k, v in num.items()}})

    ev = pd.DataFrame(out)
    ev.to_csv("data/raw/events.csv", index=False)
    pd.DataFrame(unmatched).to_csv("data/raw/events_unmatched.csv", index=False)
    print(f"\nEM-DAT rows: {len(df)}  -> state-event rows: {len(ev)}  | unmatched events: {len(unmatched)}")
    print("Years:", ev.year.min(), "-", ev.year.max())
    print(ev.groupby("hazard").size())
    print(ev.groupby("state").size().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
