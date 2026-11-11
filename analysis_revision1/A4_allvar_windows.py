"""Night-2: share of trajectory density for every variable per window, from A4's <SET>_long.csv.gz.

python A4_allvar_windows.py SETNAME   (PBS only)
Output: revision1/analysis_v2/A4_twins/<SETNAME>_allvar_windows.csv (mean, median over configs; for sets with
several twin years, the per-config value is first averaged over years).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402


def main():
    s = sys.argv[1]
    d = C.R1 / "analysis_v2" / "A4_twins"
    x = pd.read_csv(d / f"{s}_long.csv.gz", keep_default_na=False)
    x = x[(x["twin"] == "") & x["lead"].astype(str).str.startswith("w")]
    x["share_real"] = x["share_real"].astype(float)
    per = x.groupby(["var", "lead", "cfg"])["share_real"].mean().reset_index()
    out = per.groupby(["var", "lead"])["share_real"].agg(n_cfg="size", mean="mean", median="median").reset_index()
    out.to_csv(d / f"{s}_allvar_windows.csv", index=False)
    print(out.pivot_table(index="var", columns="lead", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
