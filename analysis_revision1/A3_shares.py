"""Night-2 A3: share of trajectory density per variable in the Pinatubo twin / combined runs.

Reads results_jobs_final.csv of each A3 run and the T5r baseline, and reports the median and mean over
configs of density_full_frac::<var> (whole trajectory) and density_range_frac::<var> (the paper's day range).
python A3_shares.py   (PBS only)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402

RUNS = {"baseline T5r": C.R1 / "T5r_pinatubo",
        "+TMSO203 (SO2 twin)": C.R1 / "A3_pinatubo_twin" / "so2twin_TMSO203",
        "+BURDENSO403 (SO4 twin)": C.R1 / "A3_pinatubo_twin" / "so4twin_BURDENSO403",
        "+TMSO203 +BURDENSO403 +FLUT": C.R1 / "A3_pinatubo_twin" / "combined_TMSO203_BURDENSO403_FLUT"}


def main():
    out = C.outdir("A3_shares")
    rows = []
    for name, d in RUNS.items():
        f = d / "results_jobs_final.csv"
        if not f.exists():
            print("missing", f); continue
        x = pd.read_csv(f)
        for kind in ("full", "range"):
            for c in [c for c in x.columns if c.startswith(f"density_{kind}_frac::")]:
                v = x[c]
                rows.append(dict(run=name, kind=kind, var=c.split("::")[1], n_cfg=int(v.notna().sum()),
                                 median=float(v.median()), mean=float(v.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(out / "A3_shares.csv", index=False)
    print(df.pivot_table(index=["kind", "var"], columns="run", values="mean").round(3).to_string())
    print(df.pivot_table(index=["kind", "var"], columns="run", values="median").round(3).to_string())


if __name__ == "__main__":
    main()
