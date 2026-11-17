"""Night-5 T5: adaptive group lasso (variant fit=agl) vs Elastic-Net, T1 (200 configs) and T2-EN (50 x alpha 0.2/0.6, M 20),
linear rule, same configs. EN T1 = night5/t5/en/T1_200; EN T2 = night5/t3/p03 (identical settings).
Timing = median elapsed_sec per config.   python N5_T5.py (PBS only) -> night5/analysis/T5_agl_vs_en.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from N5_T4_oos import t1, t2, N5  # noqa: E402


def sec(d, method=None):
    j = pd.read_csv(d / "results_jobs_final.csv")
    if method:
        j = j[j["method"] == method]
    return float(j["elapsed_sec"].median())


def main():
    ids1, ids2 = list(range(200)), list(range(50))
    rows = []
    for lab, d in (("Elastic-Net", N5 / "t5" / "en" / "T1_200"), ("AGL", N5 / "t5" / "agl" / "T1_200")):
        r = t1(d, ids1, lab); r["median_sec_per_config"] = sec(d, "granger"); rows.append(r)
    for lab, d in (("Elastic-Net", N5 / "t3" / "p03"), ("AGL", N5 / "t5" / "agl" / "T2en")):
        r = t2(d, ids2, lab); r["median_sec_per_config"] = sec(d); rows.append(r)
    df = pd.DataFrame(rows)
    (N5 / "analysis").mkdir(exist_ok=True)
    df.to_csv(N5 / "analysis" / "T5_agl_vs_en.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(3).to_string())


if __name__ == "__main__":
    main()
