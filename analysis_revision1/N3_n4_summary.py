"""Night-3 N4: decoy sweep (beta x phi), window sweep (W) and p8 for baseline, bf05 and hybv.

Reads revision1/night3/n4/<variant>/<run>/results_alphas_{final,snapshot}.csv and, for the beta 0.8 / phi 0.8
cell and p8, the test-bed runs night3/bed{,M10}/<variant>/{T7dec,T7p8}. Medians over configs.
python N3_n4_summary.py   (PBS only). Output: night3/analysis/n4_summary.csv
"""
from pathlib import Path

import pandas as pd

N3 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night3")
COLS = ["t7_share_density_v1v3", "t7_share_density_decoy", "t7_share_density_noise", "t7_share_density_moving",
        "t7_alpha_abs_err_density_13", "t7_track_error_median", "t7_completion_rate", "is_valid_physical"]


def load(d):
    for n in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
        if (d / n).exists():
            return pd.read_csv(d / n)
    return None


def row(variant, run, d, src):
    x = load(d)
    if x is None or not len(x):
        return None
    r = dict(variant=variant, run=run, source=src, n_cfg=len(x))
    for c in COLS:
        if c in x:
            r[c.replace("t7_", "").replace("share_density_", "sh_")] = float(x[c].median() if c != "is_valid_physical" else x[c].mean())
    return r


def main():
    rows = []
    for vd in sorted((N3 / "n4").iterdir()):
        for rd in sorted(p for p in vd.iterdir() if p.is_dir()):
            rows.append(row(vd.name, rd.name, rd, "n4"))
    for v, bed in (("base", "bed"), ("bf05", "bed"), ("hybv", "bed"), ("base", "bedM10")):
        for t, name in (("T7dec", "dec_b08_phi08"), ("T7p8", "T7p8")):
            rows.append(row(v if bed == "bed" else "base_M10", name, N3 / bed / v / t, bed))
    df = pd.DataFrame([r for r in rows if r])
    (N3 / "analysis").mkdir(exist_ok=True)
    df.to_csv(N3 / "analysis" / "n4_summary.csv", index=False)
    print(df.sort_values(["run", "variant"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
