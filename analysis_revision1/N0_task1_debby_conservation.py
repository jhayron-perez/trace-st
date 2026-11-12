"""Night-3 N0 task 1: Debby vorticity-type share conservation.

Baseline = T4r_debby_vorebuilt (vars Tb, vo, precip), A2 twin runs = A2_debby_twin/vo_<Y> (Tb, vo, precip, VO_P<Y>).
Configs = the 75 retained in the rebuilt baseline (A4_twins.retained_ids('debby')).
Share definition / binning identical to A4_twins.py (box-footprint density, step 0 excluded,
lead day k = hourly steps 24(k-1)+1..24k; window share = share of window-integrated density).
For each lead (1..15) and window (w1-4, w5-15), per twin year and pooled (year-averaged per config):
  baseline vo share, A2 real vo, A2 twin vo, A2 real+twin, and paired diff (real+twin - baseline) with
  bootstrap CI over configs (10,000, seed 0).
python N0_task1_debby_conservation.py   (PBS only)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import A4_twins as A  # noqa: E402

OUT = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night3/N0/task1")


def shares(D):
    rows = {}
    for L in range(1, D.shape[0]):
        t = D[L].sum()
        if t > 0:
            rows[str(L)] = D[L] / t
    for w0, w1 in A.WINDOWS["debby"]:
        Dw = D[w0:w1 + 1].sum(0)
        if Dw.sum() > 0:
            rows[f"w{w0}-{w1}"] = Dw / Dw.sum()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = A.retained_ids("debby")
    Db = A.densities_bundles(A.R1 / "T4r_debby_vorebuilt", 3, ids, 32, hourly=True)
    rows = []
    for cfg, D in Db.items():
        for lead, sh in shares(D).items():
            rows.append(dict(src="baseline", year="base", cfg=cfg, lead=lead, vo=sh[1], twin=np.nan,
                             Tb=sh[0], precip=sh[2]))
    for y in A.A2_YEARS:
        Dy = A.densities_bundles(A.R1 / "A2_debby_twin" / f"vo_{y}", 4, ids, 32, hourly=True)
        for cfg, D in Dy.items():
            for lead, sh in shares(D).items():
                rows.append(dict(src="A2", year=str(y), cfg=cfg, lead=lead, vo=sh[1], twin=sh[3], Tb=sh[0],
                                 precip=sh[2]))
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "task1_long.csv.gz", index=False)
    base = L[L.src == "baseline"].set_index(["cfg", "lead"])
    a2 = L[L.src == "A2"].copy()
    a2["sum"] = a2["vo"] + a2["twin"]
    # check against the existing A4 long table (same definition)
    ref = pd.read_csv(A.R1 / "analysis_v2/A4_twins/debby_A2_long.csv.gz", keep_default_na=False)
    ref = ref[ref["twin"] != ""].copy(); ref["year"] = ref["year"].astype(str)
    m = a2.merge(ref[["year", "cfg", "lead", "share_real", "share_twin"]].astype({"lead": str}),
                 on=["year", "cfg", "lead"])
    print("check vs A4 long: n", len(m), "max|dvo|", float((m.vo - m.share_real.astype(float)).abs().max()),
          "max|dtwin|", float((m.twin - m.share_twin.astype(float)).abs().max()), flush=True)
    out = []
    groups = [(str(y), a2[a2.year == str(y)]) for y in A.A2_YEARS]
    pooled = a2.groupby(["cfg", "lead"])[["vo", "twin", "sum", "Tb", "precip"]].mean().reset_index()
    pooled["year"] = "pooled"
    groups.append(("pooled", pooled))
    for yname, g in groups:
        for lead, gl in g.groupby("lead"):
            gl = gl.set_index("cfg")
            b = base.xs(lead, level="lead") if lead in base.index.get_level_values("lead") else None
            if b is None:
                continue
            common = gl.index.intersection(b.index)
            gl = gl.loc[common]; bb = b.loc[common]
            d = gl["sum"] - bb["vo"]
            mm, lo, hi = A.boot_ci(d.values)
            mr, lor, hir = A.boot_ci((gl["vo"] - bb["vo"]).values)
            out.append(dict(year=yname, lead=lead, n_cfg=len(common), base_vo=bb["vo"].mean(),
                            a2_real_vo=gl["vo"].mean(), a2_twin_vo=gl["twin"].mean(), a2_real_plus_twin=gl["sum"].mean(),
                            ratio_sum_over_base=gl["sum"].mean() / bb["vo"].mean(),
                            diff_sum_minus_base=mm, diff_lo=lo, diff_hi=hi,
                            diff_real_minus_base=mr, diffr_lo=lor, diffr_hi=hir,
                            base_Tb=bb["Tb"].mean(), a2_Tb=gl["Tb"].mean(), base_precip=bb["precip"].mean(),
                            a2_precip=gl["precip"].mean()))
    S = pd.DataFrame(out)
    S["lead_sort"] = S["lead"].map(lambda s: 100 + int(s[1:].split("-")[0]) if s.startswith("w") else int(s))
    S = S.sort_values(["year", "lead_sort"]).drop(columns="lead_sort")
    S.to_csv(OUT / "task1_vo_conservation.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[S.year == "pooled"].round(3).to_string(index=False))
    print(S[S.lead.str.startswith("w")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
