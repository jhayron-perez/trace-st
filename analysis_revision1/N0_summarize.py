"""Night-3 N0: compact markdown tables from the N0 task CSVs (PBS only).
Writes revision1/night3/N0/N0_tables.md. Also byte-compiles N0_task3_debby_retrace.py (syntax check only)."""
import py_compile
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402

N0 = C.R1 / "night3/N0"
md = []
f3 = lambda x: f"{x:.3f}"


def ci(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# task 1
t1 = pd.read_csv(N0 / "task1/task1_vo_conservation.csv")
md.append("## Task 1: Debby vo-type share conservation (75 configs; pooled = year-averaged per config)\n")
p = t1[t1.year == "pooled"]
tab = pd.DataFrame(dict(lead=p.lead, base_vo=p.base_vo.map(f3), A2_real=p.a2_real_vo.map(f3), A2_twin=p.a2_twin_vo.map(f3),
                        real_plus_twin=p.a2_real_plus_twin.map(f3), ratio=p.ratio_sum_over_base.map(f3),
                        sum_minus_base=[ci(*r) for r in p[["diff_sum_minus_base", "diff_lo", "diff_hi"]].values],
                        base_Tb=p.base_Tb.map(f3), A2_Tb=p.a2_Tb.map(f3), base_precip=p.base_precip.map(f3),
                        A2_precip=p.a2_precip.map(f3)))
md.append(C.to_markdown(tab))
w = t1[t1.lead.str.startswith("w")]
tab = pd.DataFrame(dict(year=w.year, window=w.lead, n=w.n_cfg, base_vo=w.base_vo.map(f3), A2_real=w.a2_real_vo.map(f3),
                        A2_twin=w.a2_twin_vo.map(f3), real_plus_twin=w.a2_real_plus_twin.map(f3),
                        sum_minus_base=[ci(*r) for r in w[["diff_sum_minus_base", "diff_lo", "diff_hi"]].values],
                        real_minus_base=[ci(*r) for r in w[["diff_real_minus_base", "diffr_lo", "diffr_hi"]].values]))
md.append("\n" + C.to_markdown(tab))

# task 2
t2 = pd.read_csv(N0 / "task2/task2_param_bins.csv")
t2 = t2[~t2.param.isin(["centroid_dist_to_volcano_km", "N_features", "stencil", "n_ref"])]
pc = pd.read_csv(N0 / "task2/task2_per_config.csv.gz")
md.append("\n## Task 2: parameter dependence of diff (per-config, year-averaged); bins = quartiles (continuous) or levels\n")
md.append("Per set/var/window: overall mean diff, then for each parameter the Spearman rho(param, diff) over configs, "
          "and the lowest / highest bin mean. `+` = bins whose 95 % CI > 0 (count/of bins).\n")
from scipy.stats import spearmanr
rows = []
for (s, v, wdw), g in t2.groupby(["set", "var", "window"]):
    q = pc[(pc.set == s) & (pc["var"] == v) & (pc.lead == wdw)]
    for prm, gg in g.groupby("param"):
        x = pd.to_numeric(q[prm], errors="coerce") if prm in q else pd.Series(dtype=float)
        if x.notna().sum() > 3 and x.nunique() > 1:
            rho = spearmanr(x, q["diff"], nan_policy="omit")[0]
        elif prm in q:   # categorical (2 levels): code as 0/1 by sorted level
            lev = sorted(q[prm].astype(str).unique()); rho = spearmanr(q[prm].astype(str).map(lev.index), q["diff"])[0]
        else:
            rho = np.nan
        lo = gg.loc[gg.mean_diff.idxmin()]; hi = gg.loc[gg.mean_diff.idxmax()]
        rows.append(dict(set=s, var=v, window=wdw, mean_diff=q["diff"].mean(), param=prm, rho=rho,
                         low_bin=f"{lo['bin']}: {lo.mean_diff:+.3f} (n {lo.n})",
                         high_bin=f"{hi['bin']}: {hi.mean_diff:+.3f} (n {hi.n})",
                         ci_gt0=f"{int(gg.ci_gt0.sum())}/{len(gg)}", ci_lt0=f"{int(gg.ci_lt0.sum())}/{len(gg)}"))
T2 = pd.DataFrame(rows)
T2.to_csv(N0 / "task2/task2_param_spearman.csv", index=False)
# compact: per set/var/window: top-3 |rho| params
comp = []
for k, g in T2.groupby(["set", "var", "window"]):
    g = g.reindex(g.rho.abs().sort_values(ascending=False).index).head(3)
    comp.append(dict(set=k[0], var=k[1], window=k[2], mean_diff=f"{g.mean_diff.iloc[0]:+.3f}",
                     top_params="; ".join(f"{r.param} rho {r.rho:+.2f} [{r.low_bin} .. {r.high_bin}]"
                                          for r in g.itertuples())))
md.append(C.to_markdown(pd.DataFrame(comp)))
# where does any bin have CI > 0 for sets where overall diff <= 0?
pos = t2[t2.ci_gt0].merge(pd.DataFrame(comp)[["set", "var", "window", "mean_diff"]], on=["set", "var", "window"])
pos = pos.rename(columns={"mean_diff_x": "bin_mean", "mean_diff_y": "overall"}); pos = pos[pos.overall.astype(float) <= 0]
md.append("\nBins with CI > 0 where the overall mean diff is <= 0:\n")
md.append(C.to_markdown(pos[["set", "var", "window", "param", "bin", "n", "bin_mean", "lo", "hi", "overall"]])
          if len(pos) else "none\n")

# task 3
t3 = pd.read_csv(N0 / "task3/task3_pnw_links_summary.csv")
md.append("\n## Task 3: PNW link level (A1 runs, 233 configs x 5 years; regressions k = 1..29; child != variable)\n")
a = t3[(t3.ratio_bin == "all") & (~t3.child_is_self)]
piv = []
for (v, wdw), g in a.groupby(["var", "window"]):
    r = g[g.role == "real"].iloc[0]; t = g[g.role == "twin"].iloc[0]
    piv.append(dict(var=v, window=wdw, n_reg_real=int(r.n_reg), frac_any_real=f3(r.frac_any_mean),
                    frac_any_twin=f3(t.frac_any_mean), nzfrac_real=f3(r.nz_frac_block), nzfrac_twin=f3(t.nz_frac_block),
                    absb_med_real=f"{r.absbeta_median:.4f}", absb_med_twin=f"{t.absbeta_median:.4f}",
                    absb_p90_real=f"{r.absbeta_p90:.4f}", absb_p90_twin=f"{t.absbeta_p90:.4f}"))
md.append(C.to_markdown(pd.DataFrame(piv)))
a = t3[(t3.ratio_bin == "all") & (t3.child_is_self)]
piv = []
for (v, wdw), g in a.groupby(["var", "window"]):
    r = g[g.role == "real"].iloc[0]; t = g[g.role == "twin"].iloc[0]
    piv.append(dict(var=v, window=wdw, n_reg_real=int(r.n_reg), n_reg_twin=int(t.n_reg), frac_any_real=f3(r.frac_any_mean),
                    frac_any_twin=f3(t.frac_any_mean), nzfrac_real=f3(r.nz_frac_block), nzfrac_twin=f3(t.nz_frac_block)))
md.append("\nSame, regressions whose child is the variable itself (own-lag block):\n")
md.append(C.to_markdown(pd.DataFrame(piv)))
b = t3[(t3.ratio_bin != "all") & (~t3.child_is_self)]
sv = pd.read_csv(N0 / "task3/task3_pnw_share_by_ratio.csv")
piv = []
for (v, wdw, rb), g in b.groupby(["var", "window", "ratio_bin"]):
    r = g[g.role == "real"].iloc[0]; t = g[g.role == "twin"].iloc[0]
    s = sv[(sv["var"] == v) & (sv.lead == wdw) & (sv.ratio_bin == rb)]
    piv.append(dict(var=v, window=wdw, ratio_bin=rb, n_cfg=int(r.n_cfg), frac_any_real=f3(r.frac_any_mean),
                    frac_any_twin=f3(t.frac_any_mean), nzfrac_real=f3(r.nz_frac_block), nzfrac_twin=f3(t.nz_frac_block),
                    share_real=f3(s.real.iloc[0]) if len(s) else "", share_twin=f3(s.twin.iloc[0]) if len(s) else "",
                    diff=f"{s['diff'].iloc[0]:+.3f}" if len(s) else ""))
P3 = pd.DataFrame(piv)
P3.to_csv(N0 / "task3/task3_pnw_by_ratio_compact.csv", index=False)
md.append("\nBy N_samples/N_features quartile (child != variable), with window shares from the A4 long table:\n")
md.append(C.to_markdown(P3))

# task 4
for case in ["debby", "pnw"]:
    f = N0 / f"task4/task4_{case}_summary.csv"
    if not f.exists():
        md.append(f"\n## Task 4 {case}: not available\n"); continue
    s = pd.read_csv(f)
    cols = ["group", "kind", "role", "window", "n_cfg", "across_cfg_r", "within_cfg_member_r", "H", "expH", "H_norm"]
    if case == "debby":
        cols += ["frac_mass_gt1000m_cfgmean", "frac_mass_gt1000m_pooled"]
    md.append(f"\n## Task 4 {case}: spatial reproducibility (twin_runs rows = mean over the 5 twin-year runs)\n")
    md.append(C.to_markdown(s[cols], floatfmt="{:.3f}"))
    pr = pd.read_csv(N0 / f"task4/task4_{case}_pairs_summary.csv")
    md.append("\nSame-config map correlation between runs (mean over configs, then over run pairs) and pooled-map r:\n")
    md.append(C.to_markdown(pr, floatfmt="{:.3f}"))
(N0 / "N0_tables.md").write_text("\n".join(md))
py_compile.compile(str(Path(__file__).resolve().parent / "N0_task3_debby_retrace.py"), doraise=True,
                   cfile=str(N0 / "logs/retrace_compile_check.pyc"))
print("retrace script compiles OK")
print("\n".join(md))
