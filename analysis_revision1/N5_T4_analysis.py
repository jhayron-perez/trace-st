"""Night-5 T4 (R1 Major 2, parameter guidelines), stage 2: deliverables 1-4 (tables) from N5_T4_data.py outputs.

All statistics use prob_rule == "linear" configs only (night-5 decision after T0 PASS) and, where stated, the
admissible configs = harmonized retention (rule 3; rule 6: internal diagnostics only). Agreement with an expected
answer (headline share, truth error) is reported but never used to pick a rule value.

Rules fixed here before the tables were computed (pre-stated, operational):
- N/p bins: edges 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100 (log, half-decade above 0.01).
- Target N/p (window rule): for each REAL case, N/p*_case = lower edge of the lowest bin (>= 10 linear configs) whose
  admissible rate is >= 0.9 x the largest admissible rate over bins with >= 10 configs; target = max over the three
  real cases (T1 and T2 are not used, so their scale parameters stay out of sample).
- Central 80 %: 10th-90th percentile among admissible linear configs (lambda_a, l1_ratio in log10); categorical
  score_mode: the levels holding >= 10 % of the admissible configs. Shared box = intersection over the five cases;
  a parameter whose intersection is empty is reported as case-specific and left out of the box.
- LOCO: box from the other four cases (parameters with a non-empty 4-case intersection that overlaps the left-out
  case's sampled range); left-out admissible configs inside vs all admissible configs.
- Scale rules: r_rule = max(1, ceil(p90 of the per-step Chebyshev displacement / spaceres)) (the p90 is used as a
  robust maximum: the field trackers of PNW/Debby occasionally re-identify the feature with a jump; the strict max is
  reported as r_rule_max); box_rule = L_ext + 2 r_rule
  spaceres (half-max extent of the feature plus the stencil margin, so the interior cells that carry samples cover the
  feature); tw_rule = shortest window (steps >= 2) with N/p(box_rule, r_rule, tw) >= target on the case grid.

python N5_T4_analysis.py [--source night1|night5]      (PBS only)
Outputs (revision1/night5/analysis/T4/<source>/): T4_np_bins.csv, T4_np_target.json, T4_ranges_per_case.csv,
  T4_shared_box.csv, T4_loco.csv, T4_scale_rules.csv, T4_scale_fit.csv, T4_shared_ranges.md, T4_numbers.json;
  revision1/night5/analysis/T4/oos_scale_params.json (from --source night1 only, unless --write-oos)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from N5_T4_data import START, SYN_GRID, T1_EVENTS, T4, n_over_p  # noqa: E402
from trace_st import retention  # noqa: E402

CASES = ["t1", "t2", "debby", "pinatubo", "pnw"]
REAL = ["debby", "pinatubo", "pnw"]
NAMES = dict(t1="T1", t2="T2", debby="Debby", pinatubo="Pinatubo", pnw="PNW")
EDGES = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]
FREE_NUM = ["lambda_a", "l1_ratio", "eps_dbscan", "alpha"]
LOG = {"lambda_a", "l1_ratio"}
HEADLINE = dict(t1="share of steps on the true parent A", t2="density share on the true sources V1+V3",
                debby="vo share 5-15 d", pinatubo="SO2 (TMSO201) share 4-16 d", pnw="Z500+TCWV share 1-10 d")
ERRNAME = dict(t1="mean track distance to the true path (grid units)", t2="|alpha_est - alpha| (mean over 4 alphas)")


def load(source):
    T = {}
    for c in CASES:
        f = T4 / source / f"configs_{c}.csv"
        if not f.exists() and c in ("t1", "t2"):
            f = T4 / "night1" / f"configs_{c}.csv"
        if f.exists():
            t = pd.read_csv(f, low_memory=False)
            t["adm"] = t["adm_ret"].astype(bool)
            T[c] = t
    return T


def lin(t):
    return t[t.prob_rule == "linear"].copy()


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ------------------------------------------------------------------ deliverable 1
def np_bins(T):
    rows = []
    for c, t in T.items():
        L = lin(t)
        L["bin"] = pd.cut(L.np_ratio, EDGES, right=False)
        for b, g in L.groupby("bin", observed=True):
            A = g[g.adm]
            m, lo, hi = boot_ci(A.share)
            k = int(A.rank1.astype(float).sum()) if len(A) else 0
            wl, wh = wilson(k, len(A))
            r = dict(case=c, bin_lo=b.left, bin_hi=b.right, np_median=float(g.np_ratio.median()), n=len(g),
                     n_adm=len(A), adm_rate=len(A) / len(g), completion=float(g.completion.mean()),
                     share=m, share_lo=lo, share_hi=hi,
                     share_iqr=float(A.share.quantile(0.75) - A.share.quantile(0.25)) if len(A) > 3 else np.nan,
                     frac_rank1=k / len(A) if len(A) else np.nan, rank1_lo=wl, rank1_hi=wh)
            if "err" in A:
                e, elo, ehi = boot_ci(A.err)
                r.update(err=e, err_lo=elo, err_hi=ehi)
            rows.append(r)
    B = pd.DataFrame(rows)
    tgt = {}
    for c in REAL:
        g = B[(B.case == c) & (B.n >= 10)]
        if not len(g):
            continue
        ok = g[g.adm_rate >= 0.9 * g.adm_rate.max()]
        tgt[c] = float(ok.bin_lo.min())
    target = max(tgt.values()) if tgt else np.nan
    return B, dict(per_case=tgt, target=target, rule="lowest N/p bin (>=10 linear configs) with admissible rate >= 0.9 x "
                   "the case's maximum; target = max over Debby, Pinatubo, PNW")


# ------------------------------------------------------------------ deliverable 2
def ranges(T):
    rows = []
    for c, t in T.items():
        L = lin(t); A = L[L.adm]
        for p in FREE_NUM:
            x = A[p].astype(float); xa = L[p].astype(float)
            if p in LOG:
                x, xa = np.log10(x), np.log10(xa)
            q10, q90 = np.percentile(x, [10, 90]); a10, a90 = np.percentile(xa, [10, 90])
            rows.append(dict(case=c, param=p, scale="log10" if p in LOG else "linear", n_adm=len(A), n_lin=len(L),
                             lo=q10, hi=q90, sampled_min=float(xa.min()), sampled_max=float(xa.max()),
                             all_lo=a10, all_hi=a90,
                             selectivity=(q90 - q10) / (a90 - a10) if a90 > a10 else np.nan))
        lv = A.score_mode.value_counts(normalize=True)
        rows.append(dict(case=c, param="score_mode", scale="categorical", n_adm=len(A), n_lin=len(L),
                         levels=",".join(sorted(lv[lv >= 0.1].index)), sampled_levels=",".join(sorted(L.score_mode.unique()))))
        rows.append(dict(case=c, param="min_samples_dbscan", scale="fixed", n_adm=len(A), n_lin=len(L),
                         levels=",".join(str(int(v)) for v in sorted(L.min_samples_dbscan.unique()))))
    return pd.DataFrame(rows)


def box_from(R, cases):
    out = {}
    for p in FREE_NUM:
        g = R[(R.param == p) & (R.case.isin(cases))]
        lo, hi = g.lo.max(), g.hi.min()
        out[p] = dict(lo=float(lo), hi=float(hi), empty=bool(lo > hi), scale=g.scale.iloc[0],
                      binding_lo=g.loc[g.lo.idxmax(), "case"], binding_hi=g.loc[g.hi.idxmin(), "case"])
    g = R[(R.param == "score_mode") & (R.case.isin(cases))]
    sets = [set(s.split(",")) for s in g.levels]
    inter = set.intersection(*sets) if sets else set()
    out["score_mode"] = dict(levels=sorted(inter), empty=len(inter) == 0, scale="categorical")
    return out


def in_box(t, box, skip=()):
    m = np.ones(len(t), bool)
    used = []
    for p, b in box.items():
        if b["empty"] or p in skip:
            continue
        if p == "score_mode":
            m &= t.score_mode.isin(b["levels"]).values
        else:
            x = t[p].astype(float).values
            x = np.log10(x) if p in LOG else x
            m &= (x >= b["lo"] - 1e-9) & (x <= b["hi"] + 1e-9)
        used.append(p)
    return m, used


def loco(T, R):
    rows = []
    for c in CASES:
        if c not in T:
            continue
        others = [k for k in CASES if k != c and k in T]
        box = box_from(R, others)
        L = lin(T[c]); A = L[L.adm]
        skip = []
        for p in FREE_NUM:   # interval that misses the left-out case's sampled range cannot be tested on it
            b = box[p]
            if b["empty"]:
                continue
            x = L[p].astype(float); x = np.log10(x) if p in LOG else x
            if b["hi"] < x.min() or b["lo"] > x.max():
                skip.append(p)
        mA, used = in_box(A, box, skip)
        mL, _ = in_box(L, box, skip)
        r = dict(left_out=c, box_params=",".join(used), untestable=",".join(skip),
                 empty_in_4=",".join(p for p, b in box.items() if b["empty"]),
                 n_adm=len(A), n_in=int(mA.sum()), frac_in=float(mA.mean()) if len(A) else np.nan,
                 adm_rate_in=float(L[mL].adm.mean()) if mL.any() else np.nan, adm_rate_all=float(L.adm.mean()),
                 box=json.dumps({p: ({"levels": b["levels"]} if p == "score_mode" else
                                     {"lo": round(b["lo"], 4), "hi": round(b["hi"], 4), "empty": b["empty"]})
                                 for p, b in box.items()}))
        for nm, sub in (("all", A), ("in", A[mA]), ("out", A[~mA])):
            m, lo, hi = boot_ci(sub.share)
            r.update({f"share_{nm}": m, f"share_{nm}_lo": lo, f"share_{nm}_hi": hi,
                      f"rank1_{nm}": float(sub.rank1.astype(float).mean()) if len(sub) else np.nan,
                      f"sd_{nm}": float(sub.share.std()) if len(sub) > 1 else np.nan})
            if "err" in sub:
                e, elo, ehi = boot_ci(sub.err)
                r.update({f"err_{nm}": e, f"err_{nm}_lo": elo, f"err_{nm}_hi": ehi})
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ deliverable 3/4
def case_grid(c, t):
    if c == "t1":
        return [(SYN_GRID[0], SYN_GRID[1], e) for e in T1_EVENTS]
    if c == "t2":
        return [(SYN_GRID[0], SYN_GRID[1], (100.0, 100.0))]
    if c == "pnw":
        return [(C.PNW_LATS, C.PNW_LONS, START["pnw"])]
    b = np.load(t.bundle_npz.dropna().iloc[0])
    return [(np.asarray(b["ys"], float), np.asarray(b["xs"], float), START[c])]


def scale_rules(T, SD, target):
    rows, fit = [], []
    for c in CASES:
        if c not in T or c not in SD:
            continue
        t = T[c]; L = lin(t); A = L[L.adm]
        sd = SD[c]; dx = float(sd["spaceres"]); p = int(sd["p"])
        r_rule = int(max(1, np.ceil(sd["displacement"]["p90"] - 1e-9)))   # robust max (p90), see docstring
        r_max = int(sd["displacement"]["r_rule"])
        Lf = float(sd["size"]["L_ext"])
        box_rule = Lf + 2 * r_rule * dx
        grids = case_grid(c, t)
        tw_rule, np_at = None, None
        for tw in range(2, 400):
            _, _, q = n_over_p(box_rule, r_rule, tw, p, grids)
            if q >= target:
                tw_rule, np_at = tw, q
                break
        tw1 = next((tw for tw in range(2, 400) if n_over_p(box_rule, r_rule, tw, p, grids)[2] >= 1.0), None)
        unit = "h" if c == "debby" else "d"
        rows.append(dict(case=c, spaceres=dx, p=p, disp_max=sd["displacement"]["max"], disp_p90=sd["displacement"]["p90"],
                         disp_median=sd["displacement"]["median"], disp_how=sd["displacement"]["how"],
                         disp_check_max=sd.get("displacement_check", {}).get("max", np.nan),
                         disp_check_p90=sd.get("displacement_check", {}).get("p90", np.nan),
                         disp_check_cap_hits=sd.get("displacement_check", {}).get("cap_hits", np.nan),
                         L_ext=Lf, L_eq=float(sd["size"]["L_eq"]), size_how=sd["size"]["how"],
                         r_rule=r_rule, r_rule_max=r_max, box_rule=box_rule, target_np=target, tw_rule=tw_rule, tw_rule_np1=tw1, tw_unit=unit,
                         np_at_rule=np_at,
                         drawn_r=",".join(str(int(v)) for v in sorted(L.radius.unique())),
                         drawn_box=",".join(f"{v:g}" for v in sorted(L.box_size.unique())),
                         drawn_tw=",".join(str(int(v)) for v in sorted(L.timewindow.unique()))))
        for prm, rv in (("radius", r_rule), ("box_size", box_rule), ("timewindow", tw_rule)):
            lv = np.array(sorted(L[prm].astype(float).unique()))
            near = float(lv[np.argmin(np.abs(lv - rv))]) if rv is not None else np.nan
            inside = rv is not None and lv.min() <= rv <= lv.max()
            for v in lv:
                g = L[L[prm].astype(float) == v]
                fit.append(dict(case=c, param=prm, rule_value=rv, level=v, is_rule_level=v == near, rule_inside_drawn=inside,
                                n=len(g), n_adm=int(g.adm.sum()), adm_rate=float(g.adm.mean()),
                                frac_of_adm=float(g.adm.sum() / max(A.shape[0], 1)), frac_of_drawn=len(g) / len(L),
                                share_adm=float(g[g.adm].share.mean()) if g.adm.any() else np.nan,
                                rank1_adm=float(g[g.adm].rank1.astype(float).mean()) if g.adm.any() else np.nan,
                                err_adm=float(g[g.adm].err.mean()) if ("err" in g and g.adm.any()) else np.nan,
                                frac_adm_ge_rule=float((A[prm].astype(float) >= rv - 1e-9).mean()) if rv is not None else np.nan,
                                frac_drawn_ge_rule=float((L[prm].astype(float) >= rv - 1e-9).mean()) if rv is not None else np.nan))
        fit.append(dict(case=c, param="np_ratio", rule_value=target, level=np.nan, n=len(L), n_adm=len(A),
                        frac_adm_ge_rule=float((A.np_ratio >= target).mean()), frac_drawn_ge_rule=float((L.np_ratio >= target).mean()),
                        adm_rate=float(L[L.np_ratio >= target].adm.mean()) if (L.np_ratio >= target).any() else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(fit)


def t2_alpha_sweep():
    """Harmonized sweep for T2 (config retained if the rule holds at all 4 alphas)."""
    m = pd.read_csv(C.RUNS["t2"] / "results_members_final.csv", usecols=["trial_id", "alpha_mix", "ok", "n_steps", "prob_rule"])
    g = m.groupby(["trial_id", "alpha_mix"])
    st = pd.DataFrame(dict(ns=g.ok.sum(), nr=g.ok.size(),
                           ml=g.apply(lambda x: float(np.median(x.n_steps[x.ok == 1] - 1)) if (x.ok == 1).any() else 0.0),
                           rule=g.prob_rule.first())).reset_index()
    rows = []
    for mf in (0.5, 0.6, 2 / 3, 0.75):
        for lf in (0.6, 0.7, 0.8, 0.9):
            k = retention.harmonized_retained(st.ns, st.nr, st.ml, 30, min_member_frac=mf, min_len_frac=lf).values
            ok = pd.Series(k, index=st.index).groupby(st.trial_id).all()
            rl = st.groupby("trial_id").rule.first()
            rows.append(dict(case="t2", min_member_frac=round(mf, 3), min_len_frac=lf, n_retained=int(ok.sum()),
                             n_total=len(ok), n_retained_linear=int(ok[rl == "linear"].sum()), n_linear=int((rl == "linear").sum())))
    return pd.DataFrame(rows)


def sweep_cases(T):
    rows = []
    for c, t in T.items():
        if c == "t2":
            continue
        for mf in (0.5, 0.6, 2 / 3, 0.75):
            for lf in (0.6, 0.7, 0.8, 0.9):
                k = retention.harmonized_retained(t.n_success, t.n_requested, t.median_len, float(t.n_full.iloc[0]),
                                                  min_member_frac=mf, min_len_frac=lf).values
                li = (t.prob_rule == "linear").values
                rows.append(dict(case=c, min_member_frac=round(mf, 3), min_len_frac=lf, n_retained=int(k.sum()),
                                 n_total=len(t), n_retained_linear=int(k[li].sum()), n_linear=int(li.sum())))
    return pd.concat([pd.DataFrame(rows), t2_alpha_sweep()], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="night1")
    ap.add_argument("--write-oos", action="store_true")
    a = ap.parse_args()
    od = T4 / a.source
    T = load(a.source)
    SD = json.load(open(T4 / "scale_diagnostics.json"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    # counts
    cnt = []
    for c, t in T.items():
        L = lin(t)
        cnt.append(dict(case=c, n_total=len(t), n_old=int(t.old.astype(bool).sum()), n_harmonized=int(t.adm.sum()),
                        n_linear=len(L), n_old_linear=int(L.old.astype(bool).sum()), n_harmonized_linear=int(L.adm.sum())))
    CNT = pd.DataFrame(cnt); CNT.to_csv(od / "T4_retention_counts_all.csv", index=False); print(CNT.to_string())
    SW = sweep_cases(T); SW.to_csv(od / "T4_retention_sweep_all.csv", index=False)
    # 1
    B, tgt = np_bins(T)
    B.to_csv(od / "T4_np_bins.csv", index=False); json.dump(tgt, open(od / "T4_np_target.json", "w"), indent=1)
    print(B.round(3).to_string()); print(tgt)
    # 2
    R = ranges(T); R.to_csv(od / "T4_ranges_per_case.csv", index=False)
    box5 = box_from(R, [c for c in CASES if c in T])
    box3 = box_from(R, [c for c in REAL if c in T])
    SB = pd.DataFrame([dict(param=p, set="5 cases", **{k: (",".join(v) if isinstance(v, list) else v) for k, v in b.items()})
                       for p, b in box5.items()] +
                      [dict(param=p, set="3 real cases", **{k: (",".join(v) if isinstance(v, list) else v) for k, v in b.items()})
                       for p, b in box3.items()])
    SB.to_csv(od / "T4_shared_box.csv", index=False)
    LO = loco(T, R); LO.to_csv(od / "T4_loco.csv", index=False)
    print(R.round(3).to_string()); print(SB.to_string()); print(LO.drop(columns=["box"]).round(3).T.to_string())
    # 3
    SR, FIT = scale_rules(T, SD, tgt["target"])
    SR.to_csv(od / "T4_scale_rules.csv", index=False); FIT.to_csv(od / "T4_scale_fit.csv", index=False)
    print(SR.drop(columns=["disp_how", "size_how"]).round(3).T.to_string()); print(FIT.round(3).to_string())
    # 4
    if a.source == "night1" or a.write_oos:
        oos = dict(note="Scale parameters for T1 and T2 from the rules alone (N5_T4_analysis.py). Target N/p from the three "
                        "real cases only (T1/T2 bins not used). Other parameters: shared box (5 cases) and the strict "
                        "out-of-sample box from the 3 real cases; categorical score_mode levels; prob_rule linear; "
                        "min_samples_dbscan 2. lambda_a / l1_ratio intervals are log10.",
                   target_np=tgt, rules=dict(radius="max(1, ceil(p90 of per-step Chebyshev displacement / spaceres)); r_rule_max uses the max",
                                             box_size="L_ext (half-max extent of the feature at the start time) + 2 r spaceres",
                                             timewindow="shortest tw (steps >= 2) with N/p(box, r, tw) >= target on the case grid"),
                   box_5cases=box5, box_3real=box3)
        for c in ("t1", "t2"):
            s = SR[SR.case == c].iloc[0].to_dict()
            oos[c] = dict(radius=int(s["r_rule"]), radius_strict_max_variant=int(s["r_rule_max"]), box_size=float(s["box_rule"]), timewindow=f"{int(s['tw_rule'])}d", timewindow_if_target_np1=f"{int(s['tw_rule_np1'])}d",
                          np_at_rule=s["np_at_rule"], derivation=dict(
                              displacement=SD[c]["displacement"], size={k: SD[c]["size"][k] for k in SD[c]["size"] if k != "per_event"},
                              spaceres=SD[c]["spaceres"], p=SD[c]["p"]),
                          paper_drawn=dict(radius=s["drawn_r"], box_size=s["drawn_box"], timewindow=s["drawn_tw"]))
        json.dump(oos, open(T4 / "oos_scale_params.json", "w"), indent=1, default=float)
    nums = dict(counts=cnt, target=tgt, box5=box5, box3=box3)
    json.dump(nums, open(od / "T4_numbers.json", "w"), indent=1, default=str)
    # markdown
    md = [f"# T4 shared ranges and leave-one-case-out ({a.source}; linear configs; admissible = harmonized retention)\n\n",
          "Auto-generated by analysis_revision1/N5_T4_analysis.py.\n\n## Central 80 % of admissible configs per case\n\n",
          C.to_markdown(R.drop(columns=[c for c in ("all_lo", "all_hi") if c in R])),
          "\n## Shared box (intersection)\n\n", C.to_markdown(SB),
          "\n## Leave-one-case-out\n\n", C.to_markdown(LO.drop(columns=["box"])),
          "\n## N/p target\n\n", json.dumps(tgt), "\n"]
    (od / "T4_shared_ranges_auto.md").write_text("".join(md))


if __name__ == "__main__":
    main()
