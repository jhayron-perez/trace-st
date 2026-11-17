"""Night-5 T3: synthetic extension (T2-EN, 50 configs x alpha {0.2, 0.6}, M 20, linear rule) vs number of variables p.

Runs: revision1/night5/t3/p{03,05,08,12,16} (fixed composition, pre-declared rule 8; delayed beta 0.2 by rule 4).
Variable classes by name: A, C = true parents; B = target (its own density); N* noise; R*_B response;
M* moving blob (random placement); L*_A/C delayed child (lag 2); Q*_A/C nonlinear (rectified).
Metrics per (config, alpha): share of trajectory density per class (of the total), alpha recovery
A / (A + C) and its absolute error, completion (fraction of members with a full-length track), runtime per
config (results_jobs elapsed), and link-level selection rate per class from TST_LINKSTATS (fraction of the
regressions of B with any lag-1 link from a variable of that class, averaged over the class's variables).
Means over configs with 95 % bootstrap CIs (configs resampled; alphas pooled unless split).

python N5_T3.py   (PBS only) -> night5/analysis/T3_*.csv, figures_v5/synthetic/S_extension.{pdf,png}
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
from A4_twins import boot_ci  # noqa: E402

N5 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night5")
T3 = N5 / "t3"
OUT = N5 / "analysis"
PS = [3, 5, 8, 12, 16]
CLASSES = ["true", "target", "noise", "response", "moving", "delayed", "nonlinear"]
MIX16 = ["noise", "response", "moving", "delayed", "nonlinear", "noise", "response", "moving", "delayed",
         "nonlinear", "noise", "response", "moving"]


def cls(name):
    if name in ("A", "C"):
        return "true"
    if name == "B":
        return "target"
    return {"N": "noise", "R": "response", "M": "moving", "L": "delayed", "Q": "nonlinear"}[name[0]]


def load(p):
    d = T3 / f"p{p:02d}"
    for nm in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
        if (d / nm).exists():
            return d, pd.read_csv(d / nm)
    return d, None


def per_config(p, x):
    if p == 3:
        dens = {"A": x["density_integral_var1"], "B": x["density_integral_var2"], "C": x["density_integral_var3"]}
    else:
        dens = {c.replace("t7_density_", ""): x[c] for c in x.columns if c.startswith("t7_density_")}
    D = pd.DataFrame(dens)
    tot = D.sum(1)
    out = pd.DataFrame({"trial_id": x["trial_id"], "alpha": x["alpha_mix"]})
    for k in CLASSES:
        cols = [c for c in D.columns if cls(c) == k]
        out[f"share_{k}"] = D[cols].sum(1) / tot if cols else np.nan
    out["alpha_rec"] = D["A"] / (D["A"] + D["C"])
    out["alpha_err"] = (out["alpha_rec"] - out["alpha"]).abs()
    out["completion"] = x["t7_completion_rate"] if "t7_completion_rate" in x else np.nan
    out["sec_per_traj"] = x["t7_sec_per_traj"] if "t7_sec_per_traj" in x else np.nan
    out["samples_per_feature"] = x["t7_samples_per_feature"] if "t7_samples_per_feature" in x else np.nan
    return out


def completion_from_members(d, nsteps):
    f = d / "results_members_final.csv"
    if not f.exists():
        f = d / "results_members_snapshot.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    lc = "n_centers" if "n_centers" in m else ("n_steps" if "n_steps" in m else None)
    if lc is None:
        return None
    full = m[lc] >= (nsteps + (1 if lc == "n_centers" else 0))
    return full.groupby([m["trial_id"], m["alpha_mix"]]).mean().rename("completion").reset_index()


def linkstats(d, names):
    """Per class: mean over its variables of the fraction of B-regressions with any link from that variable."""
    n = 0
    sel = np.zeros(len(names))
    for f in glob.glob(str(d / "linkstats" / "links_*.txt")):
        for line in open(f):
            q = line.split()
            if len(q) < 3 or q[0] != "1":
                continue
            n += 1
            if q[2] != "-":
                for b in map(int, q[2].split(",")):
                    if b < len(names):
                        sel[b] += 1
    if n == 0:
        return {}
    r = {"n_regressions": n}
    for k in CLASSES:
        idx = [i for i, v in enumerate(names) if cls(v) == k]
        if idx and k != "target":
            r[f"sel_{k}"] = float(sel[idx].mean() / n)
    return r


def var_names(p):
    return ["A", "B", "C"] + _mix_names(MIX16[: p - 3])


def _mix_names(mix):
    cnt, out = {}, []
    for t in mix:
        k = cnt.get(t, 0); cnt[t] = k + 1
        out.append({"noise": f"N{k + 1}", "response": f"R{k + 1}_B", "moving": f"M{k + 1}",
                    "delayed": f"L{k + 1}_{'A' if k % 2 == 0 else 'C'}",
                    "nonlinear": f"Q{k + 1}_{'A' if k % 2 == 0 else 'C'}"}[t])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summ, links = [], [], []
    for p in PS:
        d, x = load(p)
        if x is None:
            print("missing", d); continue
        pc = per_config(p, x)
        cm = completion_from_members(d, 30)
        if cm is not None and pc["completion"].isna().all():
            pc = pc.drop(columns="completion").merge(cm.rename(columns={"alpha_mix": "alpha"}), on=["trial_id", "alpha"], how="left")
        jf = d / "results_jobs_final.csv"
        if not jf.exists():
            jf = d / "results_jobs_snapshot.csv"
        if jf.exists():
            j = pd.read_csv(jf)
            if "elapsed_sec" in j:
                pc = pc.merge(j[["trial_id", "elapsed_sec"]], on="trial_id", how="left")
        pc["p"] = p
        rows.append(pc)
        for a in ("pooled", 0.2, 0.6):
            g = pc if a == "pooled" else pc[np.isclose(pc["alpha"], a)]
            per_cfg = g.groupby("trial_id").mean(numeric_only=True)
            r = dict(p=p, alpha=a, n_cfg=len(per_cfg))
            for k in CLASSES + ["alpha_err", "completion", "elapsed_sec", "samples_per_feature"]:
                col = f"share_{k}" if k in CLASSES else k
                if col in per_cfg and per_cfg[col].notna().any():
                    m, lo, hi = boot_ci(per_cfg[col].to_numpy())
                    r[col] = m; r[col + "_lo"] = lo; r[col + "_hi"] = hi
                    r[col + "_median"] = float(per_cfg[col].median())
            summ.append(r)
        ls = linkstats(d, var_names(p))
        if ls:
            links.append(dict(p=p, **ls))
    if not rows:
        return
    P = pd.concat(rows); Sm = pd.DataFrame(summ); Lk = pd.DataFrame(links)
    P.to_csv(OUT / "T3_per_config.csv", index=False); Sm.to_csv(OUT / "T3_summary.csv", index=False)
    Lk.to_csv(OUT / "T3_linkstats.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        cols = ["p", "alpha", "n_cfg"] + [c for c in Sm.columns if c.startswith("share_") and not c.endswith(("_lo", "_hi", "_median"))] \
            + ["alpha_err", "alpha_err_median", "completion", "elapsed_sec"]
        print(Sm[[c for c in cols if c in Sm]].round(3).to_string()); print(Lk.round(3).to_string())
    figure(Sm, Lk)


def figure(Sm, Lk):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    S.set_style()
    fig, axs = plt.subplots(2, 2, figsize=(S.DOUBLE, 0.72 * S.DOUBLE))
    (ax_a, ax_b), (ax_c, ax_d) = axs
    # (a) schematic
    ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10); ax_a.axis("off")
    box = dict(A=(1.4, 8.0), C=(1.4, 4.4), B=(5.0, 6.2), R=(8.6, 6.2), L=(5.0, 9.3), Q=(5.0, 3.0), N=(8.6, 9.3), M=(8.6, 3.0))
    txt = dict(A="A", C="C", B="B (target)", R="Response\nR ← B", L="Delayed\nL ← A,C (lag 2)", Q="Nonlinear\nQ ← max(A − ½, 0)",
               N="Noise N", M="Moving\nblob M")
    cl = dict(A="true", C="true", B="true", R="response", L="delayed", Q="nonlinear", N="noise", M="moving")
    for k, (xx, yy) in box.items():
        c = S.CLASS_COLORS[cl[k]] if k != "B" else "#444444"
        ax_a.add_patch(FancyBboxPatch((xx - 1.15, yy - 0.7), 2.3, 1.4, boxstyle="round,pad=0.08", fc="white", ec=c, lw=1.2))
        ax_a.text(xx, yy, txt[k], ha="center", va="center", fontsize=5.8, color=c if k != "B" else "k")

    def arrow(a, b, c, ls="-"):
        ax_a.annotate("", xy=box[b], xytext=box[a], arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, ls=ls,
                      shrinkA=13, shrinkB=13))
    arrow("A", "B", S.CLASS_COLORS["true"]); arrow("C", "B", S.CLASS_COLORS["true"])
    arrow("B", "R", S.CLASS_COLORS["response"], "--"); arrow("A", "L", S.CLASS_COLORS["delayed"], "--")
    arrow("C", "Q", S.CLASS_COLORS["nonlinear"], "--")
    ax_a.text(5.0, 0.6, "solid: causal for B (lag 1); dashed: decoy driven by a true variable, not a parent of B",
              ha="center", fontsize=5.2)
    # (b) stacked shares vs p (pooled alphas), CI on the true-parent share
    s = Sm[Sm["alpha"] == "pooled"].sort_values("p")
    xs = np.arange(len(s)); bottom = np.zeros(len(s))
    for k in ["true", "target", "noise", "response", "moving", "delayed", "nonlinear"]:
        c = f"share_{k}"
        if c not in s:
            continue
        y = s[c].fillna(0).to_numpy()
        ax_b.bar(xs, y, 0.65, bottom=bottom, color=S.CLASS_COLORS.get(k, "#DDDDDD"), lw=0,
                 label=S.CLASS_LABELS.get(k, "Target B (own density)"))
        if k == "true":
            ax_b.errorbar(xs, y, yerr=[y - s[c + "_lo"].to_numpy(), s[c + "_hi"].to_numpy() - y], fmt="none", color="k", lw=0.6, capsize=1.5)
        bottom += y
    ax_b.set_xticks(xs); ax_b.set_xticklabels(s["p"].astype(int)); ax_b.set_xlabel("Number of variables p")
    ax_b.set_ylabel("Share of trajectory density"); ax_b.set_ylim(0, 1.0)
    ax_b.legend(loc="lower left", fontsize=5.2, ncol=2)
    # (c) alpha error vs p, per alpha
    for a, mk in ((0.2, "o"), (0.6, "s")):
        q = Sm[Sm["alpha"] == a].sort_values("p")
        if not len(q):
            continue
        ax_c.errorbar(q["p"], q["alpha_err"], yerr=[q["alpha_err"] - q["alpha_err_lo"], q["alpha_err_hi"] - q["alpha_err"]],
                      marker=mk, ms=3, color="#0072B2" if a == 0.2 else "#D55E00", lw=1, capsize=1.5, label=f"α = {a}")
    ax_c.set_xlabel("Number of variables p"); ax_c.set_ylabel("|α recovered − α| (mean over configs)")
    ax_c.set_xticks(PS); ax_c.set_ylim(0, None); ax_c.legend(loc="upper left")
    # (d) selection rate vs density share per class
    if len(Lk):
        mks = {3: "o", 5: "s", 8: "^", 12: "D", 16: "v"}
        for _, r in Lk.iterrows():
            srow = s[s["p"] == r["p"]].iloc[0]
            for k in ["true", "noise", "response", "moving", "delayed", "nonlinear"]:
                if f"sel_{k}" in r and np.isfinite(r[f"sel_{k}"]) and f"share_{k}" in srow and np.isfinite(srow[f"share_{k}"]):
                    nvar = 2 if k == "true" else max(1, sum(1 for t in MIX16[: int(r["p"]) - 3] if t == k))
                    ax_d.scatter(r[f"sel_{k}"], srow[f"share_{k}"] / nvar, s=14, marker=mks[int(r["p"])],
                                 color=S.CLASS_COLORS[k], edgecolor="k", lw=0.3)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], ls="", marker="o", color=S.CLASS_COLORS[k], label=S.CLASS_LABELS[k]) for k in
             ["true", "noise", "response", "moving", "delayed", "nonlinear"]]
        h += [Line2D([], [], ls="", marker=m, color="0.5", label=f"p = {p}") for p, m in mks.items()]
        ax_d.legend(handles=h, loc="upper left", fontsize=5, ncol=2)
    ax_d.set_xlabel("Link-level selection rate (per variable)"); ax_d.set_ylabel("Density share per variable")
    ax_d.set_xlim(-0.02, 1.02); ax_d.set_ylim(0, None)
    S.panel_labels([ax_a, ax_b, ax_c, ax_d], x=-0.12)
    fig.tight_layout()
    print("fig", S.save(fig, "synthetic", "S_extension"))


if __name__ == "__main__":
    main()
