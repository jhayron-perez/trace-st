"""Night-6 T2: extended real cases on the final variable sets vs the original sets (linear rule, harmonized retention).

Runs: final sets revision1/night6/t2/{debby_final/part*, pin_final, pnw_final}; original sets reused from night 5
(night5/t2/{debby_orig/part*, pin_orig, pnw_orig}). Final sets and roles as pre-declared (night-6 report, rule 2).
Statistics as N5_T2 (same functions): harmonized retention per run; per config, share of trajectory density per
variable in each window (step 0 excluded; PNW also "w0-10" = 1-10 d plus step 0, for the rank-1 caption) and per
lead day; mean over configs with a 95 % percentile bootstrap CI (10,000 resamples of configs, seed 0); fraction of
configs where each variable is rank 1. "own" = each set's retained configs, "paired" = configs retained in both.
Windows (rule 3): Debby 1-4, 5-15 d; Pinatubo 1-3, 4-16 d; PNW 1-10, 11-20, 21-30, 11-30 d.
Debby path accuracy: mean_of_member_mean_track_dist_km (distance to the IBTrACS track), original vs final.
A case is analysed when its final-set run is complete (every part has results_jobs_final.csv; PNW
hyperparam_search_summary.csv), unless --partial. Rows of the cases analysed replace their old rows in the CSVs.

python N6_T2.py [--nproc 32] [--partial] [--cases debby pinatubo pnw] [--test-n5 DIR] [--done-note]   (PBS only)
  --test-n5 DIR: development run on the night-5 extended sets (one more variable each), outputs to DIR.
Outputs: night6/analysis/T2_*.{csv,md}; figures figures_v6/S12_original_vs_final/; with --done-note,
night6/agents/C_done_<case>.txt (headline numbers).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import N5_T2 as N5T2  # noqa: E402
import rev1_common as C  # noqa: E402
from A4_twins import boot_ci  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
N5, N6 = R1 / "night5", R1 / "night6"
OUT = N6 / "analysis"
FIG6 = R1 / "figures_v6"
WINDOWS = dict(debby=[(1, 4), (5, 15)], pinatubo=[(1, 3), (4, 16)], pnw=[(1, 10), (11, 20), (21, 30), (11, 30)])
N5T2.WINDOWS = WINDOWS   # N5_T2.shares_from_D / stats_rows read the windows from their module (adds PNW 11-30 d)
PARENTS = dict(debby=["vo"], pinatubo=["TMSO201"], pnw=["Z500", "TCWV"])
DEB3, PIN4, PNW6 = N5T2.DEB3, N5T2.PIN4, N5T2.PNW6
# variable order = parent id order in the run outputs (base set, then --extra-vars / --extra-inputs in task order)
CASES = {
    "debby": dict(orig=(N5 / "t2" / "debby_orig", DEB3), final=(N6 / "t2" / "debby_final", DEB3 + ["CAPE", "DIV200", "MSLP"]),
                  roles=dict(vo="parent", Tb="intermediate", precip="intermediate", CAPE="competitor",
                             DIV200="competitor", MSLP="child", WS10="child"), title="TS Debby 2006"),
    "pinatubo": dict(orig=(N5 / "t2" / "pin_orig", PIN4), final=(N6 / "t2" / "pin_final", PIN4 + ["TMSO203", "BURDENSO403", "T050"]),
                     roles=dict(TMSO201="parent", BURDENSO401="intermediate", AEROD_v="intermediate",
                                TMH2SO401="intermediate", TMSO203="competitor", BURDENSO403="competitor", T050="child",
                                FLUT="child", FSDS="child"), title="Pinatubo 1991"),
    "pnw": dict(orig=(N5 / "t2" / "pnw_orig", PNW6), final=(N6 / "t2" / "pnw_final", PNW6 + ["T2M"]),
                roles=dict(Z500="parent", TCWV="parent", MTNLWRF="intermediate", MSLHF="intermediate",
                           MSSHF="intermediate", Z10="competitor", T2M="child", BLH="child"), title="PNW heat wave 2021"),
}
# --test-n5: night-5 extended sets in place of the final sets
N5_EXT = dict(debby=(N5 / "t2" / "debby_ext", N5T2.CASES["debby"]["ext"][1]),
              pinatubo=(N5 / "t2" / "pin_ext", N5T2.CASES["pinatubo"]["ext"][1]),
              pnw=(N5 / "t2" / "pnw_ext", N5T2.CASES["pnw"]["ext"][1]))
SET_LABELS = dict(orig="original set", final="final set")
ROLE_SORT = ["parent", "intermediate", "competitor", "child"]


def complete(case, d):
    """Final-set run finished: every part has results_jobs_final.csv (PNW: hyperparam_search_summary.csv)."""
    parts = N5T2.run_dirs(d)
    nm = "hyperparam_search_summary.csv" if case == "pnw" else "results_jobs_final.csv"
    return bool(parts) and all((p / nm).exists() for p in parts)


def win_label(w):
    return "1–10 d\n(+ step 0)" if w == "w0-10" else f"{w[1:].replace('-', '–')} d"


def to_md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(floatfmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def merge_write(name, X, cases, out):
    """Replace the rows of `cases` in out/<name>.csv by X (other cases kept); also write <name>.md."""
    f = out / f"{name}.csv"
    if f.exists():
        old = pd.read_csv(f)
        if "case" in old:
            old = old[~old["case"].isin(cases)]
        X = pd.concat([old, X], ignore_index=True) if len(old) else X
    X.to_csv(f, index=False)
    (out / f"{name}.md").write_text(f"# {name}\n\n" + to_md(X))
    return X


def track_rows(tabs, Ds, paired):
    """Debby distance to the IBTrACS track (km): each set on its own retained configs and paired (final - original)."""
    col = "mean_of_member_mean_track_dist_km"
    rows = []
    o = tabs["orig"].set_index("trial_id")[col]; e = tabs["final"].set_index("trial_id")[col]
    for set_, s in (("orig", o), ("final", e)):
        own = sorted(Ds[set_])
        m, lo, hi = boot_ci(s.reindex(own).to_numpy())
        rows.append(dict(case="debby", set=set_, cfgset="own", n_cfg=len(own), mean_km=m, lo=lo, hi=hi))
        m, lo, hi = boot_ci(s.reindex(paired).to_numpy())
        rows.append(dict(case="debby", set=set_, cfgset="paired", n_cfg=len(paired), mean_km=m, lo=lo, hi=hi))
    d = (e.reindex(paired) - o.reindex(paired)).dropna().to_numpy()
    m, lo, hi = boot_ci(d)
    rows.append(dict(case="debby", set="final-orig", cfgset="paired", n_cfg=len(d), mean_km=m, lo=lo, hi=hi,
                     frac_final_farther=float(np.mean(d > 0)) if len(d) else np.nan))
    return rows


def headline(case, W, P):
    """One line: parent share [CI] and rank-1 % per window, original vs final (paired configs)."""
    w = W[(W["case"] == case) & (W["cfgset"] == "paired")]
    parts = []
    for v in PARENTS[case]:
        for wl in [f"w{a}-{b}" for a, b in WINDOWS[case]] + (["w0-10"] if case == "pnw" else []):
            s = []
            for set_ in ("orig", "final"):
                r = w[(w["window"] == wl) & (w["var"] == v) & (w["set"] == set_)]
                if len(r):
                    r = r.iloc[0]
                    s.append(f"{set_} {r.share:.3f} [{r.lo:.3f}, {r.hi:.3f}] rk1 {100 * r.frac_rank1:.0f}%")
            parts.append(f"{v} {win_label(wl).replace(chr(10), ' ')}: " + "; ".join(s))
    n = w.groupby("set")["n_cfg"].max().to_dict()
    txt = f"{case}: paired n={n.get('final', 0)} | " + " | ".join(parts)
    if P is not None and len(P):
        p = P[(P["cfgset"] == "paired")].set_index("set")
        txt += (f" | track km orig {p.loc['orig', 'mean_km']:.0f}, final {p.loc['final', 'mean_km']:.0f}, "
                f"final-orig {p.loc['final-orig', 'mean_km']:.0f} [{p.loc['final-orig', 'lo']:.0f}, {p.loc['final-orig', 'hi']:.0f}]")
    return txt


# ------------------------------------------------------------------ S12
def fig_s12(case, W, cfg, root):
    """S12 for one case: (a) mean share by window, original (hatched) vs final (solid), 95 % CI; (b) rank-1 fraction."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    S.set_style()
    wins = [f"w{a}-{b}" for a, b in WINDOWS[case] if (a, b) != (11, 30)] + (["w0-10"] if case == "pnw" else [])
    fv = sorted(cfg["final"][1], key=lambda v: (ROLE_SORT.index(cfg["roles"][v]), cfg["final"][1].index(v)))
    w = W[(W["case"] == case) & (W["cfgset"] == "paired") & W["window"].isin(wins)]
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE, 0.36 * S.DOUBLE), gridspec_kw=dict(wspace=0.22))
    nv = len(fv); bw = 0.84 / nv
    for ax, col in zip(axs, ("share", "frac_rank1")):
        for k, wl in enumerate(wins):
            for i, v in enumerate(fv):
                for set_, hatch, off in (("orig", "////", -bw / 4), ("final", None, bw / 4)):
                    r = w[(w["window"] == wl) & (w["var"] == v) & (w["set"] == set_)]
                    if not len(r):
                        continue
                    r = r.iloc[0]; x = k + (i - nv / 2 + 0.5) * bw + off
                    ax.bar(x, r[col], bw / 2, color=S.col(v), edgecolor="white" if hatch else S.col(v), hatch=hatch, lw=0.3)
                    if col == "share":
                        ax.errorbar(x, r.share, yerr=[[r.share - r.lo], [r.hi - r.share]], color="k", lw=0.5, capsize=0)
        ax.set_xticks(range(len(wins))); ax.set_xticklabels([win_label(wl) for wl in wins])
        for k in range(1, len(wins)):
            ax.axvline(k - 0.5, color="0.85", lw=0.5)
    axs[0].set_ylabel("Mean share of trajectory density\n(95 % CI over configs)")
    axs[1].set_ylabel("Fraction of configs with rank 1"); axs[1].set_ylim(0, 1.0)
    axs[0].set_ylim(0, max(0.05, float(w["hi"].max()) * 1.12) if len(w) else 1)
    n = int(w[w["set"] == "final"]["n_cfg"].max()) if len(w) else 0
    hv = [Patch(facecolor=S.col(v), label=f"{S.lab(v)} ({cfg['roles'][v]})") for v in fv]
    hs = [Patch(facecolor="0.6", hatch="////", edgecolor="white", label=SET_LABELS["orig"]),
          Patch(facecolor="0.6", label=SET_LABELS["final"])]
    fig.legend(handles=hv + hs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(5, len(hv) + 2), frameon=False)
    step0 = "; step 0 excluded except in '1–10 d (+ step 0)'" if case == "pnw" else ""
    fig.suptitle(f"{cfg['title']}: original vs final variable set ({n} configs retained in both{step0})",
                 x=0.02, ha="left", fontsize=7.5, fontweight="bold", y=1.02)
    S.panel_labels(axs, x=-0.12, y=1.02)
    d = root / "S12_original_vs_final"; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"s12_{case}_original_vs_final.pdf", bbox_inches="tight")
    fig.savefig(d / f"s12_{case}_original_vs_final.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return d / f"s12_{case}_original_vs_final.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--test-n5", default=None, help="develop on the night-5 extended sets; outputs to this dir")
    ap.add_argument("--done-note", action="store_true", help="write night6/agents/C_done_<case>.txt")
    a = ap.parse_args()
    out, root = OUT, FIG6
    if a.test_n5:
        out = root = Path(a.test_n5)
        for c, v in N5_EXT.items():
            CASES[c]["final"] = v
    out.mkdir(parents=True, exist_ok=True)
    Wl, Ll, Kl, counts, path, done = [], [], [], [], [], []
    for case in a.cases:
        cfg = CASES[case]
        if not (a.partial or complete(case, cfg["final"][0])):
            print("final-set run not complete:", case, cfg["final"][0], flush=True); continue
        tabs, Ds = {}, {}
        for set_ in ("orig", "final"):
            rel, vars_ = cfg[set_]
            t = N5T2.load_table(case, rel, a.partial)
            if t is None:
                print("no output yet", case, set_, flush=True); continue
            tabs[set_] = t
            Ds[set_] = N5T2.case_densities(case, rel, vars_, t, a.nproc)
            counts.append(dict(case=case, set=set_, run=str(rel), n_done=len(t),
                               n_harmonized=int(retention.harmonized_case(case, t).sum()), n_with_density=len(Ds[set_])))
        if "final" not in Ds:
            continue
        paired = sorted(set.intersection(*[set(D) for D in Ds.values()])) if len(Ds) == 2 else []
        for r in counts:
            if r["case"] == case:
                r["n_paired"] = len(paired)
        for set_, Dd in Ds.items():
            for tag, cf in (("own", sorted(Dd)), ("paired", paired)):
                if cf:
                    w, l, k = N5T2.stats_rows(case, set_, cfg[set_][1], Dd, cf, cfg["roles"], tag)
                    Wl += w; Ll += l; Kl += k
        if case == "debby" and len(tabs) == 2:
            path += track_rows(tabs, Ds, paired)
        done.append(case)
    if not done:
        return
    W = merge_write("T2_window_shares", pd.DataFrame(Wl), done, out)
    L = merge_write("T2_lead_shares", pd.DataFrame(Ll), done, out)
    merge_write("T2_role_shares", pd.DataFrame(Kl), done, out)
    merge_write("T2_counts", pd.DataFrame(counts), done, out)
    P = merge_write("T2_debby_path", pd.DataFrame(path), ["debby"] if "debby" in done else [], out) if path else None
    R = W[["case", "set", "cfgset", "window", "var", "role", "n_cfg", "frac_rank1"]]
    merge_write("T2_rank1", R[R["case"].isin(done)], done, out)
    rl = W[(W["cfgset"] == "own")][["case", "set", "window", "role", "var", "n_cfg", "share", "lo", "hi", "frac_rank1"]]
    merge_write("T2_roles_table", rl[rl["case"].isin(done)].sort_values(
        ["case", "set", "window", "role", "share"], ascending=[True, True, True, True, False]), done, out)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(pd.DataFrame(counts).to_string())
        w = W[W["case"].isin(done) & (W["cfgset"] == "paired")]
        print(w.pivot_table(index=["case", "window", "var"], columns="set", values=["share", "frac_rank1"]).round(3).to_string())
        if P is not None:
            print(P.round(1).to_string())
    for case in done:
        h = headline(case, W, P if case == "debby" else None)
        print(h, flush=True)
        if a.done_note and not a.test_n5:
            (N6 / "agents" / f"C_done_{case}.txt").write_text(h + "\n")
        if ((W["case"] == case) & (W["cfgset"] == "paired")).any():
            print("fig", fig_s12(case, W, CASES[case], root), flush=True)


if __name__ == "__main__":
    main()
