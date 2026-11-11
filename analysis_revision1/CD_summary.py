"""Night-2 C4 / D5: decoy and moving-blob tests, one table + figures.

The runs and what changed (driver mode, parent noise, engine, placement, stability) are listed in
RUNS. Every run has 50 (EN) or 30 (PCMCI) configs at alpha 0.6. The shares are the t7_* medians and
means over configs. alpha error = |density share V1/(V1+V3) - 0.6| (t7_alpha_abs_err_density_13).
Also: the moving share restricted to steps 1-8 (the target's causal window), recomputed from the
bundles.

python CD_summary.py   (PBS only)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402

R1 = C.R1
S, CD, DM = R1 / "T7_decoy_sweep", R1 / "C_decoy", R1 / "D_moving"
# name, group, dir, attributes
RUNS = [
    # C: decoys (p=4: A, B, C + one decoy of A, beta 0.8 phi 0.8 unless noted)
    ("clean, EN", "C decoy p4", S / "b08_phi08", dict(driver="clean", sigma=0.10, engine="EN")),
    ("observed, EN", "C decoy p4", CD / "obs_p4_b08_phi08", dict(driver="observed", sigma=0.10, engine="EN")),
    ("clean β0.4, EN", "C decoy p4 β0.4", S / "b04_phi08", dict(driver="clean", engine="EN")),
    ("observed β0.4, EN", "C decoy p4 β0.4", CD / "obs_p4_b04_phi08", dict(driver="observed", engine="EN")),
    ("clean σ0.05", "C2 parent noise", CD / "clean_p4_b08_phi08_psig005", dict(driver="clean", sigma=0.05)),
    ("clean σ0.10", "C2 parent noise", S / "b08_phi08", dict(driver="clean", sigma=0.10)),
    ("clean σ0.20", "C2 parent noise", CD / "clean_p4_b08_phi08_psig020", dict(driver="clean", sigma=0.20)),
    ("clean, PCMCI", "C3 engine", CD / "pcmci_clean_p4_b08_phi08", dict(driver="clean", engine="PCMCI")),
    ("observed, PCMCI", "C3 engine", CD / "pcmci_obs_p4_b08_phi08", dict(driver="observed", engine="PCMCI")),
    ("clean, EN ", "C3 engine", S / "b08_phi08", dict(driver="clean", engine="EN")),
    ("observed, EN ", "C3 engine", CD / "obs_p4_b08_phi08", dict(driver="observed", engine="EN")),
    ("clean p8 mixed", "C mixed sets", R1 / "T7_distractors" / "p08", dict(driver="clean")),
    ("observed p8 mixed", "C mixed sets", CD / "obs_p08_mixed", dict(driver="observed")),
    ("clean p16 mixed", "C mixed sets", R1 / "T7_distractors" / "p16", dict(driver="clean")),
    ("observed p16 mixed", "C mixed sets", CD / "obs_p16_mixed", dict(driver="observed")),
    ("stability, clean decoy p4 (M10)", "D4 stability", CD / "stability_clean_p4_b08_phi08_M10", dict(stability=True)),
    # D: moving blobs, no decoys (p8: 3 noise + 2 moving; p16: 7 + 6)
    ("event, clean p8", "D2 placement", S / "nodecoy_p08", dict(placement="event", driver="clean", p=8)),
    ("random, clean p8", "D2 placement", DM / "random_p08_clean", dict(placement="random", driver="clean", p=8)),
    ("event, observed p8", "D2 placement", DM / "event_p08_obs", dict(placement="event", driver="observed", p=8)),
    ("random, observed p8", "D2 placement", DM / "random_p08_obs", dict(placement="random", driver="observed", p=8)),
    ("event, clean p16", "D2 placement", S / "nodecoy_p16", dict(placement="event", driver="clean", p=16)),
    ("random, clean p16", "D2 placement", DM / "random_p16_clean", dict(placement="random", driver="clean", p=16)),
    ("event, observed p16", "D2 placement", DM / "event_p16_obs", dict(placement="event", driver="observed", p=16)),
    ("random, observed p16", "D2 placement", DM / "random_p16_obs", dict(placement="random", driver="observed", p=16)),
    ("event p8 (reference)", "D3 worst case", S / "nodecoy_p08", dict()),
    ("offset 50", "D3 worst case", DM / "event_p08_offset50", dict()),
    ("closest approach t0−15", "D3 worst case", DM / "event_p08_tclag15", dict()),
    ("W 10–12 d", "D3 worst case", DM / "event_p08_W10to12", dict()),
    ("stability, event p8 (M10)", "D4 stability", DM / "stability_event_p08_M10", dict(stability=True)),
    ("no stability, event p8 (M30)", "D4 stability", S / "nodecoy_p08", dict(stability=False)),
    ("no stability, clean decoy p4 (M30)", "D4 stability", S / "b08_phi08", dict(stability=False)),
]
COLS = ["t7_share_density_v1v3", "t7_share_density_decoy", "t7_share_density_moving", "t7_share_density_noise",
        "t7_share_density_target", "t7_alpha_abs_err_density_13", "t7_track_error_mean", "t7_completion_rate"]


def load(d):
    for f in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
        if (d / f).exists():
            x = pd.read_csv(d / f)
            if "alpha_mix" in x:
                x = x[(x["alpha_mix"] - 0.6).abs() < 1e-9]
            return x, f
    return None, None


def moving_share_window(d, n_noise_decoy, n_moving, s1=1, s2=8):
    """Mean over configs of the share of member-steps on moving blobs within steps s1..s2."""
    if n_moving == 0 or not (d / "bundles").exists():
        return np.nan
    ids = set(range(3 + n_noise_decoy + 1, 3 + n_noise_decoy + n_moving + 1))
    v = []
    for f in sorted((d / "bundles").glob("*alpha_0p6.npz")):
        b = np.load(f, allow_pickle=True)
        P, L, ok = b["parents"], b["lengths"], b["ok"]
        tot = hit = 0
        for i in range(len(L)):
            n = int(L[i])
            if int(ok[i]) != 1:
                continue
            par = np.asarray(P[i, s1:min(n, s2 + 1)], int)
            tot += par.size; hit += int(np.isin(par, list(ids)).sum())
        if tot:
            v.append(hit / tot)
    return float(np.mean(v)) if v else np.nan


def main():
    C.set_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = C.outdir("CD_summary")
    rows = []
    for name, grp, d, att in RUNS:
        x, f = load(d)
        r = dict(name=name.strip(), group=grp, dir=str(d), file=f, n_cfg=0 if x is None else len(x), **att)
        if x is not None and len(x):
            for c in COLS:
                if c in x:
                    r[c.replace("t7_", "") + "_median"] = float(x[c].median())
                    r[c.replace("t7_", "") + "_mean"] = float(x[c].mean())
            nn = int(x["t7_n_noise"].iloc[0] + x["t7_n_decoy"].iloc[0]) if "t7_n_noise" in x else 0
            nm = int(x["t7_n_moving"].iloc[0]) if "t7_n_moving" in x else 0
            r["moving_share_steps1to8_mean"] = moving_share_window(d, nn, nm)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "CD_summary.csv", index=False)
    (out / "CD_summary.md").write_text(C.to_markdown(T[[c for c in T.columns if c in (
        "group", "name", "n_cfg", "share_density_v1v3_median", "share_density_decoy_median",
        "share_density_moving_median", "moving_share_steps1to8_mean", "alpha_abs_err_density_13_median")]]))
    print(T[["group", "name", "n_cfg"] + [c for c in T.columns if c.endswith("_median")][:6]
            + ["moving_share_steps1to8_mean"]].round(3).to_string(), flush=True)

    def panel(grp, metrics, fname, title):
        g = T[(T["group"] == grp) & (T["n_cfg"] > 0)]
        if g.empty:
            return
        fig, ax = plt.subplots(figsize=(16, 7))
        x = np.arange(len(g)); w = 0.8 / len(metrics)
        for k, (col, lab, colr) in enumerate(metrics):
            if col in g:
                ax.bar(x + (k - (len(metrics) - 1) / 2) * w, g[col], w, color=colr, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(g["name"], rotation=20, ha="right")
        ax.set_ylabel("Median share of trajectory density / |α error|"); ax.set_ylim(0, 1)
        ax.legend(frameon=False, ncol=len(metrics)); ax.set_title(title)
        fig.tight_layout(); C.savefig(fig, "CD_summary", fname); plt.close(fig)

    M_DEC = [("share_density_v1v3_median", "true parents A+C", "#0072B2"),
             ("share_density_decoy_median", "decoy", "#D55E00"),
             ("alpha_abs_err_density_13_median", "|α error|", "#999999")]
    M_MOV = [("share_density_v1v3_median", "true parents A+C", "#0072B2"),
             ("share_density_moving_median", "moving blobs (all steps)", "#E69F00"),
             ("moving_share_steps1to8_mean", "moving blobs (steps 1–8)", "#CC79A7"),
             ("alpha_abs_err_density_13_median", "|α error|", "#999999")]
    panel("C decoy p4", M_DEC, "C1_decoy_driver", "Decoy (β 0.8, φ 0.8): clean vs observed driver")
    panel("C decoy p4 β0.4", M_DEC, "C1_decoy_driver_b04", "Decoy (β 0.4, φ 0.8): clean vs observed driver")
    panel("C2 parent noise", M_DEC, "C2_parent_noise", "Decoy share vs parent noise (clean driver)")
    panel("C3 engine", M_DEC, "C3_engine", "Decoy share by engine and driver (p = 4)")
    panel("C mixed sets", M_DEC + [("share_density_moving_median", "moving", "#E69F00")], "C1_mixed_sets",
          "Mixed distractor sets: clean vs observed driver")
    panel("D2 placement", M_MOV, "D2_placement", "Moving blobs: event-aimed vs random tracks")
    panel("D3 worst case", M_MOV, "D3_worst_case", "Moving blobs aimed at the event: diagnostics (p = 8)")
    panel("D4 stability", M_DEC + [("share_density_moving_median", "moving", "#E69F00")], "D4_stability",
          "Stability selection (Meinshausen & Bühlmann 2010)")
    # T2-EN observed driver: alpha recovery
    x, _ = load(CD / "obs_T2en")
    xa = pd.read_csv(CD / "obs_T2en" / "results_alphas_final.csv") if (CD / "obs_T2en" / "results_alphas_final.csv").exists() else None
    ref = pd.read_csv(R1 / "T2r_multivariate_en" / "results_alphas_final.csv")
    ref = ref[ref["trial_id"] < 50]
    if xa is not None:
        fig, ax = plt.subplots(figsize=(10, 7))
        res = []
        for lab, d, colr in (("clean driver (T2r, first 50 configs)", ref, "#0072B2"), ("observed driver", xa, "#D55E00")):
            g = d.groupby("alpha_mix")["estimated_alpha_density"]
            ax.errorbar(g.median().index + (0.01 if "observed" in lab else -0.01), g.median(),
                        yerr=[g.median() - g.quantile(.1), g.quantile(.9) - g.median()], fmt="o-", color=colr,
                        lw=2, capsize=4, label=lab)
            r = np.corrcoef(d["alpha_mix"], d["estimated_alpha_density"])[0, 1]
            s = np.polyfit(d["alpha_mix"], d["estimated_alpha_density"], 1)[0]
            res.append(dict(run=lab, pearson_r=r, slope=s, n=len(d)))
        ax.plot([0, 1], [0, 1], ":", color="#999999")
        ax.set_xlabel("True α (mixing of A vs C)"); ax.set_ylabel("Recovered α (density; median, 10–90 %)")
        ax.legend(frameon=False); ax.set_title("T2 α recovery with the observed driver")
        fig.tight_layout(); C.savefig(fig, "CD_summary", "C1_T2en_alpha_recovery"); plt.close(fig)
        pd.DataFrame(res).to_csv(out / "C1_T2en_alpha_recovery.csv", index=False)
        print(pd.DataFrame(res).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
