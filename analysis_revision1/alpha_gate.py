"""Task-3 gate: recovered vs prescribed alpha_mix for T2-EN, revision-1 re-run vs original.

Recovered alpha per (trial, alpha_mix): estimated_alpha_density (= density_frac_var1) and
estimated_alpha_end (= end_frac_var1) from results_alphas_final.csv. Gate: Pearson r between the
prescribed grid and the median recovered over trials >= 0.9 for both metrics, and monotonic.

python alpha_gate.py NEW_DIR ORIG_DIR OUT_PREFIX
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

new_dir, orig_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]


def load(d):
    for f in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
        try:
            return pd.read_csv(f"{d}/{f}"), f
        except FileNotFoundError:
            pass
    raise FileNotFoundError(d)


res = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
colors = {"original (multivariate_v4)": "#7f7f7f", "revision-1 (fixed)": "#6a3d9a"}
for label, d in (("original (multivariate_v4)", orig_dir), ("revision-1 (fixed)", new_dir)):
    df, f = load(d)
    df = df[np.isfinite(df["alpha_mix"])]
    r = {"file": f"{d}/{f}", "n_trials": int(df["trial_id"].nunique())}
    for ax, m in zip(axes, ("estimated_alpha_density", "estimated_alpha_end")):
        g = df.groupby("alpha_mix")[m]
        med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
        a = med.index.values.astype(float)
        pr = float(np.corrcoef(a, med.values)[0, 1])
        mono = bool(np.all(np.diff(med.values) > 0))
        slope, icpt = np.polyfit(a, med.values, 1)
        # pooled trial-level r as extra information
        ok = df[[m, "alpha_mix"]].dropna()
        r_pool = float(np.corrcoef(ok["alpha_mix"], ok[m])[0, 1])
        r[m] = dict(median=dict(zip(map(float, a), map(float, med.values))), pearson_r_median=pr,
                    monotonic=mono, slope=float(slope), intercept=float(icpt), pearson_r_pooled=r_pool)
        ax.errorbar(a + (0.008 if "rev" in label else -0.008), med, yerr=[med - q25, q75 - med], fmt="o-",
                    color=colors[label], capsize=3, label=f"{label}: r={pr:.3f}")
        ax.set_title({"estimated_alpha_density": "Density metric", "estimated_alpha_end": "Endpoint metric"}[m])
    res[label] = r
for ax in axes:
    ax.plot([0, 1], [0, 1], ls=":", color="k", lw=0.8)
    ax.set_xlabel("Prescribed $\\alpha_{mix}$"); ax.set_xlim(0.1, 0.9); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("Recovered (median, IQR over trials)")
fig.tight_layout(); fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf")
new = res["revision-1 (fixed)"]
gate = all(new[m]["pearson_r_median"] >= 0.9 and new[m]["monotonic"] for m in ("estimated_alpha_density", "estimated_alpha_end"))
res["gate_pass"] = bool(gate)
json.dump(res, open(out + ".json", "w"), indent=1)
print(json.dumps(res, indent=1))
