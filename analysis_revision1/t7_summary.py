"""T7 summary per p: medians over the 50 configs (alpha_mix = 0.6, M = 30)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
R, out = sys.argv[1], sys.argv[2]
rows = []
cols = ["t7_share_density_v1v3", "t7_share_density_target", "t7_share_density_noise", "t7_share_density_decoy",
        "t7_share_density_moving", "t7_alpha_abs_err_density_13", "t7_alpha_abs_err_end_13", "t7_track_error_mean",
        "t7_completion_rate", "t7_sec_per_traj", "t7_samples_per_feature", "t7_n_samples", "t7_n_features"]
for p in (3, 5, 8, 12, 16):
    d = pd.read_csv(f"{R}/T7_distractors/p{p:02d}/results_alphas_final.csv")
    if p == 3:  # no extras -> t7_ columns absent; derive from the original 3-variable summary
        d["t7_share_density_v1v3"] = d["density_frac_var1"] + d["density_frac_var3"]
        d["t7_share_density_target"] = d["density_frac_var2"]
        d["t7_alpha_abs_err_density_13"] = (d["density_frac_var1"] / (d["density_frac_var1"] + d["density_frac_var3"]) - 0.6).abs()
        d["t7_alpha_abs_err_end_13"] = (d["end_frac_var1"] / (d["end_frac_var1"] + d["end_frac_var3"]) - 0.6).abs()
    r = {"p": p, "n_configs": len(d), "abs_err_alpha_density_origdef": float((d["estimated_alpha_density"] - 0.6).abs().median())}
    for c in cols:
        if c in d:
            r[c.replace("t7_", "")] = float(d[c].median())
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(out + ".csv", index=False); print(df.round(3).T.to_string())
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].plot(df.p, df.share_density_v1v3, "o-", color="#6a3d9a", label="V1+V3 (true parents)")
for c, col, lab in (("share_density_decoy", "#e6550d", "decoys"), ("share_density_noise", "#636363", "noise"), ("share_density_moving", "#3182bd", "moving blobs"), ("share_density_target", "#bdbdbd", "target V2")):
    if c in df: ax[0].plot(df.p, df[c], "o-", color=col, label=lab)
ax[0].set_title("Median share of trajectory density"); ax[0].legend(fontsize=7); ax[0].set_xlabel("p (variables)")
ax[1].plot(df.p, df.alpha_abs_err_density_13, "o-", color="#6a3d9a", label="density"); ax[1].plot(df.p, df.alpha_abs_err_end_13, "s--", color="#e6550d", label="endpoint")
ax[1].set_title("|recovered - 0.6| (V1 vs V3)"); ax[1].legend(fontsize=7); ax[1].set_xlabel("p")
ax[2].plot(df.p, df.samples_per_feature, "o-", color="#636363"); ax[2].set_yscale("log"); ax[2].set_title("N_samples / N_features (median)"); ax[2].set_xlabel("p")
fig.tight_layout(); fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf")
