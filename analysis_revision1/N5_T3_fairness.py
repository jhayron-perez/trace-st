"""Night-5 T3, pre-declared rule 4: fairness of the synthetic decoys, checked before any trajectory run.

A decoy D must be no better a one-step predictor of B than the weaker true parent:
R2(B[t] ~ D[t-1]) <= min(R2(B[t] ~ A[t-1]), R2(B[t] ~ C[t-1])) at alpha 0.2 and 0.6.
R2 = squared Pearson correlation (single-predictor OLS with intercept), pooled over all grid cells and the
injection period t0 <= t < T (where B is driven by A and C); also reported over 1 <= t < T.
The case fields do not depend on the config (build_multivariate_case uses GLOBAL_SEED), so one check per alpha.
beta starts at 0.8 (phi 0.5, sigma 0.1); a failing type has beta halved (up to twice); the largest passing beta is used.

python N5_T3_fairness.py     (PBS only) -> revision1/night5/T3_fairness.csv, T3_fairness.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night5")
MIX16 = ["noise", "response", "moving", "delayed", "nonlinear", "noise", "response", "moving", "delayed",
         "nonlinear", "noise", "response", "moving"]


def load():
    spec = importlib.util.spec_from_file_location("syn", REPO / "Scripts_Paper" / "3_SyntheticRuns_Multivariate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def r2(y, x):
    y = y.ravel().astype(float); x = x.ravel().astype(float)
    if x.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def main():
    m = load()
    rows, chosen = [], {}
    for beta in (0.8, 0.4, 0.2):
        for a in (0.2, 0.6):
            c = m.build_multivariate_case(alpha_mix=a, extra_mix=MIX16, decoy_phi=0.5, decoy_beta=beta, decoy_sigma=0.1)
            fd = c["full_data_run"]
            t0, T = c["injection_start_idx"], c["T"]
            B = fd.sel(var="B").values
            for per, ts in (("injection", np.arange(t0, T)), ("all", np.arange(1, T))):
                ref = {v: r2(B[ts], fd.sel(var=v).values[ts - 1]) for v in ("A", "C")}
                for name, typ in c["extra_vars"]:
                    rr = r2(B[ts], fd.sel(var=name).values[ts - 1])
                    rows.append(dict(beta=beta, alpha=a, period=per, var=name, type=typ, r2=rr, r2_A=ref["A"], r2_C=ref["C"],
                                     r2_weaker=min(ref.values()), passes=rr <= min(ref.values())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "T3_fairness.csv", index=False)
    inj = df[df["period"] == "injection"]
    for typ in ("response", "delayed", "nonlinear"):
        ok = [b for b in (0.8, 0.4, 0.2) if inj[(inj["type"] == typ) & (inj["beta"] == b)]["passes"].all()]
        chosen[typ] = max(ok) if ok else None
    for typ in ("noise", "moving"):
        chosen[typ + "_passes_at_0.8"] = bool(inj[(inj["type"] == typ) & (inj["beta"] == 0.8)]["passes"].all())
    json.dump(chosen, open(OUT / "T3_fairness.json", "w"), indent=1)
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(inj[inj["beta"] == 0.8].round(4).to_string())
        print(inj.groupby(["beta", "type", "alpha"])[["r2", "r2_weaker"]].max().round(4).to_string())
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
