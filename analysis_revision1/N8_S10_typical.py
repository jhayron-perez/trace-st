"""Night-8 S10 / C4 (synthetic case 2): typical-field targets (median statistics of V1-V3 = A, B, C of the case at each alpha),
validation of the generator (mean over 20 independent realizations, member index 200..219, within 10 % on variance, rho, L_lat,
L_lon, as night 6/7) and the production realizations used by S10 (member index 100 + k, k = 0..5, sha256 of float32 arrays).
-> night8/s10/typical_validation.{csv,md}, typical_seeds.csv"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_fraction_full_length import load_script  # noqa: E402

spec = importlib.util.spec_from_file_location("mls", ROOT / "data_processing" / "surrogates" / "make_linkless_synthetic.py")
ll = importlib.util.module_from_spec(spec); spec.loader.exec_module(ll)
m = load_script("3_SyntheticRuns_Multivariate.py")
OUT = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night8/s10"); OUT.mkdir(parents=True, exist_ok=True)


def stats(x):
    s = ll.field_stats(x)
    return dict(var=float((s["sd"].astype(np.float64) ** 2).mean()), rho=s["rho"], L_lat=s["L_lat"], L_lon=s["L_lon"])


rows, seeds = [], []
for alpha in (0.2, 0.6):
    c = m.build_multivariate_case(alpha_mix=alpha, seed=m.GLOBAL_SEED)
    arrs = [c["full_data_run"].sel(var=v).values for v in ("A", "B", "C")]
    tg = ll.typical_targets(arrs)
    src = [stats(a) for a in arrs]
    tgt = dict(var=float((tg["sd"].astype(np.float64) ** 2).mean()), rho=tg["rho"], L_lat=tg["L_lat"], L_lon=tg["L_lon"])
    val = [stats(ll.typical_field(tg, ll.typical_seed(4, 200 + k), clip=False)) for k in range(20)]
    mean = {k: float(np.mean([v[k] for v in val])) for k in tgt}
    err = {k: mean[k] / tgt[k] - 1 for k in tgt}
    rows.append(dict(alpha=alpha, n_val=20, **{f"tgt_{k}": tgt[k] for k in tgt}, **{f"val_{k}": mean[k] for k in tgt},
                     **{f"err_{k}": err[k] for k in tgt}, passed=bool(all(abs(e) <= 0.10 for e in err.values())),
                     **{f"V{i + 1}_{k}": src[i][k] for i in range(3) for k in tgt}))
    for k in range(6):
        sd = ll.typical_seed(4, 100 + k)
        x = ll.typical_field(tg, sd, clip=False)
        seeds.append(dict(alpha=alpha, k=k, member=100 + k, seed=sd, sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                          **{f"info_{kk}": vv for kk, vv in stats(x).items()}))
    # the production path (extra_mix) gives the same field
    cm = m.build_multivariate_case(alpha_mix=alpha, seed=m.GLOBAL_SEED, extra_mix=["typical", "response", "nonlinear", "typical"])
    assert np.array_equal(cm["full_data_run"].sel(var="Z1").values, ll.typical_field(tg, ll.typical_seed(4, 100), clip=False))
    assert np.array_equal(cm["full_data_run"].sel(var="Z2").values, ll.typical_field(tg, ll.typical_seed(4, 101), clip=False))
V, S = pd.DataFrame(rows), pd.DataFrame(seeds)
V.to_csv(OUT / "typical_validation.csv", index=False); S.to_csv(OUT / "typical_seeds.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.4f}".format):
    txt = ["# S10 typical field (synthetic case 2): targets = median over V1-V3 (A, B, C); var = mean over cells of per-cell variance",
           "seed = 20261123 + 1e6 * 4 + member; member 100 + k = k-th added typical field (production), 200..219 = validation",
           "validation: mean of 20 independent realizations vs target (pass: |rel. err| <= 10 % on var, rho, L_lat, L_lon)", "```",
           V[["alpha", "tgt_var", "val_var", "err_var", "tgt_rho", "val_rho", "err_rho", "tgt_L_lat", "val_L_lat", "err_L_lat",
              "tgt_L_lon", "val_L_lon", "err_L_lon", "passed"]].to_string(index=False), "```", "production realizations (info):", "```",
           S.to_string(index=False), "```"]
(OUT / "typical_validation.md").write_text("\n".join(txt)); print("\n".join(txt))
