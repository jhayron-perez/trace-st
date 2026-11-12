"""Night-3 N0 task 3 (H1, link level), Debby: from the re-traced A2 vo_2004 links (N0_task3_debby_retrace.py).

Input: night3/N0/task3/debby_retrace_2004/trial_XXXXX_links.npz (member, step, child, parent, pos, coef; parents and
centers per member) + selected_configs.csv (ratio, quartile q). Variables (1-based, as in the PNW debug tables):
1 Tb, 2 vo (real), 3 precip, 4 VO_P2004 (twin). Step k = k hours back; windows 1-4 d (k 1..96), 5-15 d (97..360).
Per config, window, role (real vo / twin) and child_is_self: fraction of regressions with any non-zero coefficient on
the block, mean non-zero count per regression, |beta| median / p90 (pooled). Real vs twin also by ratio quartile.
python N0_task3_links_debby.py   (PBS only). Output: night3/N0/task3/task3_debby_links_{long,summary}.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
D = R1 / "night3/N0/task3/debby_retrace_2004"
OUT = R1 / "night3/N0/task3"
ROLE = dict(real=2, twin=4)


def wname(k):
    return "w1-4" if 1 <= k <= 96 else ("w5-15" if 97 <= k <= 360 else None)


def main():
    sel = pd.read_csv(D / "selected_configs.csv")
    rows, beta = [], []
    first = True
    for t in sel.trial_id.astype(int):
        f = D / f"trial_{t:05d}_links.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        if first:
            print("unique parent", np.unique(z["parent"])[:10], "unique child", np.unique(z["child"])[:10], flush=True)
            first = False
        L = pd.DataFrame(dict(member=z["member"], step=z["step"], child=z["child"], parent=z["parent"],
                              coef=np.abs(z["coef"].astype(float))))
        L = L[L.coef > 0]
        for mi, (par, ok) in enumerate(zip(z["parents"], z["ok"])):
            if not ok:
                continue
            par = np.asarray(par, int)
            Lm = L[L.member == mi]
            g = {k: d for k, d in Lm.groupby("step")}
            for k in range(1, len(par)):
                w = wname(k)
                if w is None:
                    continue
                d = g.get(k)
                for role, idx in ROLE.items():
                    nz = 0 if d is None else int((d.parent == idx).sum())
                    rows.append((t, w, role, int(par[k]) == idx, 1, int(nz > 0), nz))
                    if nz:
                        beta.append((t, w, role, int(par[k]) == idx, d.coef[d.parent == idx].to_numpy()))
    X = pd.DataFrame(rows, columns=["cfg", "window", "role", "child_is_self", "n_reg", "n_any", "sum_nz"])
    X = X.groupby(["cfg", "window", "role", "child_is_self"], as_index=False).sum()
    X = X.merge(sel.rename(columns=dict(trial_id="cfg"))[["cfg", "ratio", "q"]], on="cfg")
    X.to_csv(OUT / "task3_debby_links_long.csv.gz", index=False)
    B = {}
    for t, w, role, slf, v in beta:
        B.setdefault((w, role, slf), []).append(v)
        B.setdefault((w, role, slf, int(sel.set_index("trial_id").q[t])), []).append(v)
    out = []
    for binned in (False, True):
        keys = ["window", "role", "child_is_self"] + (["q"] if binned else [])
        for k, g in X.groupby(keys):
            kk = dict(zip(keys, k))
            pc = g.groupby("cfg").apply(lambda x: pd.Series(dict(frac_any=x.n_any.sum() / max(x.n_reg.sum(), 1),
                                                                 nz_per_reg=x.sum_nz.sum() / max(x.n_reg.sum(), 1))))
            bk = (kk["window"], kk["role"], kk["child_is_self"]) + ((kk["q"],) if binned else ())
            b = np.concatenate(B[bk]) if bk in B else np.array([np.nan])
            out.append(dict(kk, q=kk.get("q", "all"), n_cfg=len(pc), frac_any_mean=pc.frac_any.mean(),
                            nz_per_reg=pc.nz_per_reg.mean(), absbeta_median=float(np.nanmedian(b)),
                            absbeta_p90=float(np.nanpercentile(b, 90))))
    S = pd.DataFrame(out)
    S.to_csv(OUT / "task3_debby_links_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
