"""Night-8 T1b PNW: full share table (unfiltered, R1, R2; 7 variables; windows) with 95 % bootstrap CIs over configs and rank-1
fractions, plus the shares-by-lead panel (unfiltered vs R2). Shares are per-config renormalized among the 7 real variables
(decoy excluded) and averaged over the configs of the set; CI = percentile bootstrap over configs (2,000 resamples, seed 0).
python N8_T1b_tables.py --nproc 32 -> night8/analysis/T1b_pnw_fulltable.{csv,md}, T1b_pnw_lead_shares.csv,
figures_v8/T1b/T1b_pnw_shares_by_lead.{png,pdf}"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N8_T1b as T  # noqa: E402
import N5_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CASE = "pnw"
REAL = T.FINAL[CASE]
NAME = dict(Z500="Z500", Z10="Z10", OLR="MTNLWRF", TCWV="TCWV", SLHF="MSLHF", SSHF="MSSHF", T2m="T2M")
WINS = [(1, 10), (11, 20), (21, 30), (11, 30)]
FIG = T.R1 / "figures_v8" / "T1b"


def boot(x, rng, n=2000):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()
    data, ids_ret, _ = T.load(CASE, a.nproc)
    ids = sorted(i for i in data if i in ids_ret)
    nv = len(REAL) + 1
    W = {w: [] for w in WINS}; LS = []; drop1, drop2 = [], []
    for i in ids:
        D = data[i][0].sum(0)
        ds, _ = T.shares(D, nv)
        tot = D[1:].sum(); real_all = D[1:, :nv - 1].sum(0) / tot
        f1, f2 = T.flags(ds, real_all, nv)
        drop1.append(f1); drop2.append(f2)
        for w in WINS:
            W[w].append(T.shares(D, nv, w)[1])
        Dr = D[1:, :nv - 1]                                    # (30, 7), lead 1..30
        s = Dr / np.where(Dr.sum(1, keepdims=True) > 0, Dr.sum(1, keepdims=True), np.nan)
        LS.append(s)
    drop = dict(unfiltered=np.zeros(len(ids), bool), R1=np.array(drop1), R2=np.array(drop2))
    LS = np.array(LS)
    rows, lrows = [], []
    for sname, d in drop.items():
        k = ~d
        for w in WINS:
            M = np.array(W[w])[k]
            ci = boot(M, np.random.default_rng(0))
            top = np.argmax(M, 1)
            r1 = np.array([(top == j).astype(float) for j in range(len(REAL))]).T     # (cfg, var)
            ci1 = boot(r1, np.random.default_rng(0))
            for j, v in enumerate(REAL):
                rows.append(dict(set=sname, n_cfg=int(k.sum()), window=f"{w[0]}-{w[1]}", var=v, share=M[:, j].mean(), lo=ci[0, j], hi=ci[1, j],
                                 rank1=r1[:, j].mean(), rank1_lo=ci1[0, j], rank1_hi=ci1[1, j]))
        if sname in ("unfiltered", "R2"):
            X = LS[k]
            for L in range(X.shape[1]):
                ok = np.isfinite(X[:, L, :]).all(1)
                if ok.sum() < 5:
                    continue
                m = X[ok, L, :]; ci = boot(m, np.random.default_rng(0))
                for j, v in enumerate(REAL):
                    lrows.append(dict(set=sname, lead=L + 1, var=v, share=m[:, j].mean(), lo=ci[0, j], hi=ci[1, j], n_cfg=int(ok.sum())))
    F, Ld = pd.DataFrame(rows), pd.DataFrame(lrows)
    F.to_csv(T.OUT / "T1b_pnw_fulltable.csv", index=False); Ld.to_csv(T.OUT / "T1b_pnw_lead_shares.csv", index=False)
    lines = ["# T1b PNW: real-variable shares (renormalized among the 7 real variables), mean over configs of the set, 95 % bootstrap CI over configs (2,000, seed 0); rank-1 = fraction of configs where the variable is first (CI bootstrap)",
             f"sets: unfiltered n={len(ids)}; not-rank-1 (R1) n={int((~drop['R1']).sum())}; equal-share (R2) n={int((~drop['R2']).sum())}\n"]
    for w in WINS:
        lines.append(f"\n## window {w[0]}-{w[1]} d\n\n| variable | unfiltered share [CI] | rank-1 | R1 share [CI] | rank-1 | R2 share [CI] | rank-1 |\n|---|---|---|---|---|---|---|")
        for v in REAL:
            c = []
            for sname in ("unfiltered", "R1", "R2"):
                r = F[(F.set == sname) & (F.window == f"{w[0]}-{w[1]}") & (F["var"] == v)].iloc[0]
                c += [f"{r.share:.3f} [{r.lo:.3f}, {r.hi:.3f}]", f"{r.rank1:.2f} [{r.rank1_lo:.2f}, {r.rank1_hi:.2f}]"]
            lines.append(f"| {v} | " + " | ".join(c) + " |")
    (T.OUT / "T1b_pnw_fulltable.md").write_text("\n".join(lines))
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a_, sname, ttl in zip(ax, ("unfiltered", "R2"), (f"Unfiltered ({len(ids)} configs)", f"Equal-share filter ({int((~drop['R2']).sum())} configs kept)")):
        for v in REAL:
            g = Ld[(Ld.set == sname) & (Ld["var"] == v)].sort_values("lead")
            a_.fill_between(g.lead, g.lo, g.hi, color=S.col(NAME[v]), alpha=0.22, lw=0)
            a_.plot(g.lead, g.share, "-", color=S.col(NAME[v]), lw=1.0, marker="o", ms=1.6, label=v)
        for (w0, w1), c in zip(WINS[:3], S.WINDOW_COLORS):
            a_.axvspan(w0 - 0.5, w1 + 0.5, color=c, alpha=0.07, lw=0)
        a_.set_xlim(0.5, 30.5); a_.set_ylim(0, None); a_.set_title(ttl, fontsize=9); a_.set_xlabel("Lead before the event (days)"); a_.grid(alpha=0.2, lw=0.4)
    ax[0].set_ylabel("Share of trajectory density (real variables)"); ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"T1b_pnw_shares_by_lead.{ext}", dpi=300)


if __name__ == "__main__":
    main()
