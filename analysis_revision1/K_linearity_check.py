"""Task K (R2 major 1): is the linear (lag-1, stencil) model adequate at the tracing steps?

For each case (Debby with rebuilt vorticity = T4r_debby_vorebuilt, Pinatubo = T5r_pinatubo,
PNW 2021 = T6r_pnw, retained configs; T2-EN synthetic = T2r_multivariate_en as the
linear calibration reference) we sample n tracing steps (seed 0): random retained config,
random successful member, random executed step k of its stored track (bundles / pickles:
centers, parents). At that step we rebuild exactly what the engine saw:

  window   = full_data.sel(time=slice(date_end_k - W + dt, date_end_k)), date_end_k = date_end - k*dt
  box      = trace_st.data_io.get_data_mcastle(center_k, box_size, data_list)
  design   = trace_st.castle_core.get_MV_reduced_space(box, dependencies_wrap=False, y_ascending=True, radius=r)
             (lag-1 pairs within each cell block, as _lasso_single_child_lag1)
  parents  = get_mcastle_box(...)  (engine-selected lag-1 parents of the child centre node)
  groups   = (parent variable, sign, DBSCAN direction cluster) from
             pick_direction_then_weighted_delta (montecarlo=False; clustering does not depend on rng)

Models for the child centre node y_t:
  (1) OLS on all selected parents x_{t-1} (+ intercept)
  (2) RESET: (1) + yhat^2 + yhat^3; F-test
  (3) (1) + additive cubic B-splines (patsy bs, df=SPL_DF) of the |beta|-weighted group means of the
      top-G groups (by sum |beta|) + all pairwise products of those group means.
      Nested in (1); G is reduced until n - rank >= 10.
Reported: adjusted R^2 gain of (2) and (3) over (1); BIC preference (dBIC = BIC1 - BICk > 2, > 10).
Centers in the Debby/Pinatubo/T2 bundles are float32; we use them as stored [box edges may
differ from the original run by one grid cell in rare cases, inferred].
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trace_st.trajectory import get_mcastle_box  # noqa: E402
from trace_st.data_io import get_data_mcastle  # noqa: E402
from trace_st.castle_core import get_MV_reduced_space  # noqa: E402
from trace_st.movement import pick_direction_then_weighted_delta  # noqa: E402
from trace_st import retention  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from scipy import stats  # noqa: E402
import patsy  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
VO_REBUILT = (str(R1 / "inputs_debby/vo500700_2006_rebuilt.nc"), "vo")
G_MAX = 6
SPL_DF = 4
MIN_RESID_DF = 10


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"),
                                                  ROOT / "Scripts_Paper" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# Step catalogues: list of dict(config, member, k, lat, lon, child, params, alpha_mix?)
# -----------------------------------------------------------------------------
def _members_from_npz(npz):
    z = np.load(npz)
    out = []
    for m in range(z["centers"].shape[0]):
        L = int(z["lengths"][m])
        if int(z["ok"][m]) != 1 or L < 2:
            continue
        out.append((m, z["centers"][m, :L].astype(float), z["parents"][m, :L].astype(int)))
    return out


def sample_steps(case, n, rng):
    rows = []
    if case == "t2":
        df = pd.read_csv(R1 / "T2r_multivariate_en/results_alphas_final.csv")
        df = df[df.n_ok > 0]
        pool_cfg = list(zip(df.trial_id.astype(int), df.alpha_mix.astype(float)))
    elif case == "debby":
        j = pd.read_csv(R1 / "T4r_debby_vorebuilt/results_jobs_final.csv")
        pool_cfg = [(int(t), None) for t in j.loc[retention.debby_retained(j), "trial_id"]]
    elif case == "pinatubo":
        j = pd.read_csv(R1 / "T5r_pinatubo/results_jobs_final.csv")
        pool_cfg = [(int(t), None) for t in j.loc[retention.pinatubo_retained(j), "trial_id"]]
    elif case == "pnw":
        ids = [int(x) for x in open(R1 / "T6r_pnw/retained_run_ids.txt").read().split()]
        pool_cfg = [(i, None) for i in ids]
    n_cfg_pool = len(pool_cfg)
    cache = {}
    while len(rows) < n:
        tid, am = pool_cfg[int(rng.integers(len(pool_cfg)))]
        if (tid, am) not in cache:
            if case == "t2":
                stem = f"trial_{tid:05d}_alpha_{str(am).replace('.', 'p')}"
                meta = json.load(open(R1 / f"T2r_multivariate_en/bundles/{stem}.json"))
                cache[(tid, am)] = (meta["params"], _members_from_npz(R1 / f"T2r_multivariate_en/bundles/{stem}.npz"))
            elif case == "debby":
                meta = json.load(open(R1 / f"T4r_debby_vorebuilt/bundles/debby_trial_{tid:05d}.json"))
                cache[(tid, am)] = (meta["params"], _members_from_npz(R1 / f"T4r_debby_vorebuilt/bundles/debby_trial_{tid:05d}.npz"))
            elif case == "pinatubo":
                b = sorted((R1 / "T5r_pinatubo/bundles").glob(f"*trial_{tid:05d}.json"))[0]
                meta = json.load(open(b))
                cache[(tid, am)] = (meta["params"], _members_from_npz(b.with_suffix(".npz")))
            elif case == "pnw":
                d = R1 / f"T6r_pnw/run_{tid:04d}"
                params = json.load(open(d / "params.json"))
                res = pickle.load(open(d / "ensemble_results.pkl", "rb"))
                mem = [(int(r["member_id"]), np.asarray(r["centers"], float), np.asarray(r["parents"], int))
                       for r in res if r["success"] and r["centers"] is not None and len(r["centers"]) >= 2]
                cache[(tid, am)] = (params, mem)
        params, mem = cache[(tid, am)]
        if len(mem) == 0:
            continue
        m, centers, parents = mem[int(rng.integers(len(mem)))]
        k = int(rng.integers(len(parents) - 1))  # executed steps: 0..L-2
        rows.append(dict(case=case, config=tid, alpha_mix=am, member=m, k=k, lat=float(centers[k, 0]),
                         lon=float(centers[k, 1]), child=int(parents[k]), params=params))
    return rows, n_cfg_pool


# -----------------------------------------------------------------------------
# Regression helpers
# -----------------------------------------------------------------------------
def fit(Xd, y):
    """OLS via lstsq; returns rss, rank (number of effective params incl. intercept), yhat."""
    beta, _, rank, _ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    rss = float(np.sum((y - yhat) ** 2))
    return rss, int(rank), yhat


def stats_from(rss, p, y):
    n = len(y)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / max(n - p, 1)
    bic = n * np.log(max(rss, 1e-300) / n) + p * np.log(n)
    return r2, adj, bic


def standardize_cols(A):
    A = np.asarray(A, float)
    s = A.std(axis=0)
    s[s < 1e-12] = 1.0
    return (A - A.mean(axis=0)) / s


def analyse_step(job):
    G = WG
    t0 = time.time()
    try:
        params = job["params"]
        full_data = G["full_data"][job.get("alpha_mix")]
        date_end0 = pd.to_datetime(G["date_end"])
        dt = pd.to_timedelta(params["timeres"])
        de = date_end0 - job["k"] * dt
        ds = de - pd.to_timedelta(params["timewindow"]) + dt
        win = full_data.sel(time=slice(ds, de))
        var_list = list(full_data["var"].values)
        try:
            data_list = [win.sel(var=v).drop_vars("var") for v in var_list]
        except Exception:
            data_list = [win.sel(var=v) for v in var_list]
        r = int(params["radius"])
        center = [job["lat"], job["lon"] % 360.0]
        child = int(job["child"])
        out = get_mcastle_box(center, float(params["box_size"]), data_list, radius=r, child_variable=child,
                              cd_method=str(params.get("cd_method", "pcmci")).lower(),
                              cd_kwargs=params.get("cd_kwargs"), strength_mode=str(params.get("strength_mode", "native")))
        st = (2 * r + 1) ** 2
        rows, gidx, betas = [], [], []
        for p in range(1, len(data_list) + 1):
            g, vals = out[f"p{p}_c{child}"]
            ii, jj, kk = np.where(g == "-->")
            for a, b, c in zip(ii, jj, kk):
                rows.append((f"p{p}_c{child}", float(vals[a, b, c]), (int(a), int(b), int(c))))
                gidx.append((p - 1) * st + int(a))
                betas.append(float(vals[a, b, c]))
        base = dict(case=job["case"], config=job["config"], alpha_mix=job["alpha_mix"], member=job["member"],
                    k=job["k"], lat=job["lat"], lon=job["lon"], child=child, n_parents=len(rows))
        if len(rows) == 0:
            return dict(base, status="no_parents", sec=time.time() - t0)
        box = get_data_mcastle(center, float(params["box_size"]), data_list)
        RS = get_MV_reduced_space(box, dependencies_wrap=False, y_ascending=True, radius=r).astype(float)
        T = int(box.shape[-1])
        R = RS.shape[0] // T
        idx = np.arange(R * T).reshape(R, T)
        src, dst = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        child_node = (child - 1) * st + st // 2
        y = RS[dst, child_node]
        X = RS[src][:, gidx]
        n = len(y)
        base.update(n_obs=n, n_cells=R, T=T)
        if np.std(y) < 1e-12:
            return dict(base, status="const_child", sec=time.time() - t0)
        y = (y - y.mean()) / y.std()
        # (3) groups -> splines + pairwise products
        cand = pd.DataFrame(rows, columns=["vars", "causal_value", "position"])
        cand["causal_value_abs"] = cand["causal_value"].abs()
        _, _, _, _, dfu = pick_direction_then_weighted_delta(
            cand, radius=r, spaceres=float(params["spaceres"]), eps=float(params.get("eps_dbscan", 0.5)),
            min_samples=int(params.get("min_samples_dbscan", 3)), montecarlo=False,
            prefer_sign=str(params.get("prefer_sign", "auto")), alpha=float(params.get("alpha", 2.0)),
            score_mode=str(params.get("score_mode", "sum_over_n")), gamma=float(params.get("gamma", 0.5)),
            winner_summary=str(params.get("winner_summary", "mean")))
        gl = dfu["cluster_global"].reindex(cand.index).fillna(-1).astype(int).values
        absb = np.abs(np.asarray(betas))
        groups = []
        for gid in np.unique(gl[gl >= 0]):
            msk = gl == gid
            w = absb[msk]
            gm = (X[:, msk] * w).sum(axis=1) / w.sum()
            groups.append((float(w.sum()), gm))
        groups.sort(key=lambda t: -t[0])
        n_groups = len(groups)
        # (G) group-level check, available for every step: OLS on top-G group means vs + splines/products
        resg = None
        for Gk in range(min(G_MAX, n_groups), 0, -1):
            gms = [standardize_cols(gm[:, None])[:, 0] for _, gm in groups[:Gk]]
            gms = [g for g in gms if np.std(g) > 1e-12]
            if not gms:
                continue
            Dg1 = np.column_stack([np.ones(n)] + [g[:, None] for g in gms])
            cols = [np.asarray(patsy.dmatrix(f"bs(x, df={SPL_DF}, degree=3) - 1", {"x": g})) for g in gms]
            cols += [(gms[a] * gms[b])[:, None] for a in range(len(gms)) for b in range(a + 1, len(gms))]
            Dg3 = np.column_stack([Dg1] + cols)
            rg1, pg1, _ = fit(Dg1, y)
            rg3, pg3, _ = fit(Dg3, y)
            if n - pg3 >= MIN_RESID_DF:
                _, ag1, bg1 = stats_from(rg1, pg1, y)
                _, ag3, bg3 = stats_from(rg3, pg3, y)
                resg = dict(groups_used_g=Gk, adjr2_glin=ag1, adjr2_gspl=ag3, d_adjr2_group=ag3 - ag1, dbic_group=bg1 - bg3)
                break
        base.update(resg or dict(groups_used_g=0))
        Xs = standardize_cols(X)
        D1 = np.column_stack([np.ones(n), Xs])
        rss1, p1, yhat = fit(D1, y)
        if n - p1 < MIN_RESID_DF:
            return dict(base, status="insufficient_df_ols", p_ols=p1, sec=time.time() - t0)
        r2_1, adj1, bic1 = stats_from(rss1, p1, y)

        # (2) RESET
        yh = (yhat - yhat.mean()) / (yhat.std() if yhat.std() > 1e-12 else 1.0)
        D2 = np.column_stack([D1, yh ** 2, yh ** 3])
        rss2, p2, _ = fit(D2, y)
        r2_2, adj2, bic2 = stats_from(rss2, p2, y)
        q = p2 - p1
        if q > 0 and n - p2 > 0:
            F = ((rss1 - rss2) / q) / (rss2 / (n - p2))
            pF = float(stats.f.sf(F, q, n - p2))
        else:
            F, pF = np.nan, np.nan

        res3 = None
        for Gk in range(min(G_MAX, n_groups), 0, -1):
            cols = []
            gms = [standardize_cols(gm[:, None])[:, 0] for _, gm in groups[:Gk]]
            for gm in gms:
                if np.std(gm) < 1e-12:
                    continue
                B = np.asarray(patsy.dmatrix(f"bs(x, df={SPL_DF}, degree=3) - 1", {"x": gm}))
                cols.append(B)
            for a in range(len(gms)):
                for b in range(a + 1, len(gms)):
                    cols.append((gms[a] * gms[b])[:, None])
            if not cols:
                continue
            D3 = np.column_stack([D1] + cols)
            rss3, p3, _ = fit(D3, y)
            if n - p3 >= MIN_RESID_DF:
                res3 = (Gk, rss3, p3)
                break
        if res3 is not None:
            Gk, rss3, p3 = res3
            r2_3, adj3, bic3 = stats_from(rss3, p3, y)
            q3 = p3 - p1
            F3 = ((rss1 - rss3) / q3) / (rss3 / (n - p3)) if q3 > 0 else np.nan
            pF3 = float(stats.f.sf(F3, q3, n - p3)) if q3 > 0 else np.nan
        else:
            Gk = 0; r2_3 = adj3 = bic3 = pF3 = np.nan; p3 = np.nan
        return dict(base, status="ok", p_ols=p1, r2_ols=r2_1, adjr2_ols=adj1, bic_ols=bic1,
                    p_reset=p2, r2_reset=r2_2, adjr2_reset=adj2, bic_reset=bic2, reset_F=F, reset_p=pF,
                    n_groups=n_groups, groups_used=Gk, p_spline=p3, r2_spline=r2_3, adjr2_spline=adj3,
                    bic_spline=bic3, spline_F_p=pF3,
                    d_adjr2_reset=adj2 - adj1, d_adjr2_spline=adj3 - adj1,
                    dbic_reset=bic1 - bic2, dbic_spline=bic1 - bic3, sec=time.time() - t0)
    except Exception as e:
        return dict(case=job["case"], config=job["config"], k=job["k"], status="error",
                    error=repr(e) + traceback.format_exc()[-800:], sec=time.time() - t0)


WG = {}


def load_case(case):
    if case == "debby":
        mod = load_script("5_DebbyRuns.py")
        c = mod.build_case_debby(vo_input=VO_REBUILT)
        return {None: c["full_data"]}, c["date_end"]
    if case == "pinatubo":
        mod = load_script("6_PinatuboRuns.py")
        c = mod.build_case_pinatubo()
        return {None: c["full_data"]}, c["date_end"]
    if case == "pnw":
        mod = load_script("8_HW_PNW21_Runs_v2.py")
        c = mod.build_case_pnw()
        return {None: c["full_data"]}, c["date_end"]
    if case == "t2":
        mod = load_script("3_SyntheticRuns_Multivariate.py")
        fd = {}
        for a in mod.ALPHA_MIX_GRID:
            cs = mod.build_multivariate_case(alpha_mix=float(a), T=40, N_inject=8, seed=mod.GLOBAL_SEED)
            fd[float(a)] = cs["full_data_run"]
        return fd, cs["date_end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--cases", nargs="+", default=["t2", "debby", "pinatubo", "pnw"])
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        for case in args.cases:
            t0 = time.time()
            rng = np.random.default_rng(0)
            jobs, n_pool = sample_steps(case, args.n_steps, rng)
            print(f"[{case}] sampled {len(jobs)} steps from {n_pool} retained configs ({time.time()-t0:.0f}s)", flush=True)
            fd, de = load_case(case)
            if case != "t2":
                fd = {None: fd[None].load()}
            WG["full_data"] = fd
            WG["date_end"] = de
            print(f"[{case}] data loaded ({time.time()-t0:.0f}s)", flush=True)
            with mp.get_context("fork").Pool(args.nproc) as pool:
                res = pool.map(analyse_step, jobs, chunksize=1)
            df = pd.DataFrame(res)
            df.to_csv(outdir / f"K_steps_{case}.csv", index=False)
            print(f"[{case}] done in {time.time()-t0:.0f}s; status: {df.status.value_counts().to_dict()}", flush=True)
            if (df.status == "error").any():
                print(df.loc[df.status == "error", "error"].iloc[0], flush=True)
    summarize_and_plot(outdir, Path(args.figdir), args.cases)


def summarize_and_plot(outdir, figdir, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    dfs = [pd.read_csv(outdir / f"K_steps_{c}.csv") for c in cases if (outdir / f"K_steps_{c}.csv").exists()]
    df = pd.concat(dfs, ignore_index=True)
    names = {"t2": "T2-EN (synthetic, linear)", "debby": "Debby (rebuilt vo)", "pinatubo": "Pinatubo", "pnw": "PNW 2021"}
    rows = []
    for c in cases:
        d = df[df.case == c]
        ok = d[d.status == "ok"]
        okS = ok[ok.groups_used > 0]
        rows.append(dict(case=c, n_sampled=len(d), n_ok=len(ok), n_spline_ok=len(okS),
                         status_counts=json.dumps(d.status.value_counts().to_dict()),
                         median_n_obs=ok.n_obs.median(), median_n_parents=ok.n_parents.median(),
                         median_adjr2_ols=ok.adjr2_ols.median(),
                         median_d_adjr2_reset=ok.d_adjr2_reset.median(),
                         q25_d_adjr2_reset=ok.d_adjr2_reset.quantile(.25), q75_d_adjr2_reset=ok.d_adjr2_reset.quantile(.75),
                         median_d_adjr2_spline=okS.d_adjr2_spline.median(),
                         q25_d_adjr2_spline=okS.d_adjr2_spline.quantile(.25), q75_d_adjr2_spline=okS.d_adjr2_spline.quantile(.75),
                         frac_reset_p05=(ok.reset_p < 0.05).mean(),
                         frac_bic_reset_gt2=(ok.dbic_reset > 2).mean(), frac_bic_reset_gt10=(ok.dbic_reset > 10).mean(),
                         frac_bic_spline_gt2=(okS.dbic_spline > 2).mean(), frac_bic_spline_gt10=(okS.dbic_spline > 10).mean(),
                         median_groups_used=okS.groups_used.median(),
                         n_group_ok=int((d.get("groups_used_g", pd.Series(dtype=float)) > 0).sum()),
                         median_d_adjr2_group=d.loc[d.groups_used_g > 0, "d_adjr2_group"].median(),
                         q25_d_adjr2_group=d.loc[d.groups_used_g > 0, "d_adjr2_group"].quantile(.25),
                         q75_d_adjr2_group=d.loc[d.groups_used_g > 0, "d_adjr2_group"].quantile(.75),
                         frac_bic_group_gt2=(d.loc[d.groups_used_g > 0, "dbic_group"] > 2).mean(),
                         frac_bic_group_gt10=(d.loc[d.groups_used_g > 0, "dbic_group"] > 10).mean()))
    S = pd.DataFrame(rows)
    S.to_csv(outdir / "K_summary.csv", index=False)
    print(S.T.to_string(), flush=True)

    plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18, "legend.fontsize": 15})
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=False)
    cols = {"t2": "#999999", "debby": "#0072B2", "pinatubo": "#D55E00", "pnw": "#CC79A7"}
    for ax, met, ttl in [(axes[0], "d_adjr2_reset", "(a) RESET (+ fitted^2, fitted^3)"),
                         (axes[1], "d_adjr2_spline", "(b) + group splines, products"),
                         (axes[2], "d_adjr2_group", "(c) group-level model")]:
        data, labs = [], []
        for c in cases:
            d = df[(df.case == c)][met].dropna().values
            data.append(d); labs.append(names[c].split(" (")[0])
        pos = np.arange(len(data))
        vp = ax.violinplot([d if len(d) else [np.nan] for d in data], positions=pos, showextrema=False, widths=0.8)
        for b, c in zip(vp["bodies"], cases):
            b.set_facecolor(cols[c]); b.set_alpha(0.45); b.set_edgecolor("k")
        ax.boxplot(data, positions=pos, widths=0.25, showfliers=False, medianprops=dict(color="k", lw=2.5))
        ax.axhline(0, color="k", lw=1, ls=":")
        ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=16, rotation=20)
        ax.set_ylabel(r"Adjusted $R^2$ gain over linear (unitless)")
        ax.set_title(ttl, fontsize=20)
    fig.suptitle("Linearity check, 100 sampled tracing steps per case (T2-EN synthetic = linear reference)", fontsize=20)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(figdir / f"K_delta_adjR2.{ext}", dpi=150)
    print(f"figure: {figdir / 'K_delta_adjR2.png'}", flush=True)


if __name__ == "__main__":
    main()
