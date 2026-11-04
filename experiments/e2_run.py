#!/usr/bin/env python3
"""E2/E3/E5 runner on D1 (OGW #1): all cheap surrogates on every unit (split x frequency x path).

For each (split, frequency):
  1. calibration of the plate model on the training block only: stiffness factor s from S0 arrivals (60/100 kHz);
  2. ray (bicharacteristic) packet tables per path; pooled travel-time delay slopes kappa per window label (S0, A0, coda)
     from the training block of all 66 paths;
  3. tuning of each method's grid (configs/search_space.yaml) on the 12 fixed paths: median validation error;
  4. per path: full fit -> test errors; inner->validation fit -> stress error and conformal calibration;
     random 80/20 split of the training block -> in-range error; fit time; intervals (GP-Bayesian, conformal, law-guided).
Rows: results/e2_rows.jsonl (one JSON per unit, resumable).
"""
import argparse, itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
SPLITS = {"S1_le30_ge35": ("up", 30.0, 35.0), "S2_le35_ge40": ("up", 35.0, 40.0), "S3_le40_ge45": ("up", 40.0, 45.0),
          "S4_ge40_le35": ("down", 40.0, 35.0), "S5_ge45_le40": ("down", 45.0, 40.0), "S6_ge50_le45": ("down", 50.0, 45.0)}
GRIDS = {"copy": [{}], "knn": [{"k": k} for k in (1, 2, 3, 5)],
         "gp": [{"n_pca": n, "kernel": k} for n in (5, 10, 20) for k in ("rbf", "matern")],
         "bss": [{}], "bss_gp": [{"n_pca": n} for n in (5, 10, 20)],
         "bss_tr": [{}], "bss_tr_gp": [{"n_pca": n} for n in (5, 10, 20)],
         "br": [{}],
         "br_gp": [{"lam": l, "wf": w, "resid": r} for l in (0.0, 1000.0, np.inf) for w in (1.0, 1.5) for r in ("none", "gp")]}
VAL_FRAC, VAL_GAP = 0.25, 2.5


def masks(T, split):
    side, a, b = SPLITS[split]
    if side == "up":
        tr, te = T <= a, T >= b
    else:
        tr, te = T >= a, T <= b
    Ttr = T[tr]; lo, hi = Ttr.min(), Ttr.max(); w = VAL_FRAC * (hi - lo)
    if side == "up":
        val = tr & (T >= hi - w); inner = tr & (T < hi - w - VAL_GAP)
    else:
        val = tr & (T <= lo + w); inner = tr & (T > lo + w + VAL_GAP)
    return side, tr, te, val, inner


def subset_paths(channels, n=12):
    L, psi = ogw.path_geometry(channels)
    o = np.argsort(L)
    return sorted(o[np.linspace(0, len(o) - 1, n).astype(int)].tolist())


_S_CACHE = {}


def calib_s(split, block="tr"):
    """Stiffness factor from S0 arrivals at 60/100 kHz on the training block ('tr') or on its inner part ('inner', for
    every fit that is evaluated on the validation block)."""
    key = (split, block, VAL_FRAC, VAL_GAP)
    if key in _S_CACHE:
        return _S_CACHE[key]
    cs = []
    for fk in (60, 100):
        D = ogw.load_freq(fk); side, tr, te, val, inner = masks(D["T"], split)
        use = tr if block == "tr" else inner
        L, psi = ogw.path_geometry(D["channels"]); tc = 2.5 / (fk * 1e3); rows = []
        for n in np.where(use)[0][::3]:
            for i in np.where(L >= 0.3)[0]:
                tp, _ = ogw.pick_peak(D["t"], D["S"][n, i].astype(float), tc + L[i] / 7500, tc + L[i] / 4500)
                if np.isfinite(tp):
                    rows.append((L[i], tp))
        r = np.array(rows); A = np.c_[np.ones(len(r)), r[:, 0]]; c, *_ = np.linalg.lstsq(A, r[:, 1], rcond=None); cs.append(1 / c[1])
    c_nom = float(rays.cg_along(100e3, "S0", 0.0))
    s = float(round((np.mean(cs) / c_nom) ** 2, 3))
    _S_CACHE[key] = s
    return s


def pooled_kappa(t, S, T, side, fk, packets):
    """Pooled relative travel-time slopes per window label (S0 / A0 / pre) from the given block, all paths."""
    tc = 2.5 / (fk * 1e3); rel = {"pre": [], "S0": [], "A0": []}
    for i in range(S.shape[1]):
        ie, Tref, delta, cen, slopes = core.br_laws(t, S[:, i], T, side, fk, packets[i], {}, lam=0.0)
        _, _, labs = core.windows_from_packets(t, packets[i], fk)
        for c, sl, lb in zip(cen, slopes, labs):
            if c - tc > 50e-6:
                rel[lb].append(sl / (c - tc))
    return {k: float(np.median(v)) for k, v in rel.items() if len(v)}


def run_method(name, t, S, T, Tq, side, cfg, ctx, Sq=None, want_std=False):
    kw = dict(cfg)
    if name in ("br", "br_gp"):
        kw.update(fk=ctx["fk"], packets=ctx["packets"], kappa=ctx["kappa"])
    if name in ("dtw_or", "str_or"):
        kw["Sq"] = Sq
    if want_std and name in ("gp", "bss_gp", "br_gp", "br", "bss_tr_gp"):
        kw["want_std"] = True
    (P, sd), dt = core.timed(core.METHODS[name], t, S, T, Tq, side, **kw)
    return P, sd, dt


ONLY = [m for m in os.environ.get("E2_METHODS", "").split(",") if m]
NO_IN = os.environ.get("E2_NO_IN", "") == "1"          # confirmation pass: no in-range (random 80/20) evaluation


def task(split, fk, paths_only=None):
    D = ogw.load_freq(fk); t = D["t"]; S_all = D["S"].astype(np.float64); T = D["T"]
    S_all -= S_all.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split)
    ch = D["channels"]; L, psi = ogw.path_geometry(ch)
    s = calib_s(split); s_in = calib_s(split, "inner")
    packets = [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s) for a, b in ch]
    packets_in = [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s_in) for a, b in ch]
    # pooled travel-time slopes per window label (data-only per-window slopes, all paths). Correction of 03.11 (amendment 8):
    # every fit evaluated on the validation block (tuning, stress error, conformal calibration) uses the plate and the slopes
    # of the INNER part only; the test fit uses the whole training block; the in-range check uses the training block
    # without its held-out signals.
    rng = np.random.default_rng(1000 + fk)
    itr = np.where(tr)[0]; hold = np.zeros(len(itr), bool); hold[rng.permutation(len(itr))[: max(2, len(itr) // 5)]] = True
    kappa = pooled_kappa(t, S_all[tr], T[tr], side, fk, packets)
    need_k = (not ONLY) or any(m in ONLY for m in ("br_gp", "br"))
    kappa_in = pooled_kappa(t, S_all[inner], T[inner], side, fk, packets_in) if need_k else {}
    kappa_hold = pooled_kappa(t, S_all[itr[~hold]], T[itr[~hold]], side, fk, packets) if (need_k and not NO_IN) else {}
    sub = subset_paths(ch)
    # tuning on the 12 paths: inner -> validation
    best = {}
    for name, grid in GRIDS.items():
        if ONLY and name not in ONLY:
            continue
        if len(grid) == 1:
            best[name] = grid[0]; continue
        scores = []
        for cfg in grid:
            e = []
            for i in sub:
                ctx = {"fk": fk, "packets": packets_in[i], "kappa": kappa_in}
                P, _, _ = run_method(name, t, S_all[inner, i], T[inner], T[val], side, cfg, ctx)
                e.append(np.mean(core.rel_err(P, S_all[val, i])))
            scores.append(np.median(e))
        best[name] = grid[int(np.argmin(scores))]
    edge_tr = T[tr].max() if side == "up" else T[tr].min()
    edge_in = T[inner].max() if side == "up" else T[inner].min()
    d_test = np.abs(T[te] - edge_tr); d_cal = np.abs(T[val] - edge_in)
    rows = []
    for i in (paths_only if paths_only is not None else range(len(ch))):
        ctx = {"fk": fk, "packets": packets[i], "kappa": kappa}
        ctx_val = {"fk": fk, "packets": packets_in[i], "kappa": kappa_in}
        ctx_hold = {"fk": fk, "packets": packets[i], "kappa": kappa_hold}
        Str, Ste = S_all[tr, i], S_all[te, i]
        row = {"split": split, "side": side, "f_kHz": fk, "path": int(i), "a": int(ch[i, 0]), "b": int(ch[i, 1]), "L_m": float(L[i]),
               "subset": i in sub, "n_tr": int(tr.sum()), "n_te": int(te.sum()), "n_val": int(val.sum()), "n_inner": int(inner.sum()),
               "s": s, "s_inner": s_in, "kappa": kappa, "kappa_inner": kappa_in, "cfg": {k: {kk: (str(vv) if isinstance(vv, float) and np.isinf(vv) else vv) for kk, vv in v.items()} for k, v in best.items()},
               "dT_test": [float(x) for x in d_test], "T_test": [float(x) for x in T[te]], "m": {}, "iv": {}}
        for name in (ONLY if ONLY else list(core.CANDIDATES) + ["dtw_or", "str_or"]):
            cfg = best.get(name, {})
            want = name in ("gp", "bss_gp", "br_gp", "br", "bss_tr_gp")
            P, sd, dt = run_method(name, t, Str, T[tr], T[te], side, cfg, ctx, Sq=Ste, want_std=want)
            e = core.rel_err(P, Ste)
            m = {"err": float(np.mean(e)), "err_sig": [float(x) for x in e], "cost_s": float(dt)}
            if name in core.CANDIDATES or name in ("bss_tr", "bss_tr_gp"):
                Pv, _, _ = run_method(name, t, S_all[inner, i], T[inner], T[val], side, cfg, ctx_val, Sq=S_all[val, i])
                m["e_stress"] = float(np.mean(core.rel_err(Pv, S_all[val, i])))
                if not NO_IN:
                    Ph, _, _ = run_method(name, t, Str[~hold], T[itr[~hold]], T[itr[hold]], side, cfg, ctx_hold, Sq=Str[hold])
                    m["e_in"] = float(np.mean(core.rel_err(Ph, Str[hold])))
                if want:
                    rms = np.sqrt((Ste ** 2).mean(1))[:, None]
                    iv = {}
                    if sd is not None and np.all(np.isfinite(sd)):
                        hwb = 1.96 * sd
                        iv["bayes"] = {"cov_sig": [float(x) for x in (np.abs(Ste - P) <= hwb).mean(1)], "hw_rel": float(np.mean(hwb / rms))}
                    s_t, qz, kap, dmean = core.conformal_calibrate(S_all[val, i] - Pv, d_cal, S_all[val, i])
                    for lab, law in (("conf", False), ("law", True)):
                        hw = core.conformal_halfwidth(s_t, qz, kap, dmean, d_test, law=law)
                        iv[lab] = {"cov_sig": [float(x) for x in (np.abs(Ste - P) <= hw).mean(1)], "hw_rel": float(np.mean(hw / rms))}
                    iv["kappa_growth"] = kap
                    row["iv"][name] = iv
            row["m"][name] = m
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default=",".join(map(str, ogw.MAIN)))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(OUT, "e2_rows.jsonl"))
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["split"], r["f_kHz"]))
    jobs = [(sp, int(f)) for f in a.freqs.split(",") for sp in a.splits.split(",") if (sp, int(f)) not in done]
    print(f"{len(jobs)} (split, frequency) tasks; {len(done)} already done", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            for rows in pool.imap_unordered(_task_star, jobs):
                _write(a.out, rows, t0)
    else:
        for j in jobs:
            _write(a.out, _task_star(j), t0)


def _task_star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return task(*j)


def _write(path, rows, t0):
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    r = rows[0]
    e = {k: np.mean([x["m"][k]["err"] for x in rows]) for k in rows[0]["m"]}
    cov = np.mean([np.mean(x["iv"]["br_gp"]["law"]["cov_sig"]) for x in rows]) if "br_gp" in rows[0]["iv"] else float("nan")
    print(f"[{time.time() - t0:.0f}s] {r['split']} {r['f_kHz']} kHz: " + " ".join(f"{k}={v:.3f}" for k, v in e.items()) + f" | br_gp law cov={cov:.3f}", flush=True)


if __name__ == "__main__":
    main()
