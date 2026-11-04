#!/usr/bin/env python3
"""E13 (amendment 13 in docs/PROTOCOL.md, written before this run): on one set of D1 (D1_SLICES = main half,
reserve half or dam_D16), 12 fixed paths x 6 splits x 12 frequencies:
  13.2  windowed shift-and-stretch after Sharma & Deraemaeker (EWSHM 2026), adapted to prediction ('sdws'):
        per training signal a joint shift tau and stretch k in one window (k tracked from the edge, tau by a lag search
        within +-half a period, then k refined), optional gain g; tau, k, g regressed linearly in T; prediction
        g(T) s_ref(k(T) t - tau(T)) on the whole record. 6 configurations, chosen on the validation block of the 12 paths.
  13.3  drift-aware conformal baselines on the residuals of br_gp (configuration chosen in E2 of this set):
        nexCP (Barber et al. 2023; weights rho^rank by distance, rho = 0.99) and ACI (Gibbs & Candes 2021; gamma = 0.005,
        test signals in order of distance, feedback after every signal), next to the growth law and split conformal.
Rows: --out (unit = split x frequency x path)."""
import argparse, json, os, sys, time
import numpy as np
from scipy.signal import hilbert
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core
from experiments.e2_run import SPLITS, masks, subset_paths, calib_s, pooled_kappa

SDWS_GRID = [{"win": w, "amp": a} for w in ("all", 3, 6) for a in (False, True)]
RHO, GAMMA, ALPHA = 0.99, 0.005, 0.05


def window_mask(t, sref, fk, win):
    if win == "all":
        return np.ones(len(t), bool)
    env = np.abs(hilbert(sref)); i0 = int(np.argmax(env > 0.2 * env.max()))
    tb = 5.0 / (fk * 1e3); t0 = t[i0] - 0.5 * tb
    return (t >= t0) & (t <= t0 + win * tb)


def _model(t, sref, k, tau):
    return np.interp(k * t - tau, t, sref, left=0.0, right=0.0)


def sdws_laws(t, S, T, side, fk, win="all", amp=True, half=0.002, n=41, nlag=21):
    ie = core.edge_index(T, side); sref = S[ie]; Tref = T[ie]; w = window_mask(t, sref, fk, win)
    per = 1.0 / (fk * 1e3); lags = np.linspace(-0.5 * per, 0.5 * per, nlag)
    order = np.argsort(-T) if side == "up" else np.argsort(T)
    sub = set(order[np.linspace(0, len(order) - 1, min(40, len(order))).astype(int)].tolist())
    k_prev, tau_prev = 1.0, 0.0; rec = []
    for i in order:
        if i == ie:
            continue
        y = S[i][w]
        def score(k, tau):
            p = _model(t, sref, k, tau)[w]; return (p @ y) / (np.linalg.norm(p) * np.linalg.norm(y) + 1e-300)
        ks = k_prev + np.linspace(-half, half, n); k = ks[int(np.argmax([score(x, tau_prev) for x in ks]))]
        tau = (tau_prev + lags)[int(np.argmax([score(k, x) for x in tau_prev + lags]))]
        ks2 = k + np.linspace(-half / 4, half / 4, n // 2); k = ks2[int(np.argmax([score(x, tau) for x in ks2]))]
        k_prev, tau_prev = k, tau
        if i in sub:
            p = _model(t, sref, k, tau)[w]; g = (p @ y) / (p @ p + 1e-300) if amp else 1.0
            rec.append((T[i] - Tref, k, tau, g))
    r = np.array(rec)
    if len(r) < 2:
        c_k, c_t, c_g = np.array([0, 1.0]), np.array([0, 0.0]), np.array([0, 1.0])
    else:
        c_k, c_t, c_g = np.polyfit(r[:, 0], r[:, 1], 1), np.polyfit(r[:, 0], r[:, 2], 1), np.polyfit(r[:, 0], r[:, 3], 1)
    return sref, Tref, c_k, c_t, c_g


def fit_sdws(t, S, T, Tq, side, fk=60, win="all", amp=True, **kw):
    sref, Tref, ck, ct, cg = sdws_laws(t, S, T, side, fk, win, amp)
    return np.array([np.polyval(cg, x - Tref) * _model(t, sref, np.polyval(ck, x - Tref), np.polyval(ct, x - Tref)) for x in Tq]), None


def wquantile(z, w, q):
    o = np.argsort(z); cw = np.cumsum(w[o]); cw /= cw[-1]
    return float(z[o][min(np.searchsorted(cw, q), len(z) - 1)])


def intervals(Ste, P, Rcal, Ycal, d_cal, d_test):
    """Coverage per test signal and relative half-width for law, conf, nexCP, ACI on the same residuals."""
    s_t, qz, kap, dmean = core.conformal_calibrate(Rcal, d_cal, Ycal)
    rms = np.sqrt((Ste ** 2).mean(1))[:, None]; out = {}
    for lab, law in (("conf", False), ("law", True)):
        hw = core.conformal_halfwidth(s_t, qz, kap, dmean, d_test, law=law)
        out[lab] = {"cov_sig": [float(x) for x in (np.abs(Ste - P) <= hw).mean(1)], "hw_rel": float(np.mean(hw / rms))}
    z = (np.abs(Rcal) / s_t).ravel(); n, m = Rcal.shape
    rank = np.argsort(np.argsort(-d_cal)); wsig = RHO ** rank
    w = np.repeat(wsig / m, m); zz = np.append(z, np.inf); ww = np.append(w, 1.0 / m)
    qn = wquantile(zz, ww, 1 - ALPHA); qn = qn if np.isfinite(qn) else float(z.max())
    hw = qn * np.repeat(s_t[None, :], len(Ste), 0)
    out["nexcp"] = {"cov_sig": [float(x) for x in (np.abs(Ste - P) <= hw).mean(1)], "hw_rel": float(np.mean(hw / rms))}
    a = ALPHA; cov = np.zeros(len(Ste)); hws = np.zeros(len(Ste))
    for j in np.argsort(d_test, kind="stable"):
        lev = 1 - a; q = float(np.quantile(z, min(max(lev, 0.0), 1.0))) if lev < 1 else 2 * float(z.max())
        h = q * s_t; c = float((np.abs(Ste[j] - P[j]) <= h).mean()); cov[j] = c; hws[j] = float(np.mean(h / rms[j]))
        a = a + GAMMA * (ALPHA - (1 - c))
    out["aci"] = {"cov_sig": [float(x) for x in cov], "hw_rel": float(np.mean(hws))}
    out["kappa_growth"] = kap
    return out


def task(split, fk, cfg_br):
    D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split); ch = D["channels"]
    s, s_in = calib_s(split), calib_s(split, "inner")
    pk = [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s) for a, b in ch]
    pk_in = [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s_in) for a, b in ch]
    kap, kap_in = pooled_kappa(t, S[tr], T[tr], side, fk, pk), pooled_kappa(t, S[inner], T[inner], side, fk, pk_in)
    sub = subset_paths(ch)
    sc = []
    for c in SDWS_GRID:
        e = [np.mean(core.rel_err(fit_sdws(t, S[inner, i], T[inner], T[val], side, fk=fk, **c)[0], S[val, i])) for i in sub]
        sc.append(float(np.median(e)))
    best = SDWS_GRID[int(np.argmin(sc))]
    edge_tr = T[tr].max() if side == "up" else T[tr].min(); edge_in = T[inner].max() if side == "up" else T[inner].min()
    d_test = np.abs(T[te] - edge_tr); d_cal = np.abs(T[val] - edge_in); rows = []
    for i in sub:
        Str, Ste = S[tr, i], S[te, i]; m = {}
        (Pd, _), dt = core.timed(fit_sdws, t, Str, T[tr], T[te], side, fk=fk, **best)
        m["sdws"] = {"err": float(np.mean(core.rel_err(Pd, Ste))), "err_sig": [float(x) for x in core.rel_err(Pd, Ste)], "cost_s": float(dt)}
        (P, _), dt = core.timed(core.fit_br_gp, t, Str, T[tr], T[te], side, fk=fk, packets=pk[i], kappa=kap, **cfg_br)
        m["br_gp"] = {"err": float(np.mean(core.rel_err(P, Ste))), "err_sig": [float(x) for x in core.rel_err(P, Ste)], "cost_s": float(dt)}
        Pv, _ = core.fit_br_gp(t, S[inner, i], T[inner], T[val], side, fk=fk, packets=pk_in[i], kappa=kap_in, **cfg_br)
        iv = intervals(Ste, P, S[val, i] - Pv, S[val, i], d_cal, d_test)
        rows.append({"split": split, "side": side, "f_kHz": fk, "path": int(i), "cfg_sdws": best, "sdws_val_scores": sc,
                     "cfg_br_gp": {k: (str(v) if isinstance(v, float) and np.isinf(v) else v) for k, v in cfg_br.items()},
                     "dT_test": [float(x) for x in d_test], "m": m, "iv": {"br_gp": iv}})
    return rows


def load_cfg(rows_paths):
    cfg = {}
    for p in rows_paths:
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            c = r.get("cfg", {}).get("br_gp")
            if c:
                cfg[(r["split"], r["f_kHz"])] = {"lam": float(c["lam"]), "wf": float(c["wf"]), "resid": c["resid"]}
    return cfg


def _star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return task(*j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default="40,60,80,100,120,140,160,180,200,220,240,260")
    ap.add_argument("--splits", default=",".join(SPLITS)); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rows", required=True, help="comma-separated E2 rows of this set (later files override cfg)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); cfg = load_cfg(a.rows.split(",")); done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["split"], json.loads(l)["f_kHz"]) for l in open(a.out)}
    jobs = [(sp, int(f), cfg[(sp, int(f))]) for f in a.freqs.split(",") for sp in a.splits.split(",") if (sp, int(f)) not in done]
    print(f"{len(jobs)} tasks", flush=True); t0 = time.time()
    import multiprocessing as mp
    it = mp.get_context("fork").Pool(a.workers).imap_unordered(_star, jobs) if a.workers > 1 else map(_star, jobs)
    for rows in it:
        with open(a.out, "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        g = np.median([(r["m"]["sdws"]["err"] - r["m"]["br_gp"]["err"]) / r["m"]["sdws"]["err"] for r in rows])
        print(f"[{time.time() - t0:.0f}s] {rows[0]['split']} {rows[0]['f_kHz']} kHz cfg={rows[0]['cfg_sdws']} gain_vs_sdws={100 * g:.1f}%", flush=True)


if __name__ == "__main__":
    main()
