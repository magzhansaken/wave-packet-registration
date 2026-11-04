#!/usr/bin/env python3
"""Variant B (amendment 10): physics index E_phys and the rule "registration or stretch" from the solver and the training
block only. No measured error of any method enters.

Steps per split (training block only):
  1. plate stiffness factor s: S0 arrivals at 60/100 kHz (experiments.e2_run.calib_s);
  2. temperature law: measured relative slopes d ln t / dT of S0 (60, 100 kHz) and A0 (40, 60 kHz) from tracked packet
     lags in the training block (experiments.e1_arrivals.thermal_lags, restricted to the training block); matrix softening
     alpha and fibre softening alpha_f by least squares on the four median slopes (two Gauss-Newton steps from 0, 0);
  3. per frequency and path: ray table (S0, A0; direct and single edge reflections; calibrated plate at the training-edge
     temperature), packet amplitudes a_k = T_m(f, psi_k) * sqrt(L0 / L_k) with the solver excitability (results/b_tuning.json),
     windows as in br, window energy p_w, delay slope b_w = sum_k a_k^2 kappa_k u_w / sum_k a_k^2 with
     kappa_k = -d ln c_g,m / dT (calibrated law), best single stretch beta, residual r_w and
     E_phys(d) = sqrt(sum_w p_w (2 - 2 rho_5(r_w d) cos(2 pi f r_w d))) at d_ref = 15 C.
Outputs (directory OUT, default results/): b_calib_<tag>.json, b_phys_rows_<tag>.jsonl (TAG env, e.g. reserve)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core
from experiments.e2_run import SPLITS, masks, calib_s
from experiments.e1_arrivals import thermal_lags
from experiments.e4_law import rho_hann

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
TAG = os.environ.get("TAG", "dev")
D_REF, L0 = 15.0, 0.05
CAL = (("S0", 60), ("S0", 100), ("A0", 40), ("A0", 60))


def tuning_table():
    tb = json.load(open(os.path.join(OUT, "b_tuning.json")))
    tab = {}
    for r in tb:
        m = "S0" if r["kind"] == "membrane" else "A0"
        tab[(r["f_kHz"], m)] = (np.deg2rad(np.array(r["psi_deg"])), np.array(r["amp"]))
    return tab


def amp(tab, fk, mode, psi):
    ang, a = tab[(fk, mode)]
    ang = np.concatenate([ang - np.pi, ang, ang + np.pi]); a = np.concatenate([a, a, a])
    return float(np.interp(np.mod(psi, np.pi), ang, a))


def model_slopes(s, alpha, alpha_f, T_edge, psi):
    out = []
    for mode, fk in CAL:
        f = fk * 1e3
        c0 = rays.cg_along(f, mode, psi, s=s, alpha=alpha, alpha_f=alpha_f, T=round(T_edge, 1), T0=20.0)
        c1 = rays.cg_along(f, mode, psi, s=s, alpha=alpha, alpha_f=alpha_f, T=round(T_edge + 5.0, 1), T0=20.0)
        out.append(float(-(np.log(c1) - np.log(c0)) / 5.0))
    return np.array(out)


def calib_thermal(split):
    s = calib_s(split); meas = []
    for mode, fk in CAL:
        D = ogw.load_freq(fk); side, tr, te, val, inner = masks(D["T"], split)
        Dt = dict(D, S=D["S"][tr], T=D["T"][tr])
        lags, tof, dT = thermal_lags(Dt, fk, mode, dict(s=s))
        sl = [np.polyfit(dT[np.isfinite(lags[:, i])], lags[np.isfinite(lags[:, i]), i], 1)[0] / tof[i]
              for i in range(lags.shape[1]) if np.isfinite(lags[:, i]).sum() > 5]
        meas.append(float(np.median(sl)))
        T_edge = float(Dt["T"].max() if side == "up" else Dt["T"].min())
    meas = np.array(meas); psi = np.pi / 2
    x = np.zeros(2); h = np.array([5e-4, 2e-4])
    for _ in range(2):                                   # Gauss-Newton on (alpha, alpha_f), residual relative to measured
        m0 = model_slopes(s, x[0], x[1], T_edge, psi)
        J = np.stack([(model_slopes(s, x[0] + h[0], x[1], T_edge, psi) - m0) / h[0],
                      (model_slopes(s, x[0], x[1] + h[1], T_edge, psi) - m0) / h[1]], 1)
        W = 1 / meas
        dx, *_ = np.linalg.lstsq(J * W[:, None], (meas - m0) * W, rcond=None)
        x = x + dx
    mf = model_slopes(s, x[0], x[1], T_edge, psi)
    return {"split": split, "s": s, "alpha": float(x[0]), "alpha_f": float(x[1]), "T_edge": T_edge,
            "measured_slopes": dict(zip([f"{m}_{f}" for m, f in CAL], meas.tolist())),
            "model_slopes_fitted": dict(zip([f"{m}_{f}" for m, f in CAL], mf.tolist())),
            "rel_misfit": float(np.sqrt(np.mean(((mf - meas) / meas) ** 2)))}


def e_phys_paths(fk, cal, tab, t, ch, d_list=(D_REF,)):
    f = fk * 1e3; tc = 2.5 / f; s, al, af, Te = cal["s"], cal["alpha"], cal["alpha_f"], cal["T_edge"]
    kw = dict(s=s, alpha=al, alpha_f=af, T=round(Te, 1), T0=20.0)
    rows = []
    for i, (a_, b_) in enumerate(ch):
        pk = rays.packet_table(ogw.XY[a_], ogw.XY[b_], fk, modes=("S0", "A0"), order=1, tmax=t[-1], **kw)
        cen, hw, _ = core.windows_from_packets(t, pk, fk)
        num = np.zeros(len(cen)); den = np.zeros(len(cen)); qa0 = np.zeros(len(cen))
        for (m, name, L, psi, tk) in pk:
            w = np.where(np.abs(cen - tk) <= hw)[0]
            if not len(w):
                continue
            j = w[np.argmin(np.abs(cen[w] - tk))]
            a2 = amp(tab, fk, m, psi) ** 2 * (L0 / L)
            c0 = rays.cg_along(f, m, psi, **kw); c1 = rays.cg_along(f, m, psi, **dict(kw, T=round(Te + 5.0, 1)))
            kap = -(np.log(c1) - np.log(c0)) / 5.0
            num[j] += a2 * kap; den[j] += a2; qa0[j] += a2 * (m == "A0")
        ok = den > 0
        if not ok.any():
            rows.append({"path": i, "E_phys": {str(d): float("nan") for d in d_list}}); continue
        u = cen[ok] - tc; p = den[ok] / den[ok].sum(); b = (num[ok] / den[ok]) * u
        beta = float(np.sum(p * b * u) / np.sum(p * u * u)); r = b - beta * u
        E = {}
        for d in d_list:
            E[str(d)] = float(np.sqrt(max(np.sum(p * (2 - 2 * rho_hann(np.abs(r * d), f) * np.cos(2 * np.pi * f * r * d))), 0.0)))
        rows.append({"path": i, "E_phys": E, "share_A0": float(np.sum(qa0[ok]) / den[ok].sum()), "beta": beta,
                     "n_windows": int(ok.sum())})
    return rows


def main():
    tab = tuning_table()
    cpath = os.path.join(OUT, f"b_calib_{TAG}.json")
    cals = json.load(open(cpath)) if os.path.exists(cpath) else {}
    import multiprocessing as mp
    need = [sp for sp in SPLITS if sp not in cals]
    if need:
        with mp.get_context("fork").Pool(int(os.environ.get("WORKERS", 2))) as pool:
            for c in pool.imap_unordered(calib_thermal, need):
                cals[c["split"]] = c; json.dump(cals, open(cpath, "w"), indent=1)
                print(f"calibrated {c['split']}: s={c['s']} alpha={c['alpha']:.2e} alpha_f={c['alpha_f']:.2e} misfit={c['rel_misfit']:.2f}", flush=True)
    rpath = os.path.join(OUT, f"b_phys_rows_{TAG}.jsonl")
    done = set()
    if os.path.exists(rpath):
        done = {(json.loads(l)["split"], json.loads(l)["f_kHz"]) for l in open(rpath)}
    jobs = [(sp, fk) for fk in ogw.FREQS for sp in SPLITS if (sp, fk) not in done]
    with mp.get_context("fork").Pool(int(os.environ.get("WORKERS", 2))) as pool:
        for rows in pool.imap_unordered(_task, [(sp, fk, cals[sp]) for sp, fk in jobs]):
            with open(rpath, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"{rows[0]['split']} {rows[0]['f_kHz']} kHz: median E_phys(15 C) = {np.nanmedian([r['E_phys'][str(D_REF)] for r in rows]):.3f}", flush=True)


def _task(a):
    split, fk, cal = a
    tab = tuning_table()
    D = ogw.load_freq(fk, max_files=1)
    rows = e_phys_paths(fk, cal, tab, D["t"], D["channels"])
    for r in rows:
        r.update(split=split, f_kHz=fk)
    return rows


if __name__ == "__main__":
    main()
