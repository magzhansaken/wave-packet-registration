#!/usr/bin/env python3
"""E1 (part 2) - the plate model against the measured plate (OGW #1): packet arrival times and their thermal laws.

Calibration uses ONLY training-side data (plate temperature <= 30 C for stiffness, <= 40 C for the thermal law):
  s     - one global stiffness factor, from S0 arrivals at 60 and 100 kHz (clean early packet), T <= 30 C;
  alpha - thermal factor of the matrix-dominated constants, from A0 delays at 40 kHz, 20 <= T <= 40 C.
Held-out checks (not used in calibration):
  (i)   A0 group arrivals at 40 and 60 kHz (nominal model vs calibrated model vs measured);
  (ii)  S0 thermal delay law at 100 kHz (other mode);
  (iii) A0 thermal delay law at 60 kHz (other frequency);
  (iv)  A0 delays at T >= 50 C extrapolated from the 20-40 C law.
Optional (--solver): full-plate BCS runs at 40 kHz, actuator T4 (nominal; calibrated at the coldest and the hottest
measured temperature): same picking on simulated and measured signals.
Output: results/e1_arrivals.json (+ results/e1_solver_T4_40k.npz with --solver).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import rays, ogw
from bcs.laminate import laminate, M21_T700, LAYUP_OGW1

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))


def scaled_lam(s=1.0, alpha=0.0, T=20.0, T0=20.0):
    """Laminate of OGW #1 with stiffness factor s and thermal factor alpha at temperature T."""
    m = dict(M21_T700)
    for k in ("C11", "C12", "C13", "C22", "C23", "C33", "G23", "G13", "G12"):
        m[k] *= s
    return laminate(m, LAYUP_OGW1, T=T, T0=T0, alpha=alpha)


def s0_picks(D, fk, Tmax=30.0, Lmin=0.3):
    """Arrival picks of the S0 direct packet on long paths at T <= Tmax, with a line fit."""
    tc = 2.5 / (fk * 1e3); L, psi = ogw.path_geometry(D["channels"])
    sel = np.where(D["T"] <= Tmax)[0]
    rows = []
    for n in sel:
        for i in np.where(L >= Lmin)[0]:
            tp, _ = ogw.pick_peak(D["t"], D["S"][n, i].astype(float), tc + L[i] / 7500, tc + L[i] / 4500)
            rows.append((L[i], tp))
    r = np.array(rows); r = r[np.isfinite(r[:, 1])]
    A = np.c_[np.ones(len(r)), r[:, 0]]; c, *_ = np.linalg.lstsq(A, r[:, 1], rcond=None)
    return 1 / c[1], c[0], float(np.std(r[:, 1] - A @ c))


def a0_picks(D, fk, lam_kw, Tmax=30.0, widen=0.25):
    """A0 envelope peaks in a window +-widen around the model direct arrival (direct path, all 66 paths)."""
    f = fk * 1e3; tc = 2.5 / f; L, psi = ogw.path_geometry(D["channels"])
    tm = tc + L / rays.cg_along(f, "A0", psi, **lam_kw)
    sel = np.where(D["T"] <= Tmax)[0]
    P = np.full((len(sel), len(L)), np.nan)
    for a, n in enumerate(sel):
        for i in range(len(L)):
            tof = tm[i] - tc
            P[a, i], _ = ogw.pick_peak(D["t"], D["S"][n, i].astype(float), tc + (1 - widen) * tof, tc + (1 + widen) * tof)
    return np.nanmedian(P, 0), tm, L, psi


def thermal_lags(D, fk, mode, lam_kw, win=(0.85, 1.25)):
    """Per-path delay of the packet window relative to the coldest measurement, against T.
    Tracking: measurements sorted by T; each lag is searched within +-0.4 period around the lag of the previous
    (colder) measurement, so the lag never skips a cycle (steps in T are ~0.25 C)."""
    f = fk * 1e3; tc = 2.5 / f; L, psi = ogw.path_geometry(D["channels"])
    tm = tc + L / rays.cg_along(f, mode, psi, **lam_kw)
    o = np.argsort(D["T"]); ref = o[0]; dt = D["t"][1] - D["t"][0]
    lags = np.full((len(D["T"]), len(L)), np.nan)
    for i in range(len(L)):
        tof = tm[i] - tc
        lo, hi = tc + win[0] * tof, tc + win[1] * tof + 2.5 / f
        w = np.where((D["t"] >= lo) & (D["t"] <= hi))[0]
        if len(w) < 8:
            continue
        tap = np.hanning(len(w)); x = D["S"][ref, i, w].astype(float) * tap
        prev = 0.0
        for n in o:
            y_full = D["S"][n, i].astype(float)
            best = None
            # brute-force sub-sample search around the previous lag (fractional shifts by interpolation)
            cands = prev + np.linspace(-0.4 / f, 0.4 / f, 81)
            vals = []
            for c in cands:
                ys = np.interp(D["t"][w] + c, D["t"], y_full) * tap
                vals.append(np.dot(ys, x))
            vals = np.array(vals); j = int(np.argmax(vals))
            if 0 < j < len(vals) - 1:
                y0, y1, y2 = vals[j - 1], vals[j], vals[j + 1]; den = y0 - 2 * y1 + y2
                frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            else:
                frac = 0.0
            lag = cands[j] + frac * (cands[1] - cands[0])
            lags[n, i] = lag; prev = lag
    return lags, tm - tc, D["T"] - D["T"][ref]


def model_rel_slope(fk, mode, s, alpha, T0=20.0, dT=10.0):
    """Relative travel-time slope d ln(tof)/dT of a mode predicted by the plate model."""
    f = fk * 1e3
    c0 = rays.cg_along(f, mode, np.pi / 2, s=s, alpha=alpha, T=T0, T0=T0)
    c1 = rays.cg_along(f, mode, np.pi / 2, s=s, alpha=alpha, T=T0 + dT, T0=T0)
    return float(-(np.log(c1) - np.log(c0)) / dT)        # d ln(tof) / dT


def main(args):
    """Calibration on the training side and the held-out arrival checks; writes results/e1_arrivals.json."""
    os.makedirs(OUT, exist_ok=True)
    res = {"note": "calibration on T<=30 C (stiffness) and 20<=T<=40 C (thermal law); checks are held out"}
    D40, D60, D100 = ogw.load_freq(40), ogw.load_freq(60), ogw.load_freq(100)
    res["n_files"] = {"40": len(D40["T"]), "60": len(D60["T"]), "100": len(D100["T"])}
    res["T_range"] = [float(D40["T"].min()), float(D40["T"].max())]
    T0 = float(np.min(D40["T"]))
    # --- stiffness factor from S0 (60, 100 kHz), T <= 30 C
    c_nom = float(rays.cg_along(100e3, "S0", 0.0))
    s0 = {fk: s0_picks(D, fk) for fk, D in ((60, D60), (100, D100))}
    c_meas = float(np.mean([v[0] for v in s0.values()]))
    s = (c_meas / c_nom) ** 2
    res["S0_fit"] = {str(k): {"c": v[0], "t0_us": v[1] * 1e6, "resid_sd_us": v[2] * 1e6} for k, v in s0.items()}
    res["S0_nominal"] = c_nom; res["s"] = s
    print(f"S0 measured {c_meas:.0f} m/s vs nominal {c_nom:.0f} -> s = {s:.3f}", flush=True)
    # --- (i) A0 group arrivals at 40 and 60 kHz: nominal vs calibrated
    res["A0_arrival"] = {}
    for fk, D in ((40, D40), (60, D60)):
        out = {}
        for name, kw in (("nominal", dict(s=1.0)), ("calibrated", dict(s=s))):
            pk, tm, L, psi = a0_picks(D, fk, dict(s=s))          # same window for both (calibrated model)
            tmod = 2.5 / (fk * 1e3) + L / rays.cg_along(fk * 1e3, "A0", psi, **kw)
            ok = np.isfinite(pk)
            rel = (tmod[ok] - pk[ok]) / (pk[ok] - 2.5 / (fk * 1e3))
            out[name] = {"mean_rel_tof_err_pct": float(100 * np.mean(rel)), "mae_rel_pct": float(100 * np.mean(np.abs(rel))),
                         "mae_us": float(1e6 * np.mean(np.abs(tmod[ok] - pk[ok]))), "n_paths": int(ok.sum())}
        A = np.c_[np.ones(ok.sum()), L[ok]]; c, *_ = np.linalg.lstsq(A, pk[ok], rcond=None)
        out["measured_fit_cg"] = float(1 / c[1]); out["measured_fit_t0_us"] = float(c[0] * 1e6)
        out["model_cg_mean"] = {"nominal": float(np.mean(rays.cg_along(fk * 1e3, "A0", psi[ok], s=1.0))),
                                "calibrated": float(np.mean(rays.cg_along(fk * 1e3, "A0", psi[ok], s=s)))}
        res["A0_arrival"][str(fk)] = out
        print("A0", fk, out, flush=True)
    # --- thermal law of A0 at 40 kHz on 20..40 C -> alpha
    lags40, tof40, dT40 = thermal_lags(D40, 40, "A0", dict(s=s))
    fit = (D40["T"] <= 40.0)
    rel_slopes = []
    for i in range(lags40.shape[1]):
        ok = fit & np.isfinite(lags40[:, i])
        if ok.sum() > 5:
            rel_slopes.append(np.polyfit(dT40[ok], lags40[ok, i], 1)[0] / tof40[i])
    k40 = float(np.median(rel_slopes))
    a_grid = np.array([-6e-3, -3e-3, 0.0]); m_grid = np.array([model_rel_slope(40, "A0", s, a) for a in a_grid])
    alpha = float(np.interp(k40, m_grid, a_grid) if np.all(np.diff(m_grid) < 0) else np.polyval(np.polyfit(m_grid, a_grid, 1), k40))
    alpha = float(np.polyval(np.polyfit(m_grid, a_grid, 1), k40))
    res["alpha"] = alpha; res["A0_40k_rel_slope_per_C"] = k40
    print(f"A0 40 kHz d ln tof/dT = {k40:.2e} 1/C -> alpha = {alpha:.2e} 1/C", flush=True)
    # --- (ii) S0 at 100 kHz, (iii) A0 at 60 kHz: measured vs predicted slopes (whole range), (iv) extrapolation
    checks = {}
    for tag, fk, mode, D in (("S0_100k", 100, "S0", D100), ("A0_60k", 60, "A0", D60), ("A0_40k", 40, "A0", D40)):
        lags, tof, dT = (lags40, tof40, dT40) if tag == "A0_40k" else thermal_lags(D, fk, mode, dict(s=s))
        sl = []
        for i in range(lags.shape[1]):
            ok = np.isfinite(lags[:, i]) & (D["T"] <= 40)
            if ok.sum() > 5:
                sl.append(np.polyfit(dT[ok], lags[ok, i], 1)[0] / tof[i])
        pred = model_rel_slope(fk, mode, s, alpha)
        c = {"measured_rel_slope": float(np.median(sl)), "iqr": [float(np.percentile(sl, 25)), float(np.percentile(sl, 75))],
             "model_rel_slope": pred, "n_paths": len(sl)}
        hot = D["T"] >= 50
        if hot.any():
            # extrapolation of the per-path 20-40 C law vs model law, at T >= 50 C
            e_lin, e_mod = [], []
            for i in range(lags.shape[1]):
                ok = np.isfinite(lags[:, i]) & (D["T"] <= 40)
                if ok.sum() > 5 and np.isfinite(lags[hot, i]).all():
                    p = np.polyfit(dT[ok], lags[ok, i], 1)
                    e_lin.append(np.mean(np.abs(np.polyval(p, dT[hot]) - lags[hot, i])))
                    e_mod.append(np.mean(np.abs(pred * tof[i] * dT[hot] - lags[hot, i])))
            c["extrap_T>=50_mae_ns"] = {"per_path_linear_fit": float(1e9 * np.mean(e_lin)), "model_law": float(1e9 * np.mean(e_mod)),
                                        "measured_mean_abs_lag_ns": float(1e9 * np.nanmean(np.abs(lags[hot])))}
        checks[tag] = c
        print(tag, c, flush=True)
    res["thermal_checks"] = checks
    json.dump(res, open(os.path.join(OUT, "e1_arrivals.json"), "w"), indent=1)
    if args.solver:
        run_solver(res, D40, s, alpha, T0, args)


def run_solver(res, D40, s, alpha, T0, args):
    """Full-plate solver runs for actuator T4 at 40 kHz (nominal, calibrated cold and hot)."""
    from bcs.solver import simulate_pitch_catch
    T_hot = float(np.max(D40["T"]))
    cases = [("nominal", 1.0, 0.0, T0), ("calibrated_cold", s, alpha, T0), ("calibrated_hot", s, alpha, T_hot)]
    sims = {}
    for name, ss, aa, TT in cases:
        lam = scaled_lam(ss, aa, TT, T0)
        t0 = time.time()
        t, sig, parts = simulate_pitch_catch(lam, ogw.XY, 3, 40e3, args.tmax, h_mem=args.h_mem, h_bend=args.h_bend, r=ogw.R_PZT,
                                             cfl=0.9, p=5, scheme="yoshida")
        sims[name] = sig; sims["t"] = t
        sims[name + "_membrane"] = parts["membrane"]; sims[name + "_bending"] = parts["bending"]
        print(f"solver {name}: {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(os.path.join(OUT, "e1_solver_T4_40k.npz"), **sims, T0=T0, T_hot=T_hot, s=s, alpha=alpha)
    # compare A0 picks on paths from T4 (index 3)
    ch = D40["channels"]; L, psi = ogw.path_geometry(ch); tc = 2.5 / 40e3
    rows = []
    cold = int(np.argmin(D40["T"])); hot = int(np.argmax(D40["T"]))
    for i, (a, b) in enumerate(ch):
        if 3 not in (a, b):
            continue
        rcv = b if a == 3 else a
        tm = tc + L[i] / rays.cg_along(40e3, "A0", psi[i], s=s)
        lo, hi = tc + 0.75 * (tm - tc), tc + 1.25 * (tm - tc)
        pm_c, _ = ogw.pick_peak(D40["t"], D40["S"][cold, i].astype(float), lo, hi)
        pm_h, _ = ogw.pick_peak(D40["t"], D40["S"][hot, i].astype(float), lo, hi)
        r = {"rcv": int(rcv) + 1, "L_mm": float(L[i] * 1e3), "meas_cold_us": pm_c * 1e6, "meas_hot_us": pm_h * 1e6}
        for name in ("nominal", "calibrated_cold", "calibrated_hot"):
            r[name + "_us"] = ogw.pick_peak(sims["t"], sims[name][rcv], lo if name != "nominal" else tc + 0.6 * (tm - tc), hi)[0] * 1e6
        # thermal delay cold->hot: measured by cross-correlation, simulated by cross-correlation
        r["meas_delay_ns"] = 1e9 * ogw.xcorr_lag(D40["t"], D40["S"][cold, i].astype(float), D40["S"][hot, i].astype(float), lo, hi + 2.5 / 40e3, 0.6 / 40e3)
        r["sim_delay_ns"] = 1e9 * ogw.xcorr_lag(sims["t"], sims["calibrated_cold"][rcv], sims["calibrated_hot"][rcv], lo, hi + 2.5 / 40e3, 0.6 / 40e3)
        rows.append(r)
        print(r, flush=True)
    res["solver_T4_40k"] = {"T_cold": float(D40["T"][cold]), "T_hot": float(D40["T"][hot]), "paths": rows,
                            "h_bend_mm": args.h_bend * 1e3, "h_mem_mm": args.h_mem * 1e3}
    json.dump(res, open(os.path.join(OUT, "e1_arrivals.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", action="store_true")
    ap.add_argument("--tmax", type=float, default=0.65e-3)
    ap.add_argument("--h_bend", type=float, default=1.0e-3)
    ap.add_argument("--h_mem", type=float, default=2.5e-3)
    main(ap.parse_args())
