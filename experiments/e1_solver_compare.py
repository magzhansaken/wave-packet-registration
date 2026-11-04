#!/usr/bin/env python3
"""E1c - solver signals against the measured ones (actuator T4, 40 kHz, coldest measurement of D1).

For every receiver: lag of the best cross-correlation (searched within +-25 us) and the correlation coefficient at that
lag, in the window of the direct packets: from the start of the record to the later of the two direct A0 arrivals
(nominal and calibrated rays, eq. (4)) plus one burst length, i.e. before most edge reflections; nominal plate and
calibrated plate (cold). Signals are normalised in the window.
Input: results/e1_solver_T4_40k.npz (from e1_arrivals.py --solver). Output: results/e1_solver_compare.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays

R = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    z = np.load(os.path.join(R, "e1_solver_T4_40k.npz")); ts = z["t"]
    D = ogw.load_freq(40); ch = D["channels"]; cold = int(np.argmin(D["T"])); t = D["t"]; dt = t[1] - t[0]
    L, _ = ogw.path_geometry(ch); out = []
    for rcv in range(12):
        if rcv == 3:
            continue
        idx = [k for k in range(len(ch)) if set(ch[k].tolist()) == {3, rcv}]
        if not idx:
            continue
        i = idx[0]
        ta = [min(tk for (md, nm, Lk, ps, tk) in rays.packet_table(ogw.XY[3], ogw.XY[rcv], 40, modes=("A0",), order=0, tmax=2e-3, s=sv)
                  if nm == "direct") for sv in (1.0, float(z["s"]))]
        w = t <= max(ta) + 5 / 40e3
        m = D["S"][cold, i].astype(float)[w]; m -= m.mean(); m /= np.linalg.norm(m)
        row = {"rcv": rcv + 1, "L_mm": float(L[i] * 1e3), "window_end_us": float((max(ta) + 5 / 40e3) * 1e6)}
        K = int(round(25e-6 / dt))
        for name in ("nominal", "calibrated_cold"):
            x = np.interp(t, ts, z[name][rcv])[w]; x -= x.mean(); x /= np.linalg.norm(x)
            c = np.array([np.dot(m[max(0, k):len(m) + min(0, k)], x[max(0, -k):len(x) - max(0, k)]) for k in range(-K, K + 1)])
            k = int(np.argmax(c)); lag = (k - K) * dt
            row[name] = {"lag_us": float(lag * 1e6), "corr": float(c[k])}
        out.append(row)
    short = [r for r in out if r["L_mm"] <= 200]; long_ = [r for r in out if r["L_mm"] > 400]
    summ = {k: {"short_corr": [min(r[k]["corr"] for r in short), max(r[k]["corr"] for r in short)],
                "short_abs_lag_us_max": max(abs(r[k]["lag_us"]) for r in short),
                "long_corr": [min(r[k]["corr"] for r in long_), max(r[k]["corr"] for r in long_)]} for k in ("nominal", "calibrated_cold")}
    json.dump({"paths": out, "summary": summ, "T_cold": float(D["T"][cold])}, open(os.path.join(R, "e1_solver_compare.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
