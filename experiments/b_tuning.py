#!/usr/bin/env python3
"""Variant B, step 1 (amendment 10): excitability of S0 and A0 from the bicharacteristic-splitting solver.

For each of the 12 frequencies of D1, the membrane and the bending subsystems are run separately on a square patch of the
NOMINAL plate (no measured information) with a piezo disk (r = 5 mm, five-cycle Hann burst) in the centre and receiving
disks (r = 5 mm) on a ring of 50 mm at 8 directions psi = 0, 22.5, ..., 157.5 deg. The patch half-width is chosen so
that no edge reflection reaches a receiver before the direct packet has passed:
    M = R + c_g,max * (5/f) / 2 + 10 mm   (c_g,max: S0 for the membrane, A0 for bending).
Grid: membrane dx = min(4 mm, lambda_SH0/20); bending dx = lambda_A0/16. Yoshida composition, p = 5, nu = 0.9.
Excitability T_m(f, psi) = sqrt(energy of the received top-surface strain). Output: results/b_tuning.json."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs.laminate import laminate
from bcs import system as sy
from bcs.solver import Subsystem, hann_burst

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
FREQS = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
R, RD = 0.05, 5e-3
ANG = np.deg2rad(np.arange(0, 180, 22.5))


def job(fk, kind):
    lam = laminate(); f = fk * 1e3
    cp_sh, _ = sy.phase_group(lam, f, 0.0, "SH0"); cp_a, cg_a = sy.phase_group(lam, f, 0.0, "A0"); cp_s, cg_s = sy.phase_group(lam, f, 0.0, "S0")
    if kind == "membrane":
        dx = min(4e-3, cp_sh / f / 20); cfast = np.linalg.norm(cg_s); cslow = cp_sh
    else:
        dx = cp_a / f / 16; cfast = np.linalg.norm(cg_a); cslow = np.linalg.norm(cg_a)
    M = R + cfast * (5 / f) / 2 + 0.01
    L = 2 * M; Tend = 2.5 / f + R / cslow + 2.5 / f + 10e-6
    t0 = time.time()
    S = Subsystem(lam, kind, L, L, dx, cfl=0.9, p=5, scheme="yoshida")
    S.add_actuator(M, M, RD, lambda t: hann_burst(t, f, 5))
    for a in ANG:
        S.add_sensor(M + R * np.cos(a), M + R * np.sin(a), RD)
    ts, strain = S.run(Tend)
    dt = np.diff(ts).mean()
    E = (strain ** 2).sum(0) * dt
    env_peak = [float(ts[np.argmax(np.abs(strain[:, j]))]) for j in range(len(ANG))]
    return {"f_kHz": fk, "kind": kind, "dx_mm": dx * 1e3, "L_mm": L * 1e3, "T_us": Tend * 1e6, "steps": int(S.nstep),
            "nodes": int(S.nx * S.ny), "amp": [float(np.sqrt(e)) for e in E], "peak_time_us": [t * 1e6 for t in env_peak],
            "psi_deg": list(np.rad2deg(ANG)), "sec": round(time.time() - t0, 1)}


def main():
    out = os.path.join(OUT, "b_tuning.json")
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["f_kHz"], r["kind"]) for r in res}
    jobs = [(fk, k) for fk in FREQS for k in ("membrane", "bending") if (fk, k) not in done]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(int(os.environ.get("WORKERS", 2))) as pool:
        for r in pool.imap_unordered(_star, jobs):
            res.append(r); json.dump(res, open(out, "w"), indent=1)
            print(f"{r['f_kHz']} kHz {r['kind']}: dx={r['dx_mm']:.2f} mm L={r['L_mm']:.0f} mm nodes={r['nodes']} steps={r['steps']} "
                  f"amp={np.mean(r['amp']):.3e} ({r['sec']} s)", flush=True)


def _star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return job(*j)


if __name__ == "__main__":
    main()
