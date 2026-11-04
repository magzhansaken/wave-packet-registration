#!/usr/bin/env python3
"""E1 (part 1) - verification of the bicharacteristic-splitting solver on exact solutions.

(a) Order: exact plane waves of the anisotropic laminate (eigenvectors of the symbol), periodic square, oblique wave
    vector k = 2 pi (1, 2) / L; three grids (h, h/2, h/4) at fixed Courant number; A0 (bending), S0 and SH0 (membrane);
    second-order (Strang) and fourth-order (Yoshida) compositions, interpolation degree p = 5.
    Error: energy-norm relative error after one period; EOC = log2(e_h / e_{h/2}).
(b) Energy: free (traction-free) plate 40 x 30 mm, smooth pulse, three grids; relative energy change at 60 us.
(c) Courant limit: energy growth after 200 steps from random data at nu = 0.5 ... 1.3.
Output: results/e1_numerics.json and a short text table on stdout.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs.laminate import laminate
from bcs import system as sy
from bcs.solver import Subsystem

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(OUT, exist_ok=True)
LAM = laminate()


def plane(S, k, kind, branch):
    """Exact plane wave of one branch on the grid of subsystem S."""
    G = sy.symbol(LAM, *k); ev, V = np.linalg.eig(G); w = (1j * ev).real
    o = np.argsort(w); ev, V, w = ev[o], V[:, o], w[o]
    cand = [i for i in np.where(w > 1)[0] if np.linalg.norm(V[S.idx, i]) > 0.5 * np.linalg.norm(V[:, i])]
    i = {"A0": cand[0], "SH0": cand[0], "S0": cand[1]}[branch]
    X, Y = np.meshgrid(S.x, S.y, indexing="ij")
    v = V[S.idx, i] / np.abs(V[S.idx, i]).max()
    return np.moveaxis(np.exp(1j * (k[0] * X + k[1] * Y))[..., None] * v, -1, 0), ev[i], w[i]


def order_case(branch, scheme, cfl, Ns, L=0.06, periods=1.0, p=5):
    """Energy-norm error after the given periods on a sequence of grids (order of accuracy)."""
    kind = "bending" if branch == "A0" else "membrane"
    k = 2 * np.pi * np.array([1, 2]) / L
    rows = []
    for N in Ns:
        h = L / N
        S = Subsystem(LAM, kind, L, L, h, cfl=cfl, p=p, periodic=True, scheme=scheme)
        B, ev, w = plane(S, k, kind, branch)
        T = periods * 2 * np.pi / w; ns = int(np.ceil(T / S.dt))
        S = Subsystem(LAM, kind, L, L, h, dt=T / ns, p=p, periodic=True, scheme=scheme)
        S.q[:] = B.real; t0 = time.time()
        for _ in range(ns):
            S.step()
        qe = (B * np.exp(ev * T)).real
        E = sy.energy_matrix(LAM)[np.ix_(S.idx, S.idx)]; sc = np.sqrt(np.diag(E))[:, None, None]
        err = float(np.sqrt((((S.q - qe) * sc) ** 2).sum() / ((qe * sc) ** 2).sum()))
        lam_w = L / np.sqrt(5)
        rows.append(dict(N=N, h_mm=h * 1e3, ppw=lam_w / h, cfl=S.cfl, steps=ns, f_kHz=w / 2 / np.pi / 1e3, err=err,
                         sec=time.time() - t0))
    for a, b in zip(rows[:-1], rows[1:]):
        b["eoc"] = float(np.log2(a["err"] / b["err"]))
    return rows


def energy_case(kind, scheme, hs, T=60e-6):
    """Energy history of a smooth pulse on a free and on a periodic plate."""
    rows = []
    for h in hs:
        for per in (False, True):
            S = Subsystem(LAM, kind, 0.04, 0.03, h, cfl=0.9, p=5, scheme=scheme, periodic=per)
            X, Y = np.meshgrid(S.x, S.y, indexing="ij")
            g = np.exp(-((X - 0.015) ** 2 + (Y - 0.012) ** 2) / (2 * 0.005 ** 2))
            S.q[S.names.index("vx" if kind == "membrane" else "vz")] = g
            E0 = S.energy(); emax = 1.0
            while S.t < T - 1e-12:
                S.step(); emax = max(emax, S.energy() / E0)
            rows.append(dict(kind=kind, scheme=scheme, h_mm=h * 1e3, edges="periodic" if per else "free",
                             dE_rel=float(S.energy() / E0 - 1), max_E_over_E0=float(emax), steps=S.nstep))
    return rows


def cfl_case(kind, scheme, p, nus, nsteps=200):
    """Energy growth of white noise over nsteps at several Courant numbers."""
    rows = []
    for nu in nus:
        S = Subsystem(LAM, kind, 0.03, 0.03, 1e-3, cfl=nu, p=p, scheme=scheme, periodic=True)
        rng = np.random.default_rng(0)
        scale = np.array([1e4 if n[0] in "NMQ" else 1.0 for n in S.names])[:, None, None]
        S.q[:] = rng.standard_normal(S.q.shape) * scale
        E0 = S.energy()
        for _ in range(nsteps):
            S.step()
            if not np.isfinite(S.q).all():
                break
        g = S.energy() / E0
        rows.append(dict(kind=kind, scheme=scheme, p=p, nu=nu, growth=float(g) if np.isfinite(g) else float("inf")))
    return rows


if __name__ == "__main__":
    res = {"laminate": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in LAM.items()}}
    res["order"] = {}
    for branch in ("A0", "S0", "SH0"):
        for scheme, cfl in (("strang", 0.8), ("yoshida", 0.9)):
            Ns = (24, 48, 96) if branch == "A0" else (12, 24, 48)
            rows = order_case(branch, scheme, cfl, Ns)
            res["order"][f"{branch}/{scheme}"] = rows
            print(branch, scheme, " | ".join(f"N={r['N']} err={r['err']:.2e}" + (f" EOC={r['eoc']:.2f}" if 'eoc' in r else "") for r in rows), flush=True)
    res["energy"] = []
    for kind in ("membrane", "bending"):
        for scheme in ("strang", "yoshida"):
            rows = energy_case(kind, scheme, (1e-3, 0.5e-3, 0.25e-3))
            res["energy"] += rows
            for r in rows:
                print("energy", r, flush=True)
    res["cfl"] = []
    for kind in ("membrane", "bending"):
        for scheme, p in (("strang", 3), ("strang", 5), ("yoshida", 5)):
            rows = cfl_case(kind, scheme, p, (0.5, 0.9, 1.0, 1.05, 1.2))
            res["cfl"] += rows
            print("cfl", kind, scheme, p, [(r["nu"], f"{r['growth']:.3g}") for r in rows], flush=True)
    json.dump(res, open(os.path.join(OUT, "e1_numerics.json"), "w"), indent=1)
    print("saved", os.path.join(OUT, "e1_numerics.json"))
