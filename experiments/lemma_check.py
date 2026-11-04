#!/usr/bin/env python3
"""Numerical check of Lemma 1 (energy stability of the bicharacteristic-splitting step).

Random symmetric laminates (4 random ply angles mirrored, M21/T700 plies), white-noise initial states in every field
(each field scaled to a comparable energy share), traction-free edges, both compositions (Strang, Yoshida),
interpolation degree p in {3, 5}, Courant numbers nu in {0.5, 0.9, 1.0} (inside the Lemma) and {1.1, 1.3} (outside).
Recorded: the largest one-step energy ratio E_{n+1}/E_n over 150 steps and E_150/E_0.
Also: the symbol bound max |G_p(sigma, xi)| over sigma in [0, 1], xi in [0, pi] for p = 1..7 (foot in the central cell).
Output: results/lemma_check.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs.laminate import laminate
from bcs.solver import Subsystem
from bcs import system as sy

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))


def symbol_max(p, n_s=1001, n_x=2001):
    m = (p + 1) // 2
    nodes = np.arange(-m + 1, m + 1) if p % 2 else np.arange(-(p // 2), p // 2 + 1)
    xi = np.linspace(0, np.pi, n_x); best = 0.0
    for s in np.linspace(0, 1, n_s):
        l = np.ones(len(nodes))
        for a, ja in enumerate(nodes):
            for b, jb in enumerate(nodes):
                if a != b:
                    l[a] *= (s - jb) / (ja - jb)
        best = max(best, float(np.abs(np.sum(l[:, None] * np.exp(1j * nodes[:, None] * xi[None]), 0)).max()))
    return best


def main():
    rng = np.random.default_rng(11)
    rows = []
    for trial in range(6):
        ang = rng.uniform(-90, 90, 4).round(1).tolist(); layup = ang + ang[::-1]
        lam = laminate(layup=layup)
        for kind in ("membrane", "bending"):
            E = sy.energy_matrix(lam)
            for scheme in ("strang", "yoshida"):
                for p in (3, 5):
                    for nu in (0.5, 0.9, 1.0, 1.1, 1.3):
                        S = Subsystem(lam, kind, 0.03, 0.02, 1e-3, cfl=nu, p=p, scheme=scheme)
                        d = np.sqrt(np.diag(E)[S.idx])
                        S.q[:] = rng.standard_normal(S.q.shape) / d[:, None, None]
                        E0 = S.energy(); Ep = E0; worst = 0.0
                        for _ in range(150):
                            S.step(); En = S.energy(); worst = max(worst, En / Ep); Ep = En
                        rows.append({"layup": layup, "kind": kind, "scheme": scheme, "p": p, "nu": nu,
                                     "max_step_ratio": worst, "E_end_over_E0": Ep / E0})
        print(trial, flush=True)
    inside = [r for r in rows if r["nu"] <= 1.0]; outside = [r for r in rows if r["nu"] > 1.0]
    out = {"symbol_max": {p: symbol_max(p) for p in range(1, 8)},
           "n_runs_inside": len(inside), "max_step_ratio_inside": max(r["max_step_ratio"] for r in inside),
           "n_runs_outside": len(outside), "min_E_end_over_E0_outside": min(r["E_end_over_E0"] for r in outside),
           "share_outside_growing": float(np.mean([r["E_end_over_E0"] > 1.0 for r in outside])), "rows": rows}
    json.dump(out, open(os.path.join(OUT, "lemma_check.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
