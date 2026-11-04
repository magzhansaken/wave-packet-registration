#!/usr/bin/env python3
"""Monte-Carlo check of Proposition 2 for the intervals EXACTLY as implemented in surrogates/core.py
(conformal_calibrate + conformal_halfwidth): s(t) = time-wise std of the calibration residuals (floored), quantile of the
UNSCALED scores |r|/s(t) by np.quantile, kappa = max(b / (a + b * dbar), 0) from a least-squares line of the record RMS
against d, and half-width q s(t) (1 + kappa max(d - dbar, 0)). Same generative model and settings as prop2 in
experiments/theory_mc.py (seed 7, 2000 repetitions), so the numbers are comparable. Output: results/theory_mc_impl.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from surrogates import core

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(7)


def run(n_cal=60, n_test=60, m=200, kappa=0.08, reps=2000, d_test=(5, 15)):
    s_t = 0.5 + np.abs(np.sin(np.linspace(0, 6, m))); cov_law, cov_plain = [], []
    for _ in range(reps):
        dc = rng.uniform(0, 5, n_cal); dt = rng.uniform(*d_test, n_test); d0 = dc.mean()
        Rc = s_t * (1 + kappa * (dc - d0))[:, None] * rng.standard_t(5, (n_cal, m))
        Rt = s_t * (1 + kappa * (dt - d0))[:, None] * rng.standard_t(5, (n_test, m))
        st, qz, kap, dmean = core.conformal_calibrate(Rc, dc, np.ones((1, m)), q=0.95)
        for law, acc in ((True, cov_law), (False, cov_plain)):
            hw = core.conformal_halfwidth(st, qz, kap, dmean, dt, law=law); acc.append(float(np.mean(np.abs(Rt) <= hw)))
    f = lambda v: {"mean_coverage": float(np.mean(v)), "p05": float(np.percentile(v, 5))}
    return {"law": f(cov_law), "plain": f(cov_plain)}


if __name__ == "__main__":
    out = {"implemented": run(), "implemented_far": run(d_test=(15, 30))}
    json.dump(out, open(os.path.join(OUT, "theory_mc_impl.json"), "w"), indent=1); print(json.dumps(out))
