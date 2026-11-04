#!/usr/bin/env python3
"""Monte-Carlo checks of the two analytic statements.

Proposition 1 (phase gap). For a tone burst s(t) = w(t) sin(2 pi f t) with a smooth window w of N cycles and a copy
shifted by Delta, the relative L2 error is
    e(Delta)^2 = 2 - 2 rho_w(Delta) cos(2 pi f Delta) + O(1/N),     rho_w(Delta) = int w(t) w(t-Delta) dt / int w^2 dt.
For a signal made of packets k with energy shares p_k and delays Delta_k = kappa (t_k - t_c) dT (unmodelled drift),
    e^2 = sum_k p_k [2 - 2 rho_w(Delta_k) cos(2 pi f Delta_k)]   (cross terms vanish when packets do not overlap).
Check: exact numerical errors against the formula for N in {3, 5, 10}, f in {40, 100, 220} kHz, random Delta.

Proposition 2 (coverage with a drift law). If the residual at time t and distance d from the data edge is
    r = s(t) (1 + kappa (d - d0)) Z,   Z exchangeable between calibration and test,
then the split-conformal half-width q_(1-a) s(t) (1 + kappa (d - d0)) with the finite-sample quantile of |Z| on the
calibration block covers r with probability >= 1 - a for every test distance d. With kappa estimated on the calibration
block by least squares the guarantee is approximate; the check below measures the coverage loss.
Output: results/theory_mc.json
"""
import json, os
import numpy as np

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
rng = np.random.default_rng(7)


def burst(t, f, N):
    """Hann-windowed tone burst of N cycles at f, starting at t = 0."""
    T = N / f
    return np.where((t >= 0) & (t <= T), np.sin(2 * np.pi * f * t) * 0.5 * (1 - np.cos(2 * np.pi * f * t / N)), 0.0)


def rho_w(t, f, N, D):
    """Normalised autocorrelation of the Hann window at lag D."""
    w = lambda x: np.where((x >= 0) & (x <= N / f), 0.5 * (1 - np.cos(2 * np.pi * f * x / N)), 0.0)
    return np.sum(w(t) * w(t - D)) / np.sum(w(t) ** 2)


def prop1():
    """Monte-Carlo check of Proposition 1 (relative error of a delayed burst)."""
    rows = []
    for N in (3, 5, 10):
        for fk in (40, 100, 220):
            f = fk * 1e3; fs = 200 * f; t = np.arange(-2 * N / f, 4 * N / f, 1 / fs)
            s = burst(t, f, N)
            for D in rng.uniform(0, 1.5 / f, 40):
                sh = burst(t - D, f, N)
                e_num = np.linalg.norm(sh - s) / np.linalg.norm(s)
                e_th = np.sqrt(max(0.0, 2 - 2 * rho_w(t, f, N, D) * np.cos(2 * np.pi * f * D)))
                rows.append((N, fk, D * f, e_num, e_th))
    r = np.array(rows)
    out = {}
    for N in (3, 5, 10):
        m = r[:, 0] == N
        out[f"N={N}"] = {"max_abs_dev": float(np.max(np.abs(r[m, 3] - r[m, 4]))), "mean_abs_dev": float(np.mean(np.abs(r[m, 3] - r[m, 4])))}
    # multi-packet signal with non-overlapping packets and delays proportional to travel time
    f = 100e3; N = 5; fs = 100 * f; t = np.arange(0, 1.2e-3, 1 / fs)
    tk = np.array([1.5e-4, 3.0e-4, 4.5e-4, 6.5e-4, 9.0e-4]); ak = rng.uniform(0.3, 1.0, len(tk)); kappa, dT = 7e-4, 8.0
    s = sum(a * burst(t - x, f, N) for a, x in zip(ak, tk)); Dk = kappa * tk * dT
    sh = sum(a * burst(t - x - d, f, N) for a, x, d in zip(ak, tk, Dk))
    p = ak ** 2 / np.sum(ak ** 2)
    e_th = np.sqrt(np.sum(p * (2 - 2 * np.array([rho_w(t, f, N, d) for d in Dk]) * np.cos(2 * np.pi * f * Dk))))
    out["multi_packet"] = {"e_numeric": float(np.linalg.norm(sh - s) / np.linalg.norm(s)), "e_formula": float(e_th)}
    return out


def prop2(n_cal=60, n_test=60, m=200, alpha=0.05, kappa=0.08, reps=2000):
    """Monte-Carlo check of Proposition 2 (coverage of the growth-law intervals)."""
    res = {"known_kappa": [], "estimated_kappa": [], "plain": []}
    s_t = 0.5 + np.abs(np.sin(np.linspace(0, 6, m)))
    for _ in range(reps):
        dc = rng.uniform(0, 5, n_cal); dt = rng.uniform(5, 15, n_test); d0 = dc.mean()
        Zc = rng.standard_t(5, (n_cal, m)); Zt = rng.standard_t(5, (n_test, m))
        Rc = s_t * (1 + kappa * (dc - d0))[:, None] * Zc; Rt = s_t * (1 + kappa * (dt - d0))[:, None] * Zt
        k = int(np.ceil((1 - alpha) * (n_cal * m + 1)))
        def q_of(scale_c):
            """Conformal quantile of the scores scaled by scale_c."""
            z = np.sort((np.abs(Rc) / (s_t * scale_c[:, None])).ravel()); return z[min(k, len(z)) - 1]
        q = q_of(1 + kappa * (dc - d0)); res["known_kappa"].append(np.mean(np.abs(Rt) <= q * s_t * (1 + kappa * (dt - d0))[:, None]))
        rms = np.sqrt((Rc ** 2).mean(1)); b, a = np.polyfit(dc, rms, 1); kh = max(b / max(a + b * d0, 1e-12), 0)
        q = q_of(1 + kh * (dc - d0)); res["estimated_kappa"].append(np.mean(np.abs(Rt) <= q * s_t * (1 + kh * np.maximum(dt - d0, 0))[:, None]))
        q = q_of(np.ones(n_cal)); res["plain"].append(np.mean(np.abs(Rt) <= q * s_t))
    return {k: {"mean_coverage": float(np.mean(v)), "p05": float(np.percentile(v, 5))} for k, v in res.items()}


if __name__ == "__main__":
    out = {"prop1_phase_gap": prop1(), "prop2_coverage": prop2()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "theory_mc.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
