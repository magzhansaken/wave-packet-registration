#!/usr/bin/env python3
"""E4 - phase-gap law on all 12 frequencies of D1 (run after the findings were fixed on the 6 main frequencies).

For every unit (split x frequency x path) the edge-copy predictor (the edge training signal reused as is) is compared
with a PARAMETER-FREE prediction of its error built only from the training block:
  delay law  Delta(t; dT) = c t dT, c - median over training signals of the least-squares slope of window delays against
             t (6 windows, tracked cross-correlation), regressed on dT;  gain law g(dT) - linear in dT;
  error      e^2 = sum_t E(t) [1 + g^2 - 2 g rho_w(Delta(t)) cos(2 pi f Delta(t))] / (g^2 sum_t E(t))   (Proposition 1),
             E(t) - energy envelope of the edge signal, rho_w - autocorrelation of the 5-cycle Hann window.
Output: results/e4_law_rows.jsonl (per unit: measured and predicted error for every test signal, phase gap phi)."""
import json, os, sys, time
import numpy as np
from scipy.signal import hilbert
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw
from surrogates import core
from experiments.e2_run import SPLITS, masks

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))


def rho_hann(D, f, N=5):
    T = N / f; x = np.linspace(0, T, 400)
    w = lambda u: np.where((u >= 0) & (u <= T), 0.5 * (1 - np.cos(2 * np.pi * u / T)), 0.0)
    den = np.sum(w(x) ** 2)
    return np.array([np.sum(w(x) * w(x - d)) / den for d in np.atleast_1d(D)])


def unit(t, Str, Ttr, Ste, Tte, side, fk):
    f = fk * 1e3; ie = core.edge_index(Ttr, side); ref = Str[ie]; Tref = Ttr[ie]
    cen = np.linspace(t[0] + 0.08 * (t[-1] - t[0]), t[-1] - 0.08 * (t[-1] - t[0]), 6); hw = max(5 / f, 30e-6)
    order = np.argsort(-Ttr) if side == "up" else np.argsort(Ttr)
    L = np.array([core.track_lags(t, Str, order, ie, c - hw, c + hw, f) for c in cen]).T      # (n, 6)
    cs = (L @ cen) / (cen @ cen)                                                              # delay = c_i * t
    x = Ttr - Tref
    c_slope = float(np.sum(x * cs) / (np.sum(x * x) + 1e-12))
    g = (Str @ ref) / (ref @ ref); gl = np.polyfit(x, g, 1)
    E = np.abs(hilbert(ref)) ** 2
    rows = []
    for s, T in zip(Ste, Tte):
        dT = T - Tref; D = c_slope * t * dT; gg = float(np.polyval(gl, dT))
        rho = np.interp(np.abs(D), np.linspace(0, 5 / f, 200), rho_hann(np.linspace(0, 5 / f, 200), f))
        e2 = np.sum(E * (1 + gg ** 2 - 2 * gg * rho * np.cos(2 * np.pi * f * D))) / (gg ** 2 * np.sum(E))
        e_meas = float(np.linalg.norm(ref - s) / np.linalg.norm(s))
        tE = float(np.sum(E * t) / np.sum(E))
        rows.append((float(dT), e_meas, float(np.sqrt(max(e2, 0))), float(2 * np.pi * f * c_slope * tE * dT)))
    return c_slope, rows


def main():
    out = os.path.join(OUT, "e4_law_rows.jsonl"); t0 = time.time()
    done = {json.loads(l)["f_kHz"] for l in open(out)} if os.path.exists(out) else set()     # resumable per frequency
    if done:                                                   # drop a partly written last frequency
        cnt = {}
        for l in open(out):
            f = json.loads(l)["f_kHz"]; cnt[f] = cnt.get(f, 0) + 1
        keep = [l for l in open(out) if cnt[json.loads(l)["f_kHz"]] == len(SPLITS) * 66]
        open(out, "w").writelines(keep); done = {json.loads(l)["f_kHz"] for l in keep}
    fh = open(out, "a")
    for fk in [f for f in ogw.FREQS if f not in done]:
        D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); S -= S.mean(-1, keepdims=True); T = D["T"]
        for sp in SPLITS:
            side, tr, te, val, inner = masks(T, sp)
            for i in range(S.shape[1]):
                c, rows = unit(t, S[tr, i], T[tr], S[te, i], T[te], side, fk)
                fh.write(json.dumps({"f_kHz": fk, "split": sp, "path": i, "c_per_C": c, "rows": rows}) + "\n")
        fh.flush(); print(f"[{time.time() - t0:.0f}s] {fk} kHz done", flush=True)


if __name__ == "__main__":
    main()
