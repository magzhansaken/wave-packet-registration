#!/usr/bin/env python3
"""E4b - check of the mode-mixing hypothesis: a single stretch cannot align packets whose thermal delay slopes differ.

For every unit (split x frequency x path) of D1, from the TRAINING block only:
  windows at the ray arrivals (calibrated plate) as in br; per-window delay slopes b_k (tracked lags against the edge
  signal, least squares in T, tau = 0); energy shares p_k of the windows in the edge signal;
  best single stretch for these slopes: beta = sum p_k b_k u_k / sum p_k u_k^2 with u_k = t_k - t_c;
  residual slopes r_k = b_k - beta u_k and the heterogeneity index H = sqrt(sum p_k r_k^2 / sum p_k b_k^2);
  predicted excess error of any single stretch at distance d (Proposition 1 with the residual delays r_k d):
      e_mix(d)^2 = sum_k p_k [2 - 2 rho_w(r_k d) cos(2 pi f r_k d)].
Nothing from the test block enters the prediction. Measured counterparts (from results/e2_rows*.jsonl): oracle
stretch minus registration (str_or - br), and the strongest stretch baseline minus ours (bss_best - br_gp).
Rows: results/e4b_rows.jsonl."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core
from experiments.e2_run import SPLITS, masks, calib_s
from experiments.e4_law import rho_hann

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))


def task(split, fk):
    D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split); ch = D["channels"]; s = calib_s(split); f = fk * 1e3; tc = 2.5 / f
    edge_tr = T[tr].max() if side == "up" else T[tr].min(); d_test = np.abs(T[te] - edge_tr)
    rows = []
    for i in range(len(ch)):
        pk = rays.packet_table(ogw.XY[ch[i, 0]], ogw.XY[ch[i, 1]], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s)
        ie, Tref, delta, cen, b = core.br_laws(t, S[tr, i], T[tr], side, fk, pk, {}, lam=0.0)
        _, hw, lab = core.windows_from_packets(t, pk, fk)
        ref = S[tr, i][ie]
        p = np.array([np.sum((ref * np.clip(1 - np.abs(t - c) / hw, 0, None)) ** 2) for c in cen]); p = p / p.sum()
        u = cen - tc; beta = float(np.sum(p * b * u) / np.sum(p * u * u)); r = b - beta * u
        H = float(np.sqrt(np.sum(p * r * r) / max(np.sum(p * b * b), 1e-300)))
        emix = [float(np.sqrt(max(np.sum(p * (2 - 2 * rho_hann(np.abs(r * d), f) * np.cos(2 * np.pi * f * r * d))), 0.0))) for d in d_test]
        e_dref = float(np.sqrt(max(np.sum(p * (2 - 2 * rho_hann(np.abs(r * 15.0), f) * np.cos(2 * np.pi * f * r * 15.0))), 0.0)))
        rows.append({"split": split, "f_kHz": fk, "path": int(i), "H": H, "beta": beta, "e_mix_sig": emix, "e_mix_dref": e_dref,
                     "e_mix": float(np.mean(emix)), "share_S0": float(sum(pp for pp, l in zip(p, lab) if l == "S0")),
                     "share_A0": float(sum(pp for pp, l in zip(p, lab) if l == "A0"))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default=",".join(map(str, ogw.FREQS))); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(OUT, "e4b_rows.jsonl")); a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["split"], json.loads(l)["f_kHz"]) for l in open(a.out)}
    jobs = [(sp, int(fk)) for fk in a.freqs.split(",") for sp in SPLITS if (sp, int(fk)) not in done]
    print(f"{len(jobs)} tasks", flush=True); t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(_star, jobs):
            with open(a.out, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"[{time.time() - t0:.0f}s] {rows[0]['split']} {rows[0]['f_kHz']} kHz: median H={np.median([r['H'] for r in rows]):.3f} "
                  f"e_mix={np.median([r['e_mix'] for r in rows]):.3f}", flush=True)


def _star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return task(*j)


if __name__ == "__main__":
    main()
