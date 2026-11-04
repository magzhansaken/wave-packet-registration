#!/usr/bin/env python3
"""E8c - confirmation of finding H15 (amendment 12 in docs/PROTOCOL.md), written before any run on these data.
Ray windows against uniform windows of the wave-packet registration, on a set the finding was not searched on
(D1_SLICES points to the reserve half of D1 or to dam_D16). 12 fixed paths x 6 splits x 12 frequencies.
Variants (same definitions as E8, experiments/e8_ablation.py):
  br_cal    registration only (tau = 0, no residual), windows at ray arrivals of the calibrated plate  [primary]
  br_uni    the same with uniformly spaced windows and one pooled prior slope                     [primary]
  best_cal  br_gp with the configuration chosen on validation in E2 of this set (--rows)          [secondary]
  best_uni  that configuration with uniform windows                                               [secondary]
  bss, bss_tr references.
Rows: --out (unit = split x frequency x path)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core
from experiments.e2_run import SPLITS, masks, subset_paths, calib_s
from experiments.e8_ablation import pooled_kappa, best_cfg


def task(split, fk, rows_path):
    D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split); ch = D["channels"]; s = calib_s(split)
    pk = {"cal": [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s) for a, b in ch],
          "uni": [[] for _ in ch]}
    kap = {"cal": pooled_kappa(t, S[tr], T[tr], side, fk, pk["cal"]),
           "uni": pooled_kappa(t, S[tr], T[tr], side, fk, pk["uni"], single=True)}
    bc = best_cfg(split, fk, rows_path); rows = []
    for i in subset_paths(ch):
        Str, Ste = S[tr, i], S[te, i]; m = {}
        def ev(key, fn, **kw):
            (P, _), dt = core.timed(fn, t, Str, T[tr], T[te], side, **kw)
            e = core.rel_err(P, Ste); m[key] = {"err": float(np.mean(e)), "cost_s": float(dt)}
        for v in ("cal", "uni"):
            ev(f"br_{v}", core.fit_br, fk=fk, packets=pk[v][i], kappa=kap[v])
            ev(f"best_{v}", core.fit_br_gp, fk=fk, packets=pk[v][i], kappa=kap[v], **bc)
        ev("bss", core.fit_bss); ev("bss_tr", core.fit_bss_tr)
        rows.append({"split": split, "side": side, "f_kHz": fk, "path": int(i), "m": m})
    return rows


def _star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return task(*j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default="40,60,80,100,120,140,160,180,200,220,240,260")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["split"], json.loads(l)["f_kHz"]) for l in open(a.out)}
    jobs = [(sp, int(f), a.rows) for f in a.freqs.split(",") for sp in a.splits.split(",") if (sp, int(f)) not in done]
    print(f"{len(jobs)} tasks", flush=True); t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(_star, jobs):
            with open(a.out, "a") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
            print(f"[{time.time() - t0:.0f}s] {rows[0]['split']} {rows[0]['f_kHz']} kHz done", flush=True)


if __name__ == "__main__":
    main()
