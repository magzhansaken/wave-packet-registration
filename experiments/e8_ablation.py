#!/usr/bin/env python3
"""E8 - ablations of the bicharacteristic registration on the 12 fixed paths x 6 splits x 6 main frequencies of D1.

Variants (test error on the extrapolation block, same units as E2):
  cfg_<tau>_<wf>_<resid>  every one of the 12 configurations of the br_gp grid, used fixed (no tuning);
  br_cal       registration only, data-fitted window laws (tau = 0), windows at ray arrivals with the CALIBRATED plate;
  br_nom       the same with ray arrivals of the NOMINAL plate (s = 1, no temperature law);
  br_uni       the same with uniformly spaced windows (no ray tables, burst spacing);
  best_cal     br_gp with the configuration chosen on validation in E2 (read from results/e2_rows.jsonl);
  best_nom     that configuration with nominal-plate rays (prior slopes re-pooled with nominal labels);
  best_uni     that configuration with uniform windows and one pooled prior slope for all windows;
  bss, bss_tr  references.
Besides the relative L2 error (mean over the test signals), each variant stores two alternative metrics for the
independence check of the main conclusion: the relative L2 error of the Hilbert envelopes (amplitude only) and
1 - zero-lag correlation coefficient (shape only), and the median instead of the mean over the test signals.
Rows: results/e8_rows.jsonl (unit = split x frequency x path)."""
import argparse, json, os, sys, time
import numpy as np
from scipy.signal import hilbert
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core
from experiments.e2_run import SPLITS, GRIDS, masks, subset_paths, calib_s

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))


def pooled_kappa(t, S, T, side, fk, packets, single=False):
    tc = 2.5 / (fk * 1e3); rel = {"pre": [], "S0": [], "A0": []}
    for i in range(S.shape[1]):
        ie, Tref, delta, cen, slopes = core.br_laws(t, S[:, i], T, side, fk, packets[i], {}, lam=0.0)
        _, _, labs = core.windows_from_packets(t, packets[i], fk)
        for c, sl, lb in zip(cen, slopes, labs):
            if c - tc > 50e-6:
                rel["pre" if single else lb].append(sl / (c - tc))
    k = {kk: float(np.median(v)) for kk, v in rel.items() if len(v)}
    if single:
        k = {"pre": k["pre"], "S0": k["pre"], "A0": k["pre"]}
    return k


def best_cfg(split, fk, rows_path):
    for line in open(rows_path):
        if f'"split": "{split}"' in line and f'"f_kHz": {fk},' in line:
            c = json.loads(line)["cfg"]["br_gp"]
            return {"lam": float(c["lam"]), "wf": float(c["wf"]), "resid": c["resid"]}
    raise KeyError((split, fk))


def task(split, fk, rows_path):
    D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split); ch = D["channels"]; s = calib_s(split)
    pk = {"cal": [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s) for a, b in ch],
          "nom": [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=1.0, alpha=0.0) for a, b in ch],
          "uni": [[] for _ in ch]}
    kap = {"cal": pooled_kappa(t, S[tr], T[tr], side, fk, pk["cal"]),
           "nom": pooled_kappa(t, S[tr], T[tr], side, fk, pk["nom"]),
           "uni": pooled_kappa(t, S[tr], T[tr], side, fk, pk["uni"], single=True)}
    bc = best_cfg(split, fk, rows_path)
    rows = []
    for i in subset_paths(ch):
        Str, Ste = S[tr, i], S[te, i]; m = {}
        Eq = np.abs(hilbert(Ste, axis=1))
        def ev(key, fn, **kw):
            (P, _), dt = core.timed(fn, t, Str, T[tr], T[te], side, **kw)
            e = core.rel_err(P, Ste)
            env = np.linalg.norm(np.abs(hilbert(P, axis=1)) - Eq, axis=1) / np.linalg.norm(Eq, axis=1)   # alternative metric 1
            cc = 1 - np.sum(P * Ste, 1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Ste, axis=1) + 1e-300)  # alternative metric 2
            m[key] = {"err": float(np.mean(e)), "err_sig": [float(x) for x in e], "err_env": float(np.mean(env)),
                      "err_cc": float(np.mean(cc)), "err_median": float(np.median(e)), "cost_s": float(dt)}
        for cfg in GRIDS["br_gp"]:
            tau = "inf" if np.isinf(cfg["lam"]) else f"{cfg['lam']:g}"
            ev(f"cfg_{tau}_{cfg['wf']:g}_{cfg['resid']}", core.fit_br_gp, fk=fk, packets=pk["cal"][i], kappa=kap["cal"], **cfg)
        for v in ("cal", "nom", "uni"):
            ev(f"br_{v}", core.fit_br, fk=fk, packets=pk[v][i], kappa=kap[v])
            ev(f"best_{v}", core.fit_br_gp, fk=fk, packets=pk[v][i], kappa=kap[v], **bc)
        ev("bss", core.fit_bss); ev("bss_tr", core.fit_bss_tr)
        rows.append({"split": split, "side": side, "f_kHz": fk, "path": int(i), "L_m": float(ogw.path_geometry(ch)[0][i]),
                     "best_cfg": {k: (str(v) if isinstance(v, float) and np.isinf(v) else v) for k, v in bc.items()},
                     "kappa": kap, "m": m})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default=",".join(map(str, ogw.MAIN)))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rows", default=os.path.join(OUT, "e2_rows.jsonl"))
    ap.add_argument("--out", default=os.path.join(OUT, "e8_rows.jsonl"))
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["split"], r["f_kHz"]))
    jobs = [(sp, int(f), a.rows) for f in a.freqs.split(",") for sp in a.splits.split(",") if (sp, int(f)) not in done]
    print(f"{len(jobs)} tasks", flush=True); t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(_star, jobs):
            with open(a.out, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            e = {k: np.median([x["m"][k]["err"] for x in rows]) for k in ("bss", "bss_tr", "br_cal", "br_nom", "br_uni", "best_cal", "best_nom", "best_uni")}
            print(f"[{time.time() - t0:.0f}s] {rows[0]['split']} {rows[0]['f_kHz']} kHz: " + " ".join(f"{k}={v:.3f}" for k, v in e.items()), flush=True)


def _star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return task(*j)


if __name__ == "__main__":
    main()
