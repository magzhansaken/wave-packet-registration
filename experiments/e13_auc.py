#!/usr/bin/env python3
"""E13.1 (amendment 13.1): detection of the damage of dam_D16 after temperature compensation.
Models are trained on the training block of the MAIN half of D1 (intact) in every split; they predict the healthy
baseline at the temperature of (negatives) the test signals of the main half and (positives) the dam_D16 signals whose
temperature lies in the same test block. Damage index of a signal on a path = relative L2 residual. Unit = split x
frequency x path (12 fixed paths); AUC by Mann-Whitney between positives and negatives of the unit.
Methods: br_gp (configuration chosen on validation in E2 of the main half) and the strongest stretch baseline bss_best
(variant chosen on validation in E2 of the main half). Rows: --out."""
import argparse, json, os, sys, time
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core
from experiments.e2_run import SPLITS, masks, subset_paths, calib_s, pooled_kappa
from experiments import analyze as A

MAIN = "data/d1"; DAM = "data/d1_dam"
R = os.path.join(os.path.dirname(__file__), "..", "results")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def task(split, fk, cfg_br, bss_pick):
    D = ogw.load_freq(fk, root=MAIN); Q = ogw.load_freq(fk, root=DAM)
    t = D["t"]; assert np.allclose(t, Q["t"]) and np.array_equal(D["channels"], Q["channels"])
    S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
    SD = Q["S"].astype(np.float64); TD = Q["T"]; SD -= SD.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split); _, lo, hi = SPLITS[split]
    pos = (TD >= hi) if side == "up" else (TD <= hi)
    ch = D["channels"]; s = calib_s(split)
    pk = [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s) for a, b in ch]
    kap = pooled_kappa(t, S[tr], T[tr], side, fk, pk)
    Tq = np.concatenate([T[te], TD[pos]]); nneg = int(te.sum()); rows = []
    fb = core.fit_bss if bss_pick == "bss" else core.fit_bss_tr
    for i in subset_paths(ch):
        Y = np.concatenate([S[te, i], SD[pos, i]]); m = {}
        P, _ = core.fit_br_gp(t, S[tr, i], T[tr], Tq, side, fk=fk, packets=pk[i], kappa=kap, **cfg_br)
        e = core.rel_err(P, Y); m["br_gp"] = {"auc": auc(e[nneg:], e[:nneg]), "di_neg": e[:nneg].tolist(), "di_pos": e[nneg:].tolist()}
        P, _ = fb(t, S[tr, i], T[tr], Tq, side)
        e = core.rel_err(P, Y); m["bss_best"] = {"auc": auc(e[nneg:], e[:nneg]), "di_neg": e[:nneg].tolist(), "di_pos": e[nneg:].tolist(), "chosen": bss_pick}
        rows.append({"split": split, "side": side, "f_kHz": fk, "path": int(i), "n_neg": nneg, "n_pos": int(pos.sum()),
                     "T_pos": TD[pos].tolist(), "T_neg": T[te].tolist(), "m": m})
    return rows


def _star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return task(*j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default="60,80,100,40,120,140,160,180,200,220,240,260")
    ap.add_argument("--splits", default=",".join(SPLITS)); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(R, "e13_auc_rows.jsonl"))
    a = ap.parse_args()
    rows = A.load(os.path.join(R, "e2_rows.jsonl"), [os.path.join(R, "e2_rows_bss_tr.jsonl"), os.path.join(R, "e2_rows_brgp_fix.jsonl")]) \
        + A.load(os.path.join(R, "e2_rows_confirm.jsonl"))        # the six confirmation frequencies of the main half
    A.add_best_bss([r for r in rows if "bss_best" not in r["m"]])
    cfg, pick = {}, {}
    for r in rows:
        k = (r["split"], r["f_kHz"]); c = r["cfg"]["br_gp"]
        cfg[k] = {"lam": float(c["lam"]), "wf": float(c["wf"]), "resid": c["resid"]}; pick[k] = r["m"]["bss_best"]["chosen"]
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["split"], json.loads(l)["f_kHz"]) for l in open(a.out)}
    jobs = [(sp, int(f), cfg[(sp, int(f))], pick[(sp, int(f))]) for f in a.freqs.split(",") for sp in a.splits.split(",") if (sp, int(f)) not in done]
    print(f"{len(jobs)} tasks", flush=True); t0 = time.time()
    import multiprocessing as mp
    it = mp.get_context("fork").Pool(a.workers).imap_unordered(_star, jobs) if a.workers > 1 else map(_star, jobs)
    for rs in it:
        with open(a.out, "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rs)
        d = np.nanmedian([r["m"]["br_gp"]["auc"] - r["m"]["bss_best"]["auc"] for r in rs])
        print(f"[{time.time() - t0:.0f}s] {rs[0]['split']} {rs[0]['f_kHz']} kHz n_pos={rs[0]['n_pos']} dAUC_med={d:.3f}", flush=True)


if __name__ == "__main__":
    main()
