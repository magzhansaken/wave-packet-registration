#!/usr/bin/env python3
"""E2/E3 neural baselines, reduced protocol (amendments 7 and 9 in docs/PROTOCOL.md: 2 CPU cores, about 150 s per 600-step run
under load). Amendment 9: 2 of the 12 fixed paths (every sixth) x 6 splits x 3 frequencies = 36 units; MF-B-NO with the
tracked-BSS LF only (2 seeds); tuning on one path per frequency. Earlier text of amendment 7 for reference:
4 of the 12 fixed paths (every third) x 6 splits x 3 frequencies = 72 units; MF-B-NO with the tracked-BSS LF: 2 seeds,
with the nominal (edge-copy) LF: 1 seed; f-VAE with the authors' hyperparameters: 3 seeds. Same splits, validation
blocks and metrics as e2_run.py; every MF-B-NO run keeps the checkpoint with the lowest validation loss.
MF-B-NO tuning: grid {lr 1e-3, 3e-4} x {width 64, 128}, once per frequency on split S2 (2 of the 12 paths, seed 0,
validation loss); the chosen configuration is used for all splits, paths and seeds -> results/e2_neural_cfg.json.
Rows: results/e2_neural_rows.jsonl."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw
from surrogates import core, neural
from experiments.e2_run import SPLITS, masks, subset_paths

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
GRID = [{"lr": lr, "width": w} for lr in (1e-3, 3e-4) for w in (64, 128)]
SEEDS = (0, 1, 2)
SEEDS_BSS = tuple(int(x) for x in os.environ.get("NEURAL_SEEDS", "0,1").split(","))   # seed 2 added in a later run
SKIP_FVAE = os.environ.get("NEURAL_SKIP_FVAE", "") == "1"
LF_KINDS = ("bss",)          # amendment 9: the nominal-LF variant is dropped
STEPS = int(os.environ.get("STEPS", 600))
NEURAL_FREQS = "40,100,180"


def lf_pair(t, Str, Ttr, Tq, side):
    """Low-fidelity inputs of MF-B-NO at the training and query conditions (nominal and stretched)."""
    sref, Tref, kf, gf = core.bss_laws_tracked(t, Str, Ttr, side)      # the stronger (tracked) BSS law as LF
    nom = lambda X: np.repeat(sref[None], len(X), 0)
    bss = lambda X: np.array([gf(x) * core.warp(t, sref, kf(x)) for x in X])
    return {"nom": (nom(Ttr), nom(Tq)), "bss": (bss(Ttr), bss(Tq))}


CFG_PATH = os.path.join(OUT, "e2_neural_cfg.json")


def tune(fk, split="S2_le35_ge40"):
    """Choose the configuration of MF-B-NO for one frequency on the validation block of one split."""
    torch.set_num_threads(1)
    D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split); sub = subset_paths(D["channels"]); vi, ii = val[tr], inner[tr]
    best, scores = {}, {}
    for lf in LF_KINDS:
        sc = []
        for cfg in GRID:
            v = []
            for i in (sub[6],):                                     # amendment 9: one path per frequency
                Str = S[tr, i]; LF = lf_pair(t, Str[ii], T[tr][ii], T[tr][vi], side)[lf]
                q = neural.dec_factor(fk)
                _, curve = neural.train_bdon(T[tr][ii], neural.down(LF[0], q), neural.down(Str[ii] - LF[0], q), STEPS,
                                             cfg["lr"], cfg["width"], 0, eval_every=60,
                                             val=(T[tr][vi], neural.down(LF[1], q), neural.down(Str[vi] - LF[1], q)))
                v.append(min(c[1] for c in curve))
            sc.append(float(np.median(v)))
        best[lf] = GRID[int(np.argmin(sc))]; scores[lf] = sc
    return fk, best, scores


def task(split, fk):
    """Neural baselines on the units of one split and frequency; returns the unit rows."""
    torch.set_num_threads(1)
    D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
    side, tr, te, val, inner = masks(T, split); sub = subset_paths(D["channels"])
    vi, ii = val[tr], inner[tr]
    edge_tr = T[tr].max() if side == "up" else T[tr].min(); edge_in = T[inner].max() if side == "up" else T[inner].min()
    d_test = np.abs(T[te] - edge_tr); d_cal = np.abs(T[val] - edge_in)
    best = json.load(open(CFG_PATH))[str(fk)]
    rows = []
    for i in sub[::6]:                                          # amendment 9: 2 of the 12 fixed paths
        Str, Ste = S[tr, i], S[te, i]; LFs = lf_pair(t, Str, T[tr], T[te], side)
        LFi = lf_pair(t, Str[ii], T[tr][ii], T[tr][vi], side)          # validation stage: law from the inner part only
        row = {"split": split, "side": side, "f_kHz": fk, "path": int(i), "cfg": best, "dT_test": [float(x) for x in d_test], "m": {}}
        rms = np.sqrt((Ste ** 2).mean(1))[:, None]
        for lf in LF_KINDS:
            per = []
            for sd_ in (SEEDS_BSS if lf == "bss" else SEEDS[:1]):
                t0 = time.time()
                P, sd, nsteps, curve, Pv = neural.mfbno_unit(t, Str, T[tr], T[te], ii, vi, LFs[lf][0], LFs[lf][1], fk, seed=sd_,
                                                             lr=best[lf]["lr"], width=best[lf]["width"], steps=STEPS, LFin=LFi[lf])
                e = core.rel_err(P, Ste)
                hwb = 1.96 * sd
                s_t, qz, kap, dmean = core.conformal_calibrate(S[val, i] - Pv, d_cal, S[val, i])
                hwl = core.conformal_halfwidth(s_t, qz, kap, dmean, d_test, law=True)
                hwc = core.conformal_halfwidth(s_t, qz, kap, dmean, d_test, law=False)
                per.append({"seed": sd_, "err": float(np.mean(e)), "err_sig": [float(x) for x in e], "steps": int(nsteps),
                            "cov_bayes_sig": [float(x) for x in (np.abs(Ste - P) <= hwb).mean(1)], "hw_bayes": float(np.mean(hwb / rms)),
                            "cov_conf_sig": [float(x) for x in (np.abs(Ste - P) <= hwc).mean(1)], "hw_conf": float(np.mean(hwc / rms)),
                            "cov_law_sig": [float(x) for x in (np.abs(Ste - P) <= hwl).mean(1)], "hw_law": float(np.mean(hwl / rms)),
                            "sec": round(time.time() - t0, 1)})
            row["m"][f"mfbno_{lf}"] = per
        per = []
        for sd_ in (() if SKIP_FVAE else SEEDS):
            t0 = time.time(); P = neural.fvae_unit(Str, T[tr], T[te], seed=sd_); e = core.rel_err(P, Ste)
            per.append({"seed": sd_, "err": float(np.mean(e)), "err_sig": [float(x) for x in e], "sec": round(time.time() - t0, 1)})
        if per:
            row["m"]["fvae"] = per
        rows.append(row)
    return rows


def main():
    """Run all (split, frequency) tasks not yet in the output file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default=NEURAL_FREQS); ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default=os.path.join(OUT, "e2_neural_rows.jsonl"))
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["split"], json.loads(l)["f_kHz"]) for l in open(a.out)}
    import multiprocessing as mp
    freqs = [int(f) for f in a.freqs.split(",")]
    cfg = json.load(open(CFG_PATH)) if os.path.exists(CFG_PATH) else {}
    need = [f for f in freqs if str(f) not in cfg]
    if need:
        t0 = time.time()
        with mp.get_context("fork").Pool(min(a.workers, len(need))) as pool:
            for fk, best, scores in pool.imap_unordered(tune, need):
                cfg[str(fk)] = best; cfg[f"{fk}_scores"] = scores
                json.dump(cfg, open(CFG_PATH, "w"), indent=1)
                print(f"[{time.time() - t0:.0f}s] tuned {fk} kHz: {best}", flush=True)
    jobs = [(sp, int(f)) for f in a.freqs.split(",") for sp in a.splits.split(",") if (sp, int(f)) not in done]
    print(f"{len(jobs)} tasks", flush=True); t0 = time.time()
    import multiprocessing as mp
    it = mp.get_context("fork").Pool(a.workers).imap_unordered(_star, jobs) if a.workers > 1 else map(_star, jobs)
    for rows in it:
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        e = {k: np.mean([np.mean([s["err"] for s in r["m"][k]]) for r in rows]) for k in rows[0]["m"]}
        print(f"[{time.time() - t0:.0f}s] {rows[0]['split']} {rows[0]['f_kHz']}: " + " ".join(f"{k}={v:.3f}" for k, v in e.items()), flush=True)


def _star(j):
    """Unpack the arguments of task for Pool.imap."""
    return task(*j)


if __name__ == "__main__":
    main()
