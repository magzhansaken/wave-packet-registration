#!/usr/bin/env python3
"""E8b - does the coverage of the drift-law intervals depend on how the validation block was cut?

On the 12 fixed paths x 6 splits x 6 main frequencies, br_gp with the configuration chosen in E2 (no re-tuning) is
re-calibrated with other validation cuts: gap 0 C and 5 C between the inner part and the validation block (E2: 2.5 C),
and a validation block of 40 % of the training range (E2: 25 %). The point prediction does not change (it is fitted on
the whole training block); only the interval calibration does. Rows: results/e8b_rows.jsonl."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import experiments.e2_run as E
from bcs import ogw

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
SETTINGS = {"gap0": (0.25, 0.0), "gap5": (0.25, 5.0), "frac40": (0.40, 2.5)}


def best_cfg(split, fk):
    src = os.path.join(OUT, "e2_rows_brgp_fix.jsonl")
    for line in open(src if os.path.exists(src) else os.path.join(OUT, "e2_rows.jsonl")):
        if f'"split": "{split}"' in line and f'"f_kHz": {fk},' in line:
            c = json.loads(line)["cfg"]["br_gp"]
            return {"lam": float(c["lam"]), "wf": float(c["wf"]), "resid": c["resid"]}


def task(split, fk):
    E.ONLY[:] = ["br_gp"]; E.NO_IN = True
    E.GRIDS["br_gp"] = [best_cfg(split, fk)]
    D = ogw.load_freq(fk); sub = E.subset_paths(D["channels"]); out = []
    for name, (frac, gap) in SETTINGS.items():
        E.VAL_FRAC, E.VAL_GAP = frac, gap
        rows = E.task(split, fk, paths_only=sub)
        for r in rows:
            iv = r["iv"]["br_gp"]
            out.append({"setting": name, "split": split, "f_kHz": fk, "path": r["path"], "n_val": r["n_val"], "n_inner": r["n_inner"],
                        "cov_law": float(np.mean(iv["law"]["cov_sig"])), "hw_law": iv["law"]["hw_rel"],
                        "cov_conf": float(np.mean(iv["conf"]["cov_sig"])), "err": r["m"]["br_gp"]["err"]})
    return out


def main():
    out = os.path.join(OUT, "e8b_rows.jsonl"); done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["split"], json.loads(l)["f_kHz"]) for l in open(out)}
    jobs = [(sp, fk) for fk in ogw.MAIN for sp in E.SPLITS if (sp, fk) not in done]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("fork").Pool(int(os.environ.get("WORKERS", 2))) as pool:
        for rows in pool.imap_unordered(_star, jobs):
            with open(out, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"[{time.time() - t0:.0f}s] {rows[0]['split']} {rows[0]['f_kHz']}: " +
                  " ".join(f"{s}={np.mean([r['cov_law'] for r in rows if r['setting'] == s]):.3f}" for s in SETTINGS), flush=True)


def _star(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return task(*j)


if __name__ == "__main__":
    main()
