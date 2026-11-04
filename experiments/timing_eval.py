#!/usr/bin/env python3
"""Section 4.9: cost of all methods under equal load (the per-row times of E2 were
measured under unequal load and are not comparable). One process, one thread (OMP/MKL/torch = 1), methods run one after
another on the same units, with the method order rotated from unit to unit. Units: the neural units of splits S2 (up) and
S5 (down) of the main half: 2 paths x 3 frequencies x 2 splits = 12 units. Configurations: those chosen on validation in
E2 (rows of e2_rows*.jsonl) and, for MF-B-NO, e2_neural_cfg_d1.json, one seed, 600 steps with the validation checkpoint
and the refit, exactly as e2_neural.py (the tracked stretch law used as its low-fidelity input is part of its time).
Measured per unit: fit + prediction of the test block (wall and CPU time). Measured once per task (split x frequency):
the ray tables of the 66 paths and the pooled slopes, which br and br_gp share across the 66 paths of a task.
Output: results/timing_eval.json"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time, platform
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw, rays
from surrogates import core, neural
from experiments import analyze as A
from experiments.e2_run import masks, subset_paths, calib_s, pooled_kappa, run_method
from experiments.e2_neural import lf_pair

torch.set_num_threads(1)
R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
SPLITS = ("S2_le35_ge40", "S5_ge45_le40"); FREQS = (40, 100, 180)
CHEAP = ("copy", "knn", "gp", "bss", "bss_tr", "bss_gp", "bss_tr_gp", "br", "br_gp")
REP = 3                                   # cheap methods: median of 3 repetitions per unit


def clock(fn):
    w0, c0 = time.perf_counter(), time.process_time(); out = fn(); return out, time.perf_counter() - w0, time.process_time() - c0


def main():
    rows = {(r["split"], r["f_kHz"], r["path"]): r for r in
            A.load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl")])}
    ncfg = json.load(open(P("e2_neural_cfg_d1.json")))
    out = {"host": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version(), "threads": 1,
                    "torch": torch.__version__, "loadavg_start": open("/proc/loadavg").read().split()[:3]},
           "units": [], "task_prep": []}
    k = 0
    for split in SPLITS:
        for fk in FREQS:
            D = ogw.load_freq(fk); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
            side, tr, te, val, inner = masks(T, split); ch = D["channels"]; sub = subset_paths(ch)
            vi, ii = val[tr], inner[tr]

            def prep():
                s = calib_s(split)
                pk = [rays.packet_table(ogw.XY[a], ogw.XY[b], fk, modes=("S0", "A0"), order=1, tmax=t[-1], s=s) for a, b in ch]
                return pk, pooled_kappa(t, S[tr], T[tr], side, fk, pk)
            (packets, kappa), w, c = clock(prep)
            out["task_prep"].append({"split": split, "f_kHz": fk, "wall_s": w, "cpu_s": c, "n_paths": len(ch)})
            for i in sub[::6]:
                row = rows[(split, fk, int(i))]; Str, Ste = S[tr, i], S[te, i]
                cfgs = {m: {kk: (np.inf if vv == "inf" else vv) for kk, vv in row["cfg"].get(m, {}).items()} for m in CHEAP}
                ctx = {"fk": fk, "packets": packets[i], "kappa": kappa}
                u = {"split": split, "f_kHz": fk, "path": int(i), "m": {}}
                order = list(CHEAP[k % len(CHEAP):] + CHEAP[:k % len(CHEAP)]); k += 1
                for m in order:
                    ws, cs, es = [], [], []
                    for _ in range(REP):
                        (Pp, _, _), w, c = clock(lambda: run_method(m, t, Str, T[tr], T[te], side, cfgs[m], ctx))
                        ws.append(w); cs.append(c)
                    u["m"][m] = {"wall_s": float(np.median(ws)), "cpu_s": float(np.median(cs)),
                                 "err": float(np.mean(core.rel_err(Pp, Ste))), "err_e2": row["m"][m]["err"]}
                best = ncfg[str(fk)]["bss"]

                def mf():
                    LFs = lf_pair(t, Str, T[tr], T[te], side); LFi = lf_pair(t, Str[ii], T[tr][ii], T[tr][vi], side)
                    return neural.mfbno_unit(t, Str, T[tr], T[te], ii, vi, LFs["bss"][0], LFs["bss"][1], fk, seed=0,
                                             lr=best["lr"], width=best["width"], steps=600, LFin=LFi["bss"])
                (Pm, _, nst, _, _), w, c = clock(mf)
                u["m"]["mfbno"] = {"wall_s": w, "cpu_s": c, "steps": int(nst), "err": float(np.mean(core.rel_err(Pm, Ste)))}
                Pf, w, c = clock(lambda: neural.fvae_unit(Str, T[tr], T[te], seed=0))
                u["m"]["fvae"] = {"wall_s": w, "cpu_s": c, "err": float(np.mean(core.rel_err(Pf, Ste)))}
                out["units"].append(u)
                print(split, fk, i, {m: round(v["cpu_s"], 3) for m, v in u["m"].items()}, flush=True)
    ms = list(CHEAP) + ["mfbno", "fvae"]
    out["median_cpu_s"] = {m: float(np.median([u["m"][m]["cpu_s"] for u in out["units"]])) for m in ms}
    out["median_wall_s"] = {m: float(np.median([u["m"][m]["wall_s"] for u in out["units"]])) for m in ms}
    out["prep_cpu_s_per_task"] = float(np.median([p["cpu_s"] for p in out["task_prep"]]))
    out["prep_cpu_s_per_unit"] = out["prep_cpu_s_per_task"] / 66
    brg = np.array([u["m"]["br_gp"]["cpu_s"] for u in out["units"]]) + out["prep_cpu_s_per_unit"]
    mfb = np.array([u["m"]["mfbno"]["cpu_s"] for u in out["units"]])
    out["ratio_mfbno_to_brgp"] = {"median_of_unit_ratios": float(np.median(mfb / brg)),
                                  "ratio_of_medians": float(np.median(mfb) / np.median(brg)),
                                  "range": [float((mfb / brg).min()), float((mfb / brg).max())]}
    out["ratio_brgp_to_bss_tr"] = float(np.median(brg) / out["median_cpu_s"]["bss_tr"])
    out["host"]["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    json.dump(out, open(P("timing_eval.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("median_cpu_s", "prep_cpu_s_per_unit", "ratio_mfbno_to_brgp", "ratio_brgp_to_bss_tr")}, indent=1))


if __name__ == "__main__":
    main()
