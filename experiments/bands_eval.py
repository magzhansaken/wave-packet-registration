#!/usr/bin/env python3
"""Section 4.4: br_gp against the strongest stretch baseline (bss_best) by frequency band on the three sets of D1, with the
same family, test and effect as amendment 11 (dam_evaluate.py): bands 40 / 60-100 / 120-260 kHz, two-sided Wilcoxon over
units, Holm over the three bands, median relative difference with a unit bootstrap CI (10 000) and a split-cluster CI.
Also per frequency: median gain and its unit bootstrap CI (for Figure 4), and the median error of bss_best and br_gp.
Output: results/bands_eval.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A
from experiments.dam_evaluate import family, BANDS, ALL, rel

R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
SETS = {"main": lambda: A.load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl")]) + A.load(P("e2_rows_confirm.jsonl")),
        "reserve": lambda: A.load(P("e2_rows_reserve.jsonl")),
        "dam_D16": lambda: A.load(P("e2_rows_dam.jsonl"))}


def main():
    out = {}
    for name, f in SETS.items():
        rows = f()
        A.add_best_bss([r for r in rows if "bss_best" not in r["m"]])
        rows = [r for r in rows if "bss_best" in r["m"] and "br_gp" in r["m"]]
        fam = family(rows, "br_gp", BANDS)
        pf = {}
        for fk in ALL:
            rs = [r for r in rows if r["f_kHz"] == fk]
            a, b, d = rel(rs, "br_gp")
            lo, hi = A.boot_ci(d, B=2000)
            pf[fk] = {"n": len(d), "median_rel_pct": 100 * float(np.median(d)), "ci_pct": [100 * lo, 100 * hi],
                      "median_err_bss_best": float(np.median(a)), "median_err_br_gp": float(np.median(b))}
            for o in ("str_or", "dtw_or"):
                if rs and all(o in r["m"] for r in rs):
                    pf[fk][o + "_median_rel_pct"] = 100 * float(np.median(rel(rs, o)[2]))
        out[name] = {"n_units": len(rows), "bands": fam, "per_freq": pf,
                     "chosen_bss_tr_share": float(np.mean([r["m"]["bss_best"].get("chosen") == "bss_tr" for r in rows]))}
        print(name, len(rows), {g: (round(v["median_rel_pct"], 1), round(v["p_holm"], 6), v["win"]) for g, v in fam.items()})
    json.dump(out, open(P("bands_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
