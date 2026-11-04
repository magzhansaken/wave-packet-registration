#!/usr/bin/env python3
"""Section 5: the mode-mixing index (Equation emix, training block only; rows of e4b on the main and reserve halves)
against the measured gain of br_gp over bss_best by frequency and by unit; where the registration still loses
(oracles, in-range error against GP). Inputs: e4b_rows*.jsonl, bands_eval.json, E2 rows. Output: results/discussion_eval.json"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
KEY = lambda r: (r["split"], r["f_kHz"], r["path"])
SETS = {"main": ("e4b_rows.jsonl", lambda: A.load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl")]) + A.load(P("e2_rows_confirm.jsonl"))),
        "reserve": ("e4b_rows_reserve.jsonl", lambda: A.load(P("e2_rows_reserve.jsonl")))}
BAND = (60, 80, 100)


def main():
    be = json.load(open(P("bands_eval.json"))); out = {}
    for name, (f4, loader) in SETS.items():
        em = {KEY(r): float(np.mean(r["e_mix_sig"])) for r in (json.loads(l) for l in open(P(f4)))}
        rows = loader(); A.add_best_bss([r for r in rows if "bss_best" not in r["m"]])
        rows = [r for r in rows if "bss_best" in r["m"] and "br_gp" in r["m"] and KEY(r) in em]
        g = np.array([(r["m"]["bss_best"]["err"] - r["m"]["br_gp"]["err"]) / r["m"]["bss_best"]["err"] for r in rows])
        x = np.array([em[KEY(r)] for r in rows]); fk = np.array([r["f_kHz"] for r in rows])
        freqs = sorted(set(fk.tolist()))
        pf = {int(f): {"median_emix": float(np.median(x[fk == f])), "gain_pct": be[name]["per_freq"][str(f)]["median_rel_pct"],
                       "str_or_pct": be[name]["per_freq"][str(f)].get("str_or_median_rel_pct"),
                       "dtw_or_pct": be[name]["per_freq"][str(f)].get("dtw_or_median_rel_pct")} for f in freqs}
        top3 = sorted(pf, key=lambda f: -pf[f]["median_emix"])[:3]
        inb = np.isin(fk, BAND)
        ein = {m: float(np.median([r["m"][m]["e_in"] for r in rows if "e_in" in r["m"][m]])) for m in ("gp", "br_gp", "bss_best_gp")
               if all(m in r["m"] for r in rows[:5])}
        out[name] = {"n_units": len(rows), "per_freq": pf, "top3_emix_freqs": sorted(top3),
                     "spearman_freq": float(spearmanr([pf[f]["median_emix"] for f in freqs], [pf[f]["gain_pct"] for f in freqs])[0]),
                     "spearman_units_all": float(spearmanr(x, g)[0]),
                     "spearman_units_band": float(spearmanr(x[inb], g[inb])[0]),
                     "spearman_units_out": float(spearmanr(x[~inb], g[~inb])[0]),
                     "emix_min_band": min(pf[f]["median_emix"] for f in BAND), "emix_max_out": max(pf[f]["median_emix"] for f in freqs if f not in BAND),
                     "median_e_in": ein}
        print(name, len(rows), out[name]["top3_emix_freqs"], round(out[name]["spearman_freq"], 2), round(out[name]["spearman_units_all"], 2),
              round(out[name]["spearman_units_band"], 2), round(out[name]["spearman_units_out"], 2), ein)
        print({f: (round(v["median_emix"], 2), round(v["gain_pct"], 1), v["str_or_pct"] and round(v["str_or_pct"], 1)) for f, v in pf.items()})
    b = json.load(open(P("b_eval_reserve.json")))["S1"]              # rule B (amendment 10), reserve half
    out["rule_b"] = {"rho_phys": b["rho_phys"], "rho_train": b["rho_train_dref"], "pass": b["S1_pass"]}
    json.dump(out, open(P("discussion_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
