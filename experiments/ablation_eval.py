#!/usr/bin/env python3
"""Section 4.6: ablations. (1) On the three sets of D1, by band: gain over the strongest stretch baseline of br_gp (full),
br (no residual, tau = 0), br_gp2 (centred slope with intercept; introduced after the reserve run - post hoc); paired
effects br_gp vs br and br_gp vs br_gp2. (2) E8 on the main half (12 fixed paths, six main frequencies): ray windows of the
calibrated plate vs nominal plate vs uniform windows, for registration only (br_*) and for the chosen configuration
(best_*); prior-only configurations (tau = inf) vs the strongest stretch. (3) confirmation of H15, ray against uniform windows (amendment 12): copied from
results/e8c_eval.json when it exists. Output: results/ablation_eval.json"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A
from experiments.dam_evaluate import BANDS

R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
SETS = {"main": lambda: A.load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl"), P("a2dev_rows_main.jsonl")]) + A.load(P("e2_rows_confirm.jsonl"), [P("a2dev_rows_main.jsonl")]),
        "reserve": lambda: A.load(P("e2_rows_reserve.jsonl"), [P("a2dev_rows_reserve.jsonl")]),
        "dam_D16": lambda: A.load(P("e2_rows_dam.jsonl"), [P("a2_rows_dam.jsonl")])}


def eff(rs, ours, base):
    a = np.array([r["m"][base]["err"] for r in rs]); b = np.array([r["m"][ours]["err"] for r in rs]); d = (a - b) / a
    lo, hi = A.boot_ci(d, B=2000)
    return {"n": len(d), "median_rel_pct": 100 * float(np.median(d)), "ci_pct": [100 * lo, 100 * hi],
            "p": float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0}


def main():
    out = {"sets": {}}
    for name, f in SETS.items():
        rows = f(); A.add_best_bss([r for r in rows if "bss_best" not in r["m"]])
        rows = [r for r in rows if all(k in r["m"] for k in ("bss_best", "br_gp", "br", "br_gp2"))]
        res = {"n_units": len(rows)}
        for g, fs in BANDS.items():
            rs = [r for r in rows if r["f_kHz"] in fs]
            res[g] = {"br_gp": eff(rs, "br_gp", "bss_best"), "br": eff(rs, "br", "bss_best"), "br_gp2": eff(rs, "br_gp2", "bss_best"),
                      "residual_br_gp_vs_br": eff(rs, "br_gp", "br"), "intercept_br_gp2_vs_br_gp": eff(rs, "br_gp2", "br_gp")}
        out["sets"][name] = res
        print(name, len(rows), {g: (round(res[g]["br"]["median_rel_pct"], 1), round(res[g]["br_gp2"]["median_rel_pct"], 1)) for g in BANDS})
    e8 = [json.loads(l) for l in open(P("e8_rows.jsonl"))]
    for r in e8:
        r["m"]["bss_best_e8"] = min(r["m"]["bss"], r["m"]["bss_tr"], key=lambda x: x["err"])   # oracle min, optimistic for the baseline
    grp = {"band_60_100": [60, 100], "outside": [40, 140, 180, 220]}
    o8 = {}
    for g, fs in grp.items():
        rs = [r for r in e8 if r["f_kHz"] in fs]
        o8[g] = {"n": len(rs), "rays_vs_uniform_br": eff(rs, "br_cal", "br_uni"), "rays_vs_uniform_best": eff(rs, "best_cal", "best_uni"),
                 "calibrated_vs_nominal_br": eff(rs, "br_cal", "br_nom"), "calibrated_vs_nominal_best": eff(rs, "best_cal", "best_nom")}
        for tau in ("inf", "1000", "0"):
            ks = [k for k in rs[0]["m"] if k.startswith(f"cfg_{tau}_")]
            o8[g][f"cfg_tau_{tau}_vs_min_stretch"] = {k: eff(rs, k, "bss_best_e8")["median_rel_pct"] for k in ks}
    out["e8_main"] = o8
    if os.path.exists(P("e8c_eval.json")):
        out["h15"] = json.load(open(P("e8c_eval.json")))
    json.dump(out, open(P("ablation_eval.json"), "w"), indent=1)
    print(json.dumps(o8, indent=0)[:2500])


if __name__ == "__main__":
    main()
