#!/usr/bin/env python3
"""Verdict of amendment 12 (H15): ray vs uniform windows in the mode-mixing band, reserve half of D1 and dam_D16.
Primary: br_cal vs br_uni, band 60/80/100 kHz; effect = median (err_uni - err_cal)/err_uni; Wilcoxon two-sided;
Holm over the two sets; confirmed iff on BOTH sets p_holm < 0.05 and the unit bootstrap CI > 0. Output results/e8c_eval.json."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A
R = os.path.join(os.path.dirname(__file__), "..", "results")
SETS = {"reserve": "e8c_rows_reserve.jsonl", "dam_D16": "e8c_rows_dam.jsonl"}
BAND = [60, 80, 100]


def eff(rs, a_key, b_key):
    a = np.array([r["m"][a_key]["err"] for r in rs]); b = np.array([r["m"][b_key]["err"] for r in rs]); d = (b - a) / b
    lo, hi = A.boot_ci(d); clo, chi = A.boot_ci(d, clusters=[r["split"] for r in rs])
    return {"n": len(d), "median_rel_pct": 100 * float(np.median(d)), "ci_pct": [100 * lo, 100 * hi],
            "ci_cluster_split_pct": [100 * clo, 100 * chi], "p": float(wilcoxon(a, b).pvalue), "n_better": int((d > 0).sum())}


def main():
    rep = {}
    for k, f in SETS.items():
        rows = [json.loads(l) for l in open(os.path.join(R, f))]
        assert len(rows) == 6 * 12 * 12, (k, len(rows))
        band = [r for r in rows if r["f_kHz"] in BAND]; out = [r for r in rows if r["f_kHz"] not in BAND]
        rep[k] = {"primary_band": eff(band, "br_cal", "br_uni"), "out_of_band": eff(out, "br_cal", "br_uni"),
                  "secondary_best_band": eff(band, "best_cal", "best_uni"),
                  "per_freq": {f: eff([r for r in rows if r["f_kHz"] == f], "br_cal", "br_uni")["median_rel_pct"] for f in range(40, 261, 20)},
                  "vs_bss_band": {v: 100 * float(np.median([(min(r["m"]["bss"]["err"], r["m"]["bss_tr"]["err"]) - r["m"][v]["err"]) / min(r["m"]["bss"]["err"], r["m"]["bss_tr"]["err"]) for r in band])) for v in ("br_cal", "br_uni", "best_cal", "best_uni")}}
    ph = A.holm([rep[k]["primary_band"]["p"] for k in SETS])
    for k, p in zip(SETS, ph):
        rep[k]["primary_band"]["p_holm"] = float(p)
        rep[k]["primary_band"]["win"] = bool(p < 0.05 and rep[k]["primary_band"]["ci_pct"][0] > 0)
    rep["verdict"] = "confirmed" if all(rep[k]["primary_band"]["win"] for k in SETS) else "not_confirmed"
    json.dump(rep, open(os.path.join(R, "e8c_eval.json"), "w"), indent=1)
    print("H15:", rep["verdict"], {k: rep[k]["primary_band"] for k in SETS})


if __name__ == "__main__":
    main()
