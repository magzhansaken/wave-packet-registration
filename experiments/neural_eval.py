#!/usr/bin/env python3
"""Section 4.9: neural baselines on their 36 units (2 fixed paths x 6 splits x 3 frequencies 40/100/180 kHz) of the main
half (MF-B-NO: 3 seeds, f-VAE: 3 seeds) and of the held-out damaged set (MF-B-NO: 2 seeds, f-VAE: 3 seeds). Per unit, the
error of a neural model is its mean over seeds. Paired against br_gp (configuration chosen on validation) and against the
strongest stretch baseline bss_best: median relative difference (positive = br_gp better), unit bootstrap CI (10 000),
split-cluster CI (2000), two-sided Wilcoxon, wins. Seeds enter no p-value (protocol, Section 4.2 of the article).
Coverage of the nominal 95% bands of MF-B-NO (posterior, split conformal, growth law) per extrapolation block, and of
br_gp with the growth law on the same units. Output: results/neural_eval.json"""
import collections, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
SETS = {"main": (["e2_neural_rows.jsonl", "e2_neural_rows_seed2.jsonl"],
                 lambda: A.load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl")])),
        "dam_D16": (["e2_neural_rows_dam.jsonl"], lambda: A.load(P("e2_rows_dam.jsonl")))}
KEY = lambda r: (r["split"], r["f_kHz"], r["path"])


def paired(ours, base, clusters):
    ours, base = np.asarray(ours), np.asarray(base); d = (base - ours) / base
    return {"n": int(len(d)), "median_rel_pct": 100 * float(np.median(d)), "ci_units_pct": [100 * x for x in A.boot_ci(d)],
            "ci_splits_pct": [100 * x for x in A.boot_ci(d, clusters=clusters)], "wins_ours": int((base > ours).sum()),
            "p": float(wilcoxon(ours, base).pvalue), "median_ours": float(np.median(ours)), "median_base": float(np.median(base)),
            "mean_ours": float(np.mean(ours)), "mean_base": float(np.mean(base))}


def main():
    out = {}
    for name, (files, loader) in SETS.items():
        nr = {}
        for f in files:
            for l in open(P(f)):
                r = json.loads(l); k = KEY(r)
                if k in nr:
                    for m, v in r["m"].items():
                        nr[k]["m"][m] = nr[k]["m"].get(m, []) + v
                else:
                    nr[k] = r
        rows = loader(); A.add_best_bss([r for r in rows if "bss_best" not in r["m"]])
        base = {KEY(r): r for r in rows}
        keys = sorted(nr); cl = [k[0] for k in keys]
        ours = np.array([base[k]["m"]["br_gp"]["err"] for k in keys]); bb = np.array([base[k]["m"]["bss_best"]["err"] for k in keys])
        res = {"n_units": len(keys), "median_br_gp": float(np.median(ours)), "median_bss_best": float(np.median(bb)),
               "median_err_on_units": {m: float(np.median([base[k]["m"][m]["err"] for k in keys]))
                                       for m in ("copy", "knn", "gp", "bss", "bss_tr", "bss_gp", "bss_tr_gp", "br", "br_gp", "bss_best")
                                       if all(m in base[k]["m"] for k in keys)},
               "br_gp_vs_bss_best": paired(ours, bb, cl)}
        for m in ("mfbno_bss", "fvae"):
            e = np.array([np.mean([s["err"] for s in nr[k]["m"][m]]) for k in keys])
            res[m] = {"n_seeds": sorted({len(nr[k]["m"][m]) for k in keys}), "median": float(np.median(e)), "mean": float(np.mean(e)),
                      "seed_sd_median": float(np.median([np.std([s["err"] for s in nr[k]["m"][m]]) for k in keys])),
                      "br_gp_vs": paired(ours, e, cl), "vs_bss_best": paired(e, bb, cl),
                      "by_freq": {fk: paired(ours[[k[1] == fk for k in keys]], e[[k[1] == fk for k in keys]],
                                             [c for c, k in zip(cl, keys) if k[1] == fk]) for fk in (40, 100, 180)}}
            if m == "mfbno_bss":
                res[m]["checkpoint_steps_median"] = float(np.median([s["steps"] for k in keys for s in nr[k]["m"][m]]))
                res[m]["sec_median_logged"] = float(np.median([s["sec"] for k in keys for s in nr[k]["m"][m]]))
                for kind in ("bayes", "conf", "law"):
                    cov = collections.defaultdict(list)
                    for k in keys:
                        for s in nr[k]["m"][m]:
                            cov[k[0]] += s[f"cov_{kind}_sig"]
                    res[m][f"cov_{kind}_min_block"] = float(min(np.mean(v) for v in cov.values()))
                    res[m][f"hw_{kind}_mean"] = float(np.mean([s[f"hw_{kind}"] for k in keys for s in nr[k]["m"][m]]))
        cov = collections.defaultdict(list)
        for k in keys:
            iv = base[k].get("iv", {}).get("br_gp", {})
            if "law" in iv:
                cov[k[0]] += iv["law"]["cov_sig"]
        if cov:
            res["br_gp_cov_law_min_block"] = float(min(np.mean(v) for v in cov.values()))
        out[name] = res
        mb = res["mfbno_bss"]["br_gp_vs"]
        print(name, len(keys), "br_gp vs mfbno", round(mb["median_rel_pct"], 1), [round(x, 1) for x in mb["ci_units_pct"]],
              round(mb["p"], 3), mb["wins_ours"], "| fvae", round(res["fvae"]["br_gp_vs"]["median_rel_pct"], 1),
              "| cov mfbno bayes/conf/law", round(res["mfbno_bss"]["cov_bayes_min_block"], 3), round(res["mfbno_bss"]["cov_conf_min_block"], 3),
              round(res["mfbno_bss"]["cov_law_min_block"], 3), "| br_gp law", round(res.get("br_gp_cov_law_min_block", np.nan), 3))
    json.dump(out, open(P("neural_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
