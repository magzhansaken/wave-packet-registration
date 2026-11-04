#!/usr/bin/env python3
"""Section 4.10: transfer to fatigue drift (D3, NASA CFRP coupons; rows of e6_d3.py, unit = coupon x frequency x path).
br_gp against the baselines of D3 (copy, kNN, GP, stretch, stretch + GP residual; the tracked stretch was not run on D3):
median relative difference (positive = br_gp better), unit bootstrap CI (10 000), coupon-cluster CI (2000), two-sided
Wilcoxon, Holm over the five baselines. The same for br (registration without residual). Per frequency: gain over the
stronger stretch. Oracles. Coverage of the 95 % bands per coupon (growth law / split conformal / posterior), half-widths,
share of units in which the estimated growth slope is zero. Selection rules (findings.json, d3.selection_group).
Output: results/d3_eval.json"""
import collections, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
BASES = ("copy", "knn", "gp", "bss", "bss_gp")


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def comp(rows, ours, bases):
    cl = [r["coupon"] for r in rows]; res = {}
    for b in bases:
        a = np.array([r["m"][ours]["err"] for r in rows]); e = np.array([r["m"][b]["err"] for r in rows]); d = (e - a) / e
        res[b] = {"median_rel_pct": 100 * float(np.median(d)), "ci_units_pct": [100 * x for x in A.boot_ci(d)],
                  "ci_coupons_pct": [100 * x for x in A.boot_ci(d, clusters=cl)], "wins": int((e > a).sum()), "n": len(d),
                  "p": float(wilcoxon(a, e).pvalue)}
    for b, q in zip(bases, holm([res[b]["p"] for b in bases])):
        res[b]["p_holm"] = float(q)
    return res


def main():
    rows = [json.loads(l) for l in open(P("e6_rows.jsonl"))]
    out = {"n_units": len(rows), "n_coupons": len({r["coupon"] for r in rows}), "freqs": sorted({r["f_kHz"] for r in rows}),
           "paths_per_coupon": len({r["path"] for r in rows if r["coupon"] == rows[0]["coupon"]}),
           "levels": [int(min(r["levels"] for r in rows)), int(max(r["levels"] for r in rows))],
           "median_err": {m: float(np.median([r["m"][m]["err"] for r in rows])) for m in rows[0]["m"]},
           "mean_err": {m: float(np.mean([r["m"][m]["err"] for r in rows])) for m in rows[0]["m"]}}
    out["br_gp_vs"] = comp(rows, "br_gp", BASES); out["br_vs"] = comp(rows, "br", BASES)
    out["br_gp_vs_br"] = comp(rows, "br_gp", ("br",))["br"]
    out["per_freq_vs_bss_gp"] = {fk: comp([r for r in rows if r["f_kHz"] == fk], "br_gp", ("bss_gp", "copy")) for fk in out["freqs"]}
    out["oracles_vs_bss_gp"] = {o: comp(rows, o, ("bss_gp",))["bss_gp"]["median_rel_pct"] for o in ("str_or", "dtw_or")}
    out["stretch_vs_copy"] = comp(rows, "bss_gp", ("copy",))["copy"]
    # per-coupon gain over the stronger stretch
    out["per_coupon_vs_bss_gp"] = {c: comp([r for r in rows if r["coupon"] == c], "br_gp", ("bss_gp",))["bss_gp"]["median_rel_pct"]
                                   for c in sorted({r["coupon"] for r in rows})}
    cov = {k: collections.defaultdict(list) for k in ("law", "conf", "bayes")}; hw = collections.defaultdict(list); kz = []
    for r in rows:
        iv = r["iv"]["br_gp"]
        for k in cov:
            if k in iv:
                cov[k][r["coupon"]] += iv[k]["cov_sig"]; hw[k].append(iv[k]["hw_rel"])
        kz.append(abs(iv["law"]["hw_rel"] - iv["conf"]["hw_rel"]) < 1e-12)
    out["coverage"] = {k: {"per_coupon": {c: float(np.mean(v)) for c, v in cov[k].items()}} for k in cov}
    for k in cov:
        pc = out["coverage"][k]["per_coupon"]
        out["coverage"][k].update({"min": min(pc.values()), "max": max(pc.values()), "n_ge_090": int(sum(v >= 0.90 for v in pc.values())),
                                   "hw_rel_median": float(np.median(hw[k]))})
    out["share_units_zero_growth"] = float(np.mean(kz))
    sel = json.load(open(P("findings.json")))["d3"]["selection_group"]
    out["selection"] = {"n_groups": sel["n_groups"], "tau_stress": sel["tau_median"]["e_stress"], "tau_in": sel["tau_median"]["e_in"],
                        "regret": {k: v["mean_regret"] for k, v in sel["rules"].items()}}
    # which method is best per coupon x frequency (test error, median over paths)
    G = collections.defaultdict(list)
    for r in rows:
        G[(r["coupon"], r["f_kHz"])].append(r)
    ms = [m for m in rows[0]["m"] if m not in ("dtw_or", "str_or")]
    best = collections.Counter(min(ms, key=lambda m: np.median([r["m"][m]["err"] for r in g])) for g in G.values())
    out["best_method_groups"] = dict(best)
    json.dump(out, open(P("d3_eval.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_units", "n_coupons", "freqs", "levels", "median_err", "oracles_vs_bss_gp", "share_units_zero_growth", "best_method_groups")}, indent=0))
    for k in ("br_gp_vs", "br_vs"):
        print(k, {b: (round(v["median_rel_pct"], 1), [round(x, 1) for x in v["ci_coupons_pct"]], f"{v['p_holm']:.1e}", v["wins"]) for b, v in out[k].items()})
    print("vs br", out["br_gp_vs_br"]); print("stretch vs copy", round(out["stretch_vs_copy"]["median_rel_pct"], 1))
    print({fk: (round(v["bss_gp"]["median_rel_pct"], 1), round(v["copy"]["median_rel_pct"], 1)) for fk, v in out["per_freq_vs_bss_gp"].items()})
    print({c: round(v, 1) for c, v in out["per_coupon_vs_bss_gp"].items()})
    print({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "per_coupon"} for k, v in out["coverage"].items()})
    print(out["selection"])


if __name__ == "__main__":
    main()
