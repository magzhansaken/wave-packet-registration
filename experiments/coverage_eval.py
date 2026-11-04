#!/usr/bin/env python3
"""Section 4.7: coverage of the nominal 95 % bands of br_gp on the three sets of D1 - drift-law conformal ('law'), split
conformal ('conf') and the Gaussian-process posterior ('bayes') - per extrapolation block (split; mean over frequencies,
paths and test signals), per task (split x frequency), by distance from the training edge (2.5 C bins), and the mean
relative half-width. D3 per coupon from results/findings.json. E8b sensitivity to the validation cut from
results/e8b_summary.json. Output: results/coverage_eval.json"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
SETS = {"main": lambda: A.load(P("e2_rows.jsonl"), [P("e2_rows_brgp_fix.jsonl")]) + A.load(P("e2_rows_confirm.jsonl")),
        "reserve": lambda: A.load(P("e2_rows_reserve.jsonl")), "dam_D16": lambda: A.load(P("e2_rows_dam.jsonl"))}
BINS = np.arange(0, 35, 2.5)


def main():
    out = {}
    for name, f in SETS.items():
        rows = [r for r in f() if "br_gp" in r.get("iv", {})]
        res = {"n_units": len(rows)}
        for k in ("law", "conf", "bayes"):
            blk = collections.defaultdict(list); task = collections.defaultdict(list); dist = collections.defaultdict(list); hw = []
            for r in rows:
                iv = r["iv"]["br_gp"].get(k)
                if not iv:
                    continue
                c = np.mean(iv["cov_sig"]); blk[r["split"]].append(c); task[(r["split"], r["f_kHz"])].append(c); hw.append(iv["hw_rel"])
                for d, cs in zip(r["dT_test"], iv["cov_sig"]):
                    dist[int(np.digitize(d, BINS)) - 1].append(cs)
            if not blk:
                continue
            b = {s: float(np.mean(v)) for s, v in sorted(blk.items())}; t = [float(np.mean(v)) for v in task.values()]
            res[k] = {"blocks": b, "min_block": min(b.values()), "max_block": max(b.values()), "task_min": min(t),
                      "share_tasks_below_090": float(np.mean(np.array(t) < 0.90)), "hw_rel_mean": float(np.mean(hw)),
                      "by_distance": {f"{BINS[i]:.1f}": [float(np.mean(v)), len(v)] for i, v in sorted(dist.items()) if len(v) >= 50}}
        out[name] = res
        print(name, len(rows), {k: (round(res[k]["min_block"], 3), round(res[k]["hw_rel_mean"], 2)) for k in ("law", "conf", "bayes") if k in res})
    fd = json.load(open(P("findings.json")))["d3"]["coverage"]
    law = {k: v["coverage"] for k, v in fd["br_gp/law"].items() if k != "min_block"}
    out["d3"] = {"law_min": min(law.values()), "law_below_090": int(sum(v < 0.90 for v in law.values())), "n_coupons": len(law),
                 "conf_min": fd["br_gp/conf"]["min_block"], "bayes_min": fd["br_gp/bayes"]["min_block"]}
    e8b = json.load(open(P("e8b_summary.json")))
    out["e8b"] = {k: {"min_block": v["min_block"], "share_tasks_below_090": v["share_tasks_below_0.90"], "hw": v["hw"]} for k, v in e8b.items() if isinstance(v, dict) and "min_block" in v}
    json.dump(out, open(P("coverage_eval.json"), "w"), indent=1)
    print(json.dumps(out["d3"]), json.dumps(out["e8b"]))


if __name__ == "__main__":
    main()
