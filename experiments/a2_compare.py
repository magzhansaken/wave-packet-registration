"""A2 development summary on the two D1 halves (post-hoc analysis): br / br_gp (through the origin) and br2 / br_gp2
(intercept) against bss_best, per frequency; coverage of the law-guided conformal intervals per split."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.analyze import load
R = os.path.join(os.path.dirname(__file__), "..", "results")
P = lambda f: os.path.join(R, f)

def summ(rows, tag):
    rows = [r for r in rows if "br_gp2" in r["m"] and "bss_best" in r["m"]]
    out = {"half": tag, "n_units": len(rows), "per_f": {}, "pooled": {}, "cov_split": {}}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["f_kHz"]].append(r)
    def rel(rs, m):
        return np.array([(r["m"]["bss_best"]["err"] - r["m"][m]["err"]) / r["m"]["bss_best"]["err"] for r in rs])
    for f in sorted(by):
        out["per_f"][f] = {m: round(100 * float(np.median(rel(by[f], m))), 1) for m in ("br", "br_gp", "br2", "br_gp2")}
    for band, fs in (("40", [40]), ("60-100", [60, 80, 100]), ("120-260", list(range(120, 261, 20))), ("all", list(range(40, 261, 20)))):
        rs = [r for r in rows if r["f_kHz"] in fs]
        if rs:
            out["pooled"][band] = {m: round(100 * float(np.median(rel(rs, m))), 2) for m in ("br_gp", "br_gp2")}
    bys = collections.defaultdict(list)
    for r in rows:
        bys[r["split"]].append(r)
    for s, rs in sorted(bys.items()):
        out["cov_split"][s] = {m: round(float(np.mean([np.mean(r["iv"][m]["law"]["cov_sig"]) for r in rs])), 3) for m in ("br_gp", "br_gp2")}
    return out

if __name__ == "__main__":
    main = load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl"), P("a2dev_rows_main.jsonl")])
    conf = load(P("e2_rows_confirm.jsonl"), [P("a2dev_rows_main.jsonl")])
    res = load(P("e2_rows_reserve.jsonl"), [P("a2dev_rows_reserve.jsonl")])
    rep = [summ(main + conf, "main (main + confirmation frequencies)"), summ(res, "reserve")]
    json.dump(rep, open(P("a2dev_summary.json"), "w"), indent=1)
    for x in rep:
        print(x["half"], x["n_units"]); print(" per f:", x["per_f"]); print(" pooled:", x["pooled"]); print(" cov:", x["cov_split"])
