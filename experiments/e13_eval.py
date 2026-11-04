#!/usr/bin/env python3
"""Evaluation of amendment 13 (docs/PROTOCOL.md, written before the runs):
13.1 AUC of damage detection on dam_D16 after compensation (e13_auc_rows.jsonl);
13.2 br_gp against the windowed shift-and-stretch sdws on three sets (e13_rows_*.jsonl);
13.3 coverage of the growth law, split conformal, nexCP and ACI on the same units;
13.4 br_gp against MF-B-NO in the band (60, 80 kHz: e2_neural_rows_band*.jsonl; 100 kHz: earlier neural rows).
Output: results/e13_eval.json"""
import collections, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results"); P = lambda f: os.path.join(R, f)
BANDS = {"low": lambda f: f == 40, "mix": lambda f: 60 <= f <= 100, "high": lambda f: f >= 120}
rows_of = lambda f: [json.loads(l) for l in open(P(f))] if os.path.exists(P(f)) else []


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def effect(d, cl):
    d = np.asarray(d, float)
    return {"n": int(len(d)), "median": float(np.median(d)), "ci_units": list(A.boot_ci(d)), "ci_splits": list(A.boot_ci(d, clusters=cl)),
            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "wins": int((d > 0).sum())}


def main():
    out = {}
    # 13.2 / 13.3
    sd, cov = {}, {}
    for s in ("main", "reserve", "dam"):
        rs = rows_of(f"e13_rows_{s}.jsonl")
        if not rs:
            continue
        res = {"n_units": len(rs), "freqs": sorted({r["f_kHz"] for r in rs})}
        for b, fn in BANDS.items():
            u = [r for r in rs if fn(r["f_kHz"])]
            if u:
                d = [100 * (r["m"]["sdws"]["err"] - r["m"]["br_gp"]["err"]) / r["m"]["sdws"]["err"] for r in u]
                res[b] = effect(d, [r["split"] for r in u])
                res[b]["median_err_sdws"] = float(np.median([r["m"]["sdws"]["err"] for r in u]))
                res[b]["median_err_br_gp"] = float(np.median([r["m"]["br_gp"]["err"] for r in u]))
        res["cfg_sdws"] = dict(collections.Counter(json.dumps(r["cfg_sdws"], sort_keys=True) for r in rs if r["path"] == min(x["path"] for x in rs)).most_common(3))
        res["by_freq"] = {f: float(np.median([100 * (r["m"]["sdws"]["err"] - r["m"]["br_gp"]["err"]) / r["m"]["sdws"]["err"] for r in rs if r["f_kHz"] == f])) for f in res["freqs"]}
        sd[s] = res
        cv = {}
        rb = [r for r in rs if BANDS["mix"](r["f_kHz"])]          # coverage compared on the band units of every set
        for k in ("law", "conf", "nexcp", "aci"):
            G = collections.defaultdict(list); T = collections.defaultdict(list); hw = []
            for r in rb:
                c = r["iv"]["br_gp"][k]["cov_sig"]; G[r["split"]] += c; T[(r["split"], r["f_kHz"])] += c; hw.append(r["iv"]["br_gp"][k]["hw_rel"])
            blocks = {b: float(np.mean(v)) for b, v in G.items()}
            cv[k] = {"min_block": min(blocks.values()), "blocks": blocks, "share_tasks_below_090": float(np.mean([np.mean(v) < 0.90 for v in T.values()])),
                     "hw_rel_mean": float(np.mean(hw)), "kappa_zero_share": float(np.mean([r["iv"]["br_gp"]["kappa_growth"] == 0 for r in rb])), "n_units": len(rb)}
        cov[s] = cv
    names = [s for s in ("main", "reserve", "dam") if s in sd and "mix" in sd[s]]
    for s, q in zip(names, holm([sd[s]["mix"]["p"] for s in names])):
        sd[s]["mix"]["p_holm"] = q
    sd["verdict_registration_better_all_sets"] = bool(names) and all(sd[s]["mix"]["p_holm"] < 0.05 and sd[s]["mix"]["ci_units"][0] > 0 for s in names)
    out["sdws"] = sd; out["coverage"] = cov
    if cov:
        nl = [s for s in cov]
        out["coverage_verdict"] = {"law_ge_090_main_dam": all(cov[s]["law"]["min_block"] >= 0.90 for s in ("main", "dam") if s in cov),
                                   "law_above_nexcp_sets": int(sum(cov[s]["law"]["min_block"] > cov[s]["nexcp"]["min_block"] for s in nl)), "n_sets": len(nl)}
    # 13.1 AUC
    ar = rows_of("e13_auc_rows.jsonl")
    if ar:
        au = {"n_units": len(ar), "n_pos_median": float(np.median([r["n_pos"] for r in ar])), "n_neg_median": float(np.median([r["n_neg"] for r in ar]))}
        ps = {}
        for b, fn in BANDS.items():
            u = [r for r in ar if fn(r["f_kHz"]) and np.isfinite(r["m"]["br_gp"]["auc"]) and np.isfinite(r["m"]["bss_best"]["auc"])]
            if not u:
                continue
            d = [r["m"]["br_gp"]["auc"] - r["m"]["bss_best"]["auc"] for r in u]
            au[b] = effect(d, [r["split"] for r in u])
            au[b]["median_auc_br_gp"] = float(np.median([r["m"]["br_gp"]["auc"] for r in u]))
            au[b]["median_auc_bss_best"] = float(np.median([r["m"]["bss_best"]["auc"] for r in u]))
            au[b]["mean_auc_br_gp"] = float(np.mean([r["m"]["br_gp"]["auc"] for r in u]))
            au[b]["mean_auc_bss_best"] = float(np.mean([r["m"]["bss_best"]["auc"] for r in u]))
            au[b]["share_ge"] = float(np.mean(np.array(d) >= 0)); ps[b] = au[b]["p"]
            # aggregated index: mean residual over the 12 paths per split x frequency
            G = collections.defaultdict(list)
            for r in u:
                G[(r["split"], r["f_kHz"])].append(r)
            agg = {}
            for meth in ("br_gp", "bss_best"):
                vals = []
                for g in G.values():
                    pos = np.mean([x["m"][meth]["di_pos"] for x in g], axis=0); neg = np.mean([x["m"][meth]["di_neg"] for x in g], axis=0)
                    vals.append(float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])))
                agg[meth] = vals
            au[b]["agg_auc_br_gp_median"] = float(np.median(agg["br_gp"])); au[b]["agg_auc_bss_best_median"] = float(np.median(agg["bss_best"]))
            au[b]["agg_tasks"] = len(agg["br_gp"])
        for b, q in zip(ps, holm(list(ps.values()))):
            au[b]["p_holm"] = q
        au["by_freq"] = {f: {"auc_br_gp": float(np.median([r["m"]["br_gp"]["auc"] for r in ar if r["f_kHz"] == f])),
                             "auc_bss_best": float(np.median([r["m"]["bss_best"]["auc"] for r in ar if r["f_kHz"] == f]))}
                         for f in sorted({r["f_kHz"] for r in ar})}
        au["verdict"] = bool("mix" in au and au["mix"].get("p_holm", 1) < 0.05 and au["mix"]["ci_units"][0] > 0)
        out["auc"] = au
    # 13.4 MF-B-NO in the band
    mf = {}
    for s, new, old, base in (("main", "e2_neural_rows_band.jsonl", ["e2_neural_rows.jsonl", "e2_neural_rows_seed2.jsonl"],
                               lambda: A.load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl")]) + A.load(P("e2_rows_confirm.jsonl"))),
                              ("dam", "e2_neural_rows_dam_band.jsonl", ["e2_neural_rows_dam.jsonl"], lambda: A.load(P("e2_rows_dam.jsonl")))):
        nr = {}
        for f in [new] + old:
            for r in rows_of(f):
                if not (60 <= r["f_kHz"] <= 100):
                    continue
                k = (r["split"], r["f_kHz"], r["path"])
                nr.setdefault(k, r)
                if nr[k] is not r:
                    nr[k]["m"]["mfbno_bss"] = nr[k]["m"]["mfbno_bss"] + r["m"]["mfbno_bss"]
        if not nr:
            continue
        b = {(r["split"], r["f_kHz"], r["path"]): r for r in base()}
        keys = sorted(nr); e_mf = [np.mean([x["err"] for x in nr[k]["m"]["mfbno_bss"]]) for k in keys]; e_br = [b[k]["m"]["br_gp"]["err"] for k in keys]
        d = [100 * (a - c) / a for a, c in zip(e_mf, e_br)]
        mf[s] = effect(d, [k[0] for k in keys]); mf[s]["freqs"] = sorted({k[1] for k in keys})
        mf[s]["median_err_mfbno"] = float(np.median(e_mf)); mf[s]["median_err_br_gp"] = float(np.median(e_br))
        mf[s]["by_freq"] = {f: float(np.median([x for x, k in zip(d, keys) if k[1] == f])) for f in mf[s]["freqs"]}
        mf[s]["seeds"] = sorted({len(nr[k]["m"]["mfbno_bss"]) for k in keys})
    names = list(mf)
    for s, q in zip(names, holm([mf[s]["p"] for s in names])):
        mf[s]["p_holm"] = q
    mf["registration_not_worse_stop"] = any(mf[s]["median"] <= 0 or mf[s]["ci_units"][0] <= 0 for s in names) if names else None
    out["mfbno_band"] = mf
    json.dump(out, open(P("e13_eval.json"), "w"), indent=1)
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in out.items()}, indent=0))


if __name__ == "__main__":
    main()
