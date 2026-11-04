#!/usr/bin/env python3
"""Summary numbers of all experiments, from the result files only -> results/findings.json.

Main pass (6 frequencies, with the stronger-BSS pass merged), confirmation pass (6 frequencies never opened during
development), E8 ablations, E4b mixing law, E6 (D3), neural baselines. Every block is skipped if its file is missing."""
import collections, json, os, sys
import numpy as np
from scipy.stats import wilcoxon, spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = lambda n: os.path.join(R, n)
out = {}


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b); d = b - a; rel = d / b
    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    return {"n": int(len(a)), "median_rel_gain": float(np.median(rel)), "ci95": A.boot_ci(rel), "wins": int((d > 0).sum()), "p": p,
            "median_ours": float(np.median(a)), "median_base": float(np.median(b))}


def block(rows, tag):
    res = {"n_units": len(rows)}
    ms = [m for m in ("copy", "knn", "gp", "bss", "bss_tr", "bss_best", "bss_gp", "bss_tr_gp", "bss_best_gp", "br", "br_gp", "str_or", "dtw_or") if m in rows[0]["m"]]
    res["summary"] = A.summary_table(rows, ms)
    res["ours_vs"] = A.compare(rows, "br_gp", [m for m in ("bss_best", "bss_best_gp", "bss", "bss_tr", "gp", "copy", "br") if m in rows[0]["m"]], "split")
    res["by_freq"] = {f: {"summary": A.summary_table([r for r in rows if r["f_kHz"] == f], ms),
                          "vs_bss_best": A.compare([r for r in rows if r["f_kHz"] == f], "br_gp", ["bss_best"], "split")["bss_best"]}
                      for f in sorted({r["f_kHz"] for r in rows})}
    res["by_split"] = {s: A.compare([r for r in rows if r["split"] == s], "br_gp", ["bss_best"], "split")["bss_best"] for s in sorted({r["split"] for r in rows})}
    cands = [m for m in ("copy", "knn", "gp", "bss", "bss_tr", "bss_gp", "bss_tr_gp", "br", "br_gp") if m in rows[0]["m"]]
    fr = A.friedman(rows, cands, ("split", "f_kHz"))
    res["friedman"] = {k: v for k, v in fr.items() if k != "task_matrix"}; res["friedman_posthoc"] = A.posthoc(fr, "br_gp")
    res["coverage"] = A.coverage(rows, "split", methods=("br_gp", "bss_best_gp", "gp"))
    res["coverage_task_min"] = {}
    for m in ("br_gp",):
        G = collections.defaultdict(list)
        for r in rows:
            if m in r["iv"] and "law" in r["iv"][m]:
                G[(r["split"], r["f_kHz"])] += r["iv"][m]["law"]["cov_sig"]
        if G:
            v = {f"{k[0]}|{k[1]}": float(np.mean(x)) for k, x in G.items()}
            res["coverage_task_min"][m] = {"min": min(v.values()), "share_below_0.90": float(np.mean([x < 0.90 for x in v.values()])), "tasks": v}
    res["selection_group"] = A.selection(rows, cands, ("split", "f_kHz"))
    res["chosen_bss_best"] = dict(collections.Counter(r["m"]["bss_best"]["chosen"] for r in rows)) if "bss_best" in rows[0]["m"] else {}
    res["resid_effect"] = {f"{a}_vs_{b}": paired([r["m"][a]["err"] for r in rows], [r["m"][b]["err"] for r in rows])
                           for a, b in (("br_gp", "br"), ("bss_tr_gp", "bss_tr"), ("bss_gp", "bss")) if a in rows[0]["m"] and b in rows[0]["m"]}
    if "e_in" in rows[0]["m"]["gp"]:
        res["in_range"] = {m: float(np.median([r["m"][m]["e_in"] for r in rows])) for m in cands}
    return res


if os.path.exists(F("e2_rows.jsonl")):
    rows = A.load(F("e2_rows.jsonl"), [F("e2_rows_bss_tr.jsonl"), F("e2_rows_brgp_fix.jsonl")])
    if "bss_best" in rows[0]["m"]:
        out["main"] = block(rows, "main")
        out["main"]["cfg_br_gp"] = dict(collections.Counter(json.dumps(r["cfg"]["br_gp"], sort_keys=True) for r in rows))
        K = collections.defaultdict(list)
        for r in rows:
            if r["path"] == 0:
                K[r["f_kHz"]].append(r["kappa"])
        out["main"]["kappa_by_freq"] = {f: {"S0": float(np.median([k["S0"] for k in v])), "A0": float(np.median([k["A0"] for k in v])),
                                            "ratio_A0_S0": float(np.median([k["A0"] / k["S0"] for k in v]))} for f, v in sorted(K.items())}
        edges = [0, 10, 20, 40]
        def bydist(m, getter):
            o = {}
            for f in sorted({r["f_kHz"] for r in rows}):
                b = collections.defaultdict(list)
                for r in rows:
                    if r["f_kHz"] != f:
                        continue
                    vals = getter(r, m)
                    if vals is None:
                        continue
                    for d, e in zip(r["dT_test"], vals):
                        b[int(np.digitize(d, edges)) - 1].append(e)
                o[f] = {f"{edges[k]}-{edges[k + 1]}": float(np.median(v)) for k, v in sorted(b.items())}
            return o
        out["main"]["err_by_distance"] = {m: bydist(m, lambda r, mm: r["m"][mm]["err_sig"]) for m in ("br_gp", "bss_best", "br", "str_or")}
        def covget(kind):
            return lambda r, mm: r["iv"].get(mm, {}).get(kind, {}).get("cov_sig")
        cb = {}
        for kind in ("law", "conf", "bayes"):
            pts = collections.defaultdict(list)
            for r in rows:
                v = r["iv"].get("br_gp", {}).get(kind, {}).get("cov_sig")
                if v:
                    for d, c in zip(r["dT_test"], v):
                        pts[int(d // 5) * 5].append(c)
            cb[kind] = {f"{k}-{k + 5}": float(np.mean(v)) for k, v in sorted(pts.items()) if len(v) >= 50}
        out["main"]["coverage_br_gp_by_distance"] = cb
if os.path.exists(F("e2_rows_confirm.jsonl")):
    rc = [json.loads(l) for l in open(F("e2_rows_confirm.jsonl"))]
    if len(rc) == 2376:
        A.add_best_bss(rc); out["confirm"] = block(rc, "confirm")

if os.path.exists(F("e8_rows.jsonl")):
    e8 = [json.loads(l) for l in open(F("e8_rows.jsonl"))]; keys = list(e8[0]["m"])
    E = lambda k, f="err": np.array([r["m"][k][f] for r in e8])
    res = {"n_units": len(e8), "median": {k: float(np.median(E(k))) for k in keys}}
    res["br_cal_vs_br_uni"] = paired(E("br_cal"), E("br_uni")); res["br_cal_vs_br_nom"] = paired(E("br_cal"), E("br_nom"))
    res["best_cal_vs_best_uni"] = paired(E("best_cal"), E("best_uni")); res["best_cal_vs_best_nom"] = paired(E("best_cal"), E("best_nom"))
    bb = np.minimum(E("bss"), E("bss_tr"))          # per-unit better of the two (an optimistic baseline)
    res["configs_vs_bss"] = {k: paired(E(k), E("bss")) for k in keys if k.startswith("cfg_")}
    res["share_configs_beating_bss"] = float(np.mean([v["median_rel_gain"] > 0 for v in res["configs_vs_bss"].values()]))
    res["best_cal_vs_bss_metrics"] = {f: paired(E("best_cal", f), E("bss", f)) for f in ("err", "err_env", "err_cc", "err_median")}
    res["best_cal_vs_oracle_min_bss"] = paired(E("best_cal"), bb)
    res["by_freq"] = {f: {k: float(np.median([r["m"][k]["err"] for r in e8 if r["f_kHz"] == f])) for k in ("bss", "bss_tr", "br_cal", "br_nom", "br_uni", "best_cal", "best_nom", "best_uni")}
                      for f in sorted({r["f_kHz"] for r in e8})}
    res["by_freq_uni_test"] = {f: paired([r["m"]["br_cal"]["err"] for r in e8 if r["f_kHz"] == f], [r["m"]["br_uni"]["err"] for r in e8 if r["f_kHz"] == f])
                               for f in sorted({r["f_kHz"] for r in e8})}
    dmap = {(r["split"], r["f_kHz"], r["path"]): r["dT_test"] for r in (json.loads(l) for l in open(F("e2_rows.jsonl")))}
    ratio = collections.defaultdict(list)
    for r in e8:
        dd = dmap[(r["split"], r["f_kHz"], r["path"])]
        for d, a, b in zip(dd, r["m"]["cfg_inf_1_gp"]["err_sig"], r["m"]["cfg_1000_1_gp"]["err_sig"]):
            ratio["<10" if d < 10 else ("10-20" if d < 20 else ">=20")].append(a / b - 1)
    res["tau_inf_vs_1000_excess_by_distance"] = {k: float(np.median(v)) for k, v in ratio.items()}
    out["e8"] = res

if os.path.exists(F("e4b_rows.jsonl")):
    mx = {(r["split"], r["f_kHz"], r["path"]): r for r in (json.loads(l) for l in open(F("e4b_rows.jsonl")))}
    res = {}
    for tag, path in (("main", "e2_rows.jsonl"), ("confirm", "e2_rows_confirm.jsonl")):
        if not os.path.exists(F(path)):
            continue
        rr = A.load(F(path), [F("e2_rows_bss_tr.jsonl"), F("e2_rows_brgp_fix.jsonl")]) if tag == "main" else [json.loads(l) for l in open(F(path))]
        if tag == "confirm":
            A.add_best_bss(rr)
        X, Y1, Y2, fq = [], [], [], []
        for r in rr:
            k = (r["split"], r["f_kHz"], r["path"])
            if k in mx and "str_or" in r["m"] and "bss_best" in r["m"]:
                X.append(mx[k]["e_mix"]); Y1.append(r["m"]["str_or"]["err"] - r["m"]["br"]["err"]); Y2.append(r["m"]["bss_best"]["err"] - r["m"]["br_gp"]["err"]); fq.append(r["f_kHz"])
        if X:
            X, Y1, Y2, fq = map(np.array, (X, Y1, Y2, fq))
            res[tag] = {"n": int(len(X)), "spearman_emix_vs_stror_minus_br": float(spearmanr(X, Y1).correlation),
                        "spearman_emix_vs_bssbest_minus_brgp": float(spearmanr(X, Y2).correlation),
                        "by_freq": {int(f): {"median_emix": float(np.median(X[fq == f])), "median_stror_minus_br": float(np.median(Y1[fq == f])),
                                             "median_bssbest_minus_brgp": float(np.median(Y2[fq == f])),
                                             "median_H": float(np.median([mx[(r["split"], r["f_kHz"], r["path"])]["H"] for r in rr if r["f_kHz"] == f]))}
                                    for f in sorted(set(fq.tolist()))}}
            fm = sorted(res[tag]["by_freq"]); res[tag]["spearman_over_freqs"] = float(spearmanr([res[tag]["by_freq"][f]["median_emix"] for f in fm],
                                                                                              [res[tag]["by_freq"][f]["median_bssbest_minus_brgp"] for f in fm]).correlation)
    out["e4b"] = res

if os.path.exists(F("e4_law_rows.jsonl")):
    em, ep, ph, fq, sp = [], [], [], [], []
    for l in open(F("e4_law_rows.jsonl")):
        r = json.loads(l)
        for d, a, b, p_ in r["rows"]:
            em.append(a); ep.append(b); ph.append(abs(p_)); fq.append(r["f_kHz"]); sp.append(r["split"])
    em, ep, ph, fq = map(np.array, (em, ep, ph, fq))
    ok = np.isfinite(ep) & (em > 0)
    res = {"n_signals": int(ok.sum()), "spearman_pred_meas": float(spearmanr(ep[ok], em[ok]).correlation),
           "pearson_pred_meas": float(np.corrcoef(ep[ok], em[ok])[0, 1]), "median_ratio_pred_over_meas": float(np.median(ep[ok] / em[ok])),
           "by_freq": {int(f): {"spearman": float(spearmanr(ep[ok & (fq == f)], em[ok & (fq == f)]).correlation),
                                "median_ratio": float(np.median(ep[ok & (fq == f)] / em[ok & (fq == f)])),
                                "median_meas": float(np.median(em[ok & (fq == f)]))} for f in sorted(set(fq.tolist()))}}
    # collapse: measured copy error against the phase gap, spread between frequencies inside phase bins
    bins = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 10])
    coll = {}
    for k in range(len(bins) - 1):
        sel = ok & (ph >= bins[k]) & (ph < bins[k + 1])
        if sel.sum() < 100:
            continue
        per_f = [float(np.median(em[sel & (fq == f)])) for f in sorted(set(fq[sel].tolist())) if (sel & (fq == f)).sum() >= 30]
        coll[f"{bins[k]}-{bins[k + 1]}"] = {"median": float(np.median(em[sel])), "freq_medians_min_max": [min(per_f), max(per_f)], "n_freqs": len(per_f)}
    res["collapse_by_phase"] = coll
    out["e4"] = res

if os.path.exists(F("e6_rows.jsonl")):
    d3 = [json.loads(l) for l in open(F("e6_rows.jsonl"))]
    ms = [m for m in ("copy", "knn", "gp", "bss", "bss_gp", "br", "br_gp") if m in d3[0]["m"]]
    out["d3"] = {"n_units": len(d3), "summary": A.summary_table(d3, ms), "ours_vs": A.compare(d3, "br_gp", [m for m in ms if m != "br_gp"], "coupon"),
                 "coverage": A.coverage(d3, "coupon", methods=("br_gp", "bss_gp", "gp")),
                 "br_vs": A.compare(d3, "br", [m for m in ms if m not in ("br",)], "coupon"),
                 "selection_group": A.selection(d3, ms, ("coupon", "f_kHz")),
                 "act_fallback_coupons": sorted({r["coupon"] for r in d3 if r.get("act_source", r["coupon"]) != r["coupon"]})}
    out["d3"]["coverage_blocks_below_0.90"] = int(sum(1 for b, x in out["d3"]["coverage"]["br_gp/law"].items() if b != "min_block" and x["coverage"] < 0.90))

if os.path.exists(F("e2_neural_rows.jsonl")):
    nr = [json.loads(l) for l in open(F("e2_neural_rows.jsonl"))]
    if os.path.exists(F("e2_neural_rows_seed2.jsonl")):          # third MF-B-NO seed
        extra = {(r["split"], r["f_kHz"], r["path"]): r for r in (json.loads(l) for l in open(F("e2_neural_rows_seed2.jsonl")))}
        for r in nr:
            e = extra.get((r["split"], r["f_kHz"], r["path"]))
            if e:
                r["m"]["mfbno_bss"] = r["m"]["mfbno_bss"] + e["m"]["mfbno_bss"]
    main = {(r["split"], r["f_kHz"], r["path"]): r for r in A.load(F("e2_rows.jsonl"), [F("e2_rows_bss_tr.jsonl"), F("e2_rows_brgp_fix.jsonl")])}
    res = {"n_units": len(nr)}
    for k in [k for k in ("mfbno_bss", "mfbno_nom", "fvae") if k in nr[0]["m"]]:
        e = np.array([np.mean([s["err"] for s in r["m"][k]]) for r in nr])
        ours = np.array([main[(r["split"], r["f_kHz"], r["path"])]["m"]["br_gp"]["err"] for r in nr])
        res[k] = {"median": float(np.median(e)), "seed_sd_median": float(np.median([np.std([s["err"] for s in r["m"][k]]) for r in nr])),
                  "ours_vs": paired(ours, e)}
        if k.startswith("mfbno"):
            for kind in ("bayes", "conf", "law"):
                cov = collections.defaultdict(list)
                for r in nr:
                    for s in r["m"][k]:
                        cov[r["split"]] += s[f"cov_{kind}_sig"]
                res[k][f"coverage_{kind}_min_block"] = float(min(np.mean(v) for v in cov.values()))
    res["bss_best_on_units"] = float(np.median([main[(r["split"], r["f_kHz"], r["path"])]["m"]["bss_best"]["err"] for r in nr]))
    res["mfbno_better_than_bss_best"] = int(sum(np.mean([s_["err"] for s_ in r["m"]["mfbno_bss"]]) < main[(r["split"], r["f_kHz"], r["path"])]["m"]["bss_best"]["err"] for r in nr))
    res["cost_s_median"] = {"mfbno_bss_per_run": float(np.median([s_["sec"] for r in nr for s_ in r["m"]["mfbno_bss"]])),
                            "mfbno_checkpoint_steps": float(np.median([s_["steps"] for r in nr for s_ in r["m"]["mfbno_bss"]])),
                            "br_gp_fit": float(np.median([main[(r["split"], r["f_kHz"], r["path"])]["m"]["br_gp"]["cost_s"] for r in nr]))}
    out["neural"] = res

json.dump(out, open(F("findings.json"), "w"), indent=1)
print(json.dumps({k: list(v.keys()) for k, v in out.items()}, indent=0))
