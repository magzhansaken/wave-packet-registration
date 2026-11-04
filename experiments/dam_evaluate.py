#!/usr/bin/env python3
"""Verdict of amendment 11 on dam_D16 (clean test). Rules exactly as written in docs/PROTOCOL.md, § 11:
bands low {40}, mixing {60, 80, 100}, high {120..260} kHz; unit = split x frequency x path; two-sided Wilcoxon per band,
Holm within the family of each claim (A1: 3 bands; A2: all frequencies + 3 bands); effect = median relative difference
(bss_best - ours) / bss_best with a unit bootstrap CI (10 000) and a split-cluster bootstrap; coverage of the law-guided
conformal intervals >= 0.90 in every split; br_gp vs MF-B-NO on the 36 neural units.
Output: results/dam_eval.json."""
import json, os, sys, collections
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results")
P = lambda f: os.path.join(R, f)
BANDS = {"low_40": [40], "mix_60_100": [60, 80, 100], "high_120_260": list(range(120, 261, 20))}
ALL = list(range(40, 261, 20))


def rel(rs, m, ref="bss_best"):
    a = np.array([r["m"][ref]["err"] for r in rs]); b = np.array([r["m"][m]["err"] for r in rs])
    return a, b, (a - b) / a


def test(rs, m):
    a, b, d = rel(rs, m)
    lo, hi = A.boot_ci(d); clo, chi = A.boot_ci(d, clusters=[r["split"] for r in rs])
    return {"n": len(d), "median_rel_pct": 100 * float(np.median(d)), "ci_pct": [100 * lo, 100 * hi],
            "ci_cluster_split_pct": [100 * clo, 100 * chi], "p": float(wilcoxon(a, b).pvalue), "n_better": int((d > 0).sum()),
            "mean_err_ours": float(b.mean()), "mean_err_bss_best": float(a.mean())}


def family(rows, m, groups):
    out = {g: test([r for r in rows if r["f_kHz"] in fs], m) for g, fs in groups.items()}
    ph = A.holm([out[g]["p"] for g in groups])
    for g, p in zip(groups, ph):
        out[g]["p_holm"] = float(p); out[g]["win"] = bool(p < 0.05 and out[g]["ci_pct"][0] > 0)
    return out


def main():
    rows = A.load(P("e2_rows_dam.jsonl"), [P("a2_rows_dam.jsonl")])
    assert all("bss_best" in r["m"] and "br_gp2" in r["m"] for r in rows), "incomplete rows"
    rep = {"n_units": len(rows), "splits": sorted({r["split"] for r in rows}), "freqs": sorted({r["f_kHz"] for r in rows})}
    per_f = {}
    for f in ALL:
        rs = [r for r in rows if r["f_kHz"] == f]
        per_f[f] = {m: 100 * float(np.median(rel(rs, m)[2])) for m in ("br", "br_gp", "br2", "br_gp2", "str_or")}
        per_f[f]["bss_best_is_tr"] = float(np.mean([r["m"]["bss_best"].get("which", "") == "bss_tr" for r in rs])) if rs and "which" in rs[0]["m"]["bss_best"] else None
    rep["per_freq_median_rel_pct"] = per_f
    # A1
    a1 = family(rows, "br_gp", BANDS)
    a1a = a1["mix_60_100"]["win"] and all(per_f[f]["br_gp"] > 0 for f in (60, 80, 100))
    a1b = not a1["low_40"]["win"] and not a1["high_120_260"]["win"]
    rep["A1"] = {"tests": a1, "A1a": bool(a1a), "A1b": bool(a1b),
                 "verdict": "confirmed" if (a1a and a1b) else ("new_finding_win_outside_band" if a1a else "not_confirmed")}
    # A2
    g2 = {"all": ALL, **BANDS}
    a2 = family(rows, "br_gp2", g2)
    rep["A2"] = {"tests": a2, "verdict": "confirmed" if a2["all"]["win"] else "not_confirmed"}
    # coverage per split (law-guided conformal intervals), br_gp and br_gp2
    cov = {}
    for m in ("br_gp", "br_gp2"):
        bys = collections.defaultdict(list)
        for r in rows:
            bys[r["split"]].append(np.mean(r["iv"][m]["law"]["cov_sig"]))
        cov[m] = {s: float(np.mean(v)) for s, v in sorted(bys.items())}
        cov[m + "_min"] = min(cov[m].values()); cov[m + "_pass_0.90"] = bool(cov[m + "_min"] >= 0.90)
        cov[m + "_hw_rel_mean"] = float(np.mean([r["iv"][m]["law"]["hw_rel"] for r in rows]))
    rep["coverage"] = cov
    # other baselines: mean error over all units, and Friedman by task
    meth = ["copy", "knn", "gp", "bss", "bss_gp", "bss_tr", "bss_tr_gp", "bss_best", "br", "br_gp", "br2", "br_gp2"]
    rep["mean_err"] = {m: float(np.mean([r["m"][m]["err"] for r in rows])) for m in meth + ["str_or", "dtw_or"] if m in rows[0]["m"]}
    # neural baselines on the 36 units
    if os.path.exists(P("e2_neural_rows_dam.jsonl")):
        nr = {(r["split"], r["f_kHz"], r["path"]): r for r in (json.loads(l) for l in open(P("e2_neural_rows_dam.jsonl")))}
        U = [(r, nr[(r["split"], r["f_kHz"], r["path"])]) for r in rows if (r["split"], r["f_kHz"], r["path"]) in nr]
        out = {"n_units": len(U)}
        for k in ("mfbno_bss", "fvae"):
            if U and k in U[0][1]["m"]:
                a = np.array([np.mean([s["err"] for s in n["m"][k]]) for _, n in U]); b = np.array([r["m"]["br_gp"]["err"] for r, _ in U])
                sd = np.array([np.std([s["err"] for s in n["m"][k]]) for _, n in U])
                d = (a - b) / a; lo, hi = A.boot_ci(d)
                out[k] = {"seeds": len(U[0][1]["m"][k]), "mean_err": float(a.mean()), "seed_sd_mean": float(sd.mean()), "br_gp_mean_err": float(b.mean()),
                          "median_rel_pct": 100 * float(np.median(d)), "ci_pct": [100 * lo, 100 * hi], "p": float(wilcoxon(a, b).pvalue)}
        rep["neural"] = out
    json.dump(rep, open(P("dam_eval.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n_units",)}), "A1:", rep["A1"]["verdict"], "A2:", rep["A2"]["verdict"],
          "cov br_gp min:", round(cov["br_gp_min"], 3))


if __name__ == "__main__":
    main()
