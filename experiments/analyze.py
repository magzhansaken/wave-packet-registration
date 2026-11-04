#!/usr/bin/env python3
"""Statistics for E2-E6 from the unit rows (docs/PROTOCOL.md).

Unit = block (split) x frequency x path (D1) or coupon x frequency x path (D3); the unit error is the mean relative L2
error over the test signals of the block. Tests: paired two-sided Wilcoxon signed-rank over units, ours vs each baseline,
Holm correction within the family of a dataset; effect size = median paired relative difference (e_base - e_ours) / e_base
with a 95 % bootstrap CI (10 000 resamples of units; cluster bootstrap over splits/coupons as a robustness check).
Friedman over tasks (dataset x split x frequency; error averaged over paths) with Holm-corrected Wilcoxon post-hoc.
Coverage threshold: mean coverage of the 95 % intervals >= 0.90 in each external block (split / coupon).
Selection (E5): rules R0 in-range, R1 TOPSIS (0.5/0.2/0.3), R1e TOPSIS-Entropy, R1c TOPSIS-CRITIC, R2 gate, R3 stress;
regret = true error of the selected surrogate minus the best true error; Kendall tau of each criterion with the truth;
Dirichlet weight perturbation of TOPSIS (SMAA-style acceptability); rank reversal when one alternative is removed.
"""
import argparse, collections, json, os
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, kendalltau

RNG = np.random.default_rng(12345)


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def boot_ci(x, clusters=None, B=10000, stat=np.median):
    """95 % percentile bootstrap CI of a statistic; resampling of units or of whole clusters."""
    x = np.asarray(x)
    if clusters is None:
        idx = RNG.integers(0, len(x), (B, len(x))); bs = np.array([stat(x[i]) for i in idx])
    else:
        cl = np.asarray(clusters); u = np.unique(cl); groups = [np.where(cl == c)[0] for c in u]; bs = []
        for _ in range(min(B, 2000)):
            pick = RNG.integers(0, len(u), len(u)); ii = np.concatenate([groups[k] for k in pick]); bs.append(stat(x[ii]))
        bs = np.array(bs)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load(path, supp=None, key=("split", "f_kHz", "path")):
    """Rows of a pass; supplementary passes (one path or a list) replace or add methods unit by unit, in order."""
    rows = [json.loads(l) for l in open(path)]
    supps = [supp] if isinstance(supp, str) else list(supp or [])
    merged = False
    for sp in supps:
        if not (sp and os.path.exists(sp)):
            continue
        extra = {tuple(r[k] for k in key): r for r in (json.loads(l) for l in open(sp))}
        for r in rows:
            e = extra.get(tuple(r[k] for k in key))
            if e:
                r["m"].update(e["m"]); r["iv"].update(e.get("iv", {})); r.setdefault("cfg", {}).update(e.get("cfg", {}))
                for k in ("kappa_inner", "s_inner"):
                    if k in e:
                        r[k] = e[k]
        merged = True
    if merged:
        add_best_bss(rows)
    return rows


def add_best_bss(rows, group=("split", "f_kHz")):
    """bss_best / bss_best_gp: per (split, frequency) the better of bss and bss_tr by the median validation (stress)
    error over the 12 subset paths - the strongest stretch baseline chosen without test data."""
    G = collections.defaultdict(list)
    for r in rows:
        G[tuple(r[k] for k in group)].append(r)
    for g, rs in G.items():
        if not all("bss_tr" in r["m"] for r in rs):
            continue
        sub = [r for r in rs if r.get("subset")] or rs
        pick = min(("bss", "bss_tr"), key=lambda m: np.median([r["m"][m]["e_stress"] for r in sub]))
        has_gp = all(m in rs[0]["m"] for m in ("bss_gp", "bss_tr_gp"))       # the reserve pass has no GP variants
        pick_gp = min(("bss_gp", "bss_tr_gp"), key=lambda m: np.median([r["m"][m]["e_stress"] for r in sub])) if has_gp else None
        for r in rs:
            r["m"]["bss_best"] = dict(r["m"][pick], chosen=pick)
            if has_gp:
                r["m"]["bss_best_gp"] = dict(r["m"][pick_gp], chosen=pick_gp)
                if pick_gp in r["iv"]:
                    r["iv"]["bss_best_gp"] = r["iv"][pick_gp]


def compare(rows, ours, bases, cluster_key):
    """Paired comparison of ours against each baseline: Wilcoxon, Holm, median relative difference and CIs."""
    E = {m: np.array([r["m"][m]["err"] for r in rows]) for m in [ours] + bases if m in rows[0]["m"]}
    cl = [r[cluster_key] for r in rows]
    out = {}; ps = []
    for b in bases:
        if b not in E:
            continue
        d = E[b] - E[ours]
        p = float(wilcoxon(E[ours], E[b], alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
        rel = d / E[b]
        out[b] = {"n_units": len(d), "mean_ours": float(E[ours].mean()), "mean_base": float(E[b].mean()),
                  "median_rel_gain": float(np.median(rel)), "ci95": boot_ci(rel), "ci95_cluster": boot_ci(rel, cl),
                  "wins": int((d > 0).sum()), "p": p}
        ps.append(p)
    for b, pa in zip([b for b in bases if b in E], holm(ps)):
        out[b]["p_holm"] = float(pa)
    return out


def summary_table(rows, methods):
    """Mean, standard deviation, median and quartiles of the unit error of every method."""
    res = {}
    for m in methods:
        if m not in rows[0]["m"]:
            continue
        e = np.array([r["m"][m]["err"] for r in rows])
        res[m] = {"mean": float(e.mean()), "std": float(e.std()), "median": float(np.median(e)), "q25": float(np.percentile(e, 25)),
                  "q75": float(np.percentile(e, 75))}
    return res


def friedman(rows, methods, task_keys):
    """Friedman test over tasks on the task-mean errors of the methods present in every task."""
    tasks = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        k = tuple(r[x] for x in task_keys)
        for m in methods:
            if m in r["m"]:
                tasks[k][m].append(r["m"][m]["err"])
    keys = sorted(tasks); M = [m for m in methods if all(m in tasks[k] for k in keys)]
    A = np.array([[np.mean(tasks[k][m]) for m in M] for k in keys])
    st = friedmanchisquare(*A.T)
    ranks = np.argsort(np.argsort(A, 1), 1).mean(0) + 1
    return {"n_tasks": len(keys), "methods": M, "chi2": float(st.statistic), "p": float(st.pvalue),
            "mean_rank": {m: float(r) for m, r in zip(M, ranks)}, "task_matrix": A.tolist()}


def posthoc(fr, ours):
    """Wilcoxon post-hoc tests of ours against every other method over tasks, Holm-adjusted."""
    A = np.array(fr["task_matrix"]); M = fr["methods"]; i0 = M.index(ours); ps = []; names = []
    for j, m in enumerate(M):
        if j == i0:
            continue
        ps.append(float(wilcoxon(A[:, i0], A[:, j]).pvalue)); names.append(m)
    return {m: {"p": p, "p_holm": float(ph), "ours_better_tasks": int((A[:, i0] < A[:, M.index(m)]).sum())} for m, p, ph in zip(names, ps, holm(ps))}


def coverage(rows, block_key, methods=("br_gp", "bss_gp", "gp"), kinds=("bayes", "conf", "law")):
    """Coverage of the nominal bands per block and the mean relative half-width, by method and kind."""
    out = {}
    for m in methods:
        for k in kinds:
            blocks = collections.defaultdict(list); hw = collections.defaultdict(list)
            for r in rows:
                iv = r.get("iv", {}).get(m, {}).get(k)
                if iv:
                    blocks[r[block_key]] += iv["cov_sig"]; hw[r[block_key]].append(iv["hw_rel"])
            if blocks:
                out[f"{m}/{k}"] = {b: {"coverage": float(np.mean(v)), "rel_halfwidth": float(np.mean(hw[b]))} for b, v in sorted(blocks.items())}
                out[f"{m}/{k}"]["min_block"] = float(min(np.mean(v) for v in blocks.values()))
    return out


def topsis(M, w):
    """TOPSIS closeness of the alternatives (rows) for cost criteria (columns) with weights w."""
    N = M / (np.linalg.norm(M, axis=0) + 1e-30) * w; best, worst = N.min(0), N.max(0)
    dp = np.linalg.norm(N - best, axis=1); dm = np.linalg.norm(N - worst, axis=1); return dm / (dp + dm + 1e-30)


def entropy_w(M):
    """Entropy weights of the criteria (columns)."""
    P = M / (M.sum(0) + 1e-30); k = 1 / np.log(len(M)); e = -k * np.nansum(P * np.log(P + 1e-30), 0); d = 1 - e
    return d / d.sum()


def critic_w(M):
    """CRITIC weights of the criteria (columns)."""
    Z = (M.max(0) - M) / (M.max(0) - M.min(0) + 1e-30); s = Z.std(0); R = np.corrcoef(Z.T); R = np.nan_to_num(R)
    c = s * (1 - R).sum(0); return c / c.sum()


def selection(rows, cands, group_keys):
    """Decisions per group (e.g. split x frequency): criteria = median over the group's units; truth = mean true error."""
    cands = [c for c in cands if c in rows[0]["m"] and "e_stress" in rows[0]["m"][c]]
    G = collections.defaultdict(list)
    for r in rows:
        G[tuple(r[k] for k in group_keys)].append(r)
    res = {"rules": collections.defaultdict(list), "tau": collections.defaultdict(list), "smaa_regret": [], "rank_reversal": 0,
           "n_groups": len(G)}
    for g, rs in G.items():
        C = [c for c in cands if all(c in r["m"] for r in rs)]
        has_in = all("e_in" in r["m"][c] for r in rs for c in C)       # the confirmation pass has no in-range errors
        Mx = np.array([[np.median([r["m"][c]["e_in"] for r in rs]) if has_in else np.nan, np.median([r["m"][c]["cost_s"] for r in rs]) + 1e-3,
                        np.median([r["m"][c]["e_stress"] for r in rs])] for c in C])
        et = np.array([np.mean([r["m"][c]["err"] for r in rs]) for c in C]); best = et.min()
        if not has_in:
            for k, i in {"R3_stress": int(np.argmin(Mx[:, 2])), "fixed_br_gp": C.index("br_gp")}.items():
                res["rules"][k].append({"group": list(g), "pick": C[i], "regret": float(et[i] - best), "best": C[int(np.argmin(et))]})
            res["tau"]["e_stress"].append(kendalltau(Mx[:, 2], et).correlation)
            continue
        picks = {"R0_in_range": int(np.argmin(Mx[:, 0])), "R1_topsis": int(np.argmax(topsis(Mx, np.array([.5, .2, .3])))),
                 "R1e_topsis_entropy": int(np.argmax(topsis(Mx, entropy_w(Mx)))), "R1c_topsis_critic": int(np.argmax(topsis(Mx, critic_w(Mx)))),
                 "R3_stress": int(np.argmin(Mx[:, 2]))}
        ok = Mx[:, 2] <= 1.25 * Mx[:, 2].min(); cand = [i for i in range(len(C)) if ok[i]]
        picks["R2_gate_then_in_range"] = min(cand, key=lambda i: Mx[i, 0])
        picks["fixed_br_gp"] = C.index("br_gp")
        for k, i in picks.items():
            res["rules"][k].append({"group": list(g), "pick": C[i], "regret": float(et[i] - best), "best": C[int(np.argmin(et))]})
        res["tau"]["e_in"].append(kendalltau(Mx[:, 0], et).correlation); res["tau"]["e_stress"].append(kendalltau(Mx[:, 2], et).correlation)
        W = RNG.dirichlet(np.ones(3), 1000)
        res["smaa_regret"].append(float(np.mean([et[int(np.argmax(topsis(Mx, w)))] - best for w in W])))
        top = int(np.argmax(topsis(Mx, np.array([.5, .2, .3]))))
        for drop in range(len(C)):
            if drop == top:
                continue
            keep = [i for i in range(len(C)) if i != drop]
            if keep[int(np.argmax(topsis(Mx[keep], np.array([.5, .2, .3]))))] != top:
                res["rank_reversal"] += 1; break
    out = {"n_groups": res["n_groups"], "rank_reversal_groups": res["rank_reversal"],
           "smaa_mean_regret": float(np.mean(res["smaa_regret"])) if res["smaa_regret"] else None,
           "tau_median": {k: float(np.nanmedian(v)) for k, v in res["tau"].items()}, "rules": {}}
    for k, v in res["rules"].items():
        rg = np.array([x["regret"] for x in v])
        out["rules"][k] = {"mean_regret": float(rg.mean()), "median_regret": float(np.median(rg)), "zero_regret_share": float((rg < 1e-12).mean()),
                           "picks": dict(collections.Counter(x["pick"] for x in v))}
    out["best_counts"] = dict(collections.Counter(x["best"] for x in res["rules"]["R3_stress"]))
    return out


def law(rows, methods, dkey="dT_test"):
    """Error against the distance from the training edge, per frequency (E4 material)."""
    out = {}
    for m in methods:
        pts = collections.defaultdict(list)
        for r in rows:
            if m not in r["m"]:
                continue
            for d, e in zip(r[dkey], r["m"][m]["err_sig"]):
                pts[(r["f_kHz"], int(round(d)))].append(e)
        out[m] = {f"{k[0]}|{k[1]}": float(np.median(v)) for k, v in sorted(pts.items())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--d1", default="results/e2_rows.jsonl"); ap.add_argument("--d3", default="results/e6_rows.jsonl")
    ap.add_argument("--d1supp", default="results/e2_rows_bss_tr.jsonl")
    ap.add_argument("--out", default="results/e2_stats.json"); a = ap.parse_args()
    res = {}
    allm = ["copy", "knn", "gp", "bss", "bss_tr", "bss_best", "bss_gp", "bss_tr_gp", "bss_best_gp", "br", "br_gp", "dtw_or", "str_or"]
    base_list = ["copy", "knn", "gp", "bss", "bss_tr", "bss_best", "bss_gp", "bss_best_gp", "br"]
    cand_list = ["copy", "knn", "gp", "bss", "bss_tr", "bss_gp", "bss_tr_gp", "br", "br_gp"]
    for tag, path, ck, bk in (("D1", a.d1, "split", "split"), ("D3", a.d3, "coupon", "coupon")):
        if not os.path.exists(path):
            continue
        rows = load(path, a.d1supp if tag == "D1" else None)
        res[tag] = {"n_units": len(rows), "summary": summary_table(rows, allm),
                    "ours_vs": compare(rows, "br_gp", base_list, ck),
                    "br_vs": compare(rows, "br", ["bss", "bss_tr", "bss_best", "bss_gp", "bss_best_gp"], ck),
                    "coverage": coverage(rows, bk, methods=("br_gp", "bss_gp", "bss_best_gp", "gp")),
                    "selection_group": selection(rows, cand_list, (ck, "f_kHz")),
                    "selection_unit": selection(rows, cand_list, (ck, "f_kHz", "path"))}
        fr = friedman(rows, [m for m in cand_list if m in rows[0]["m"]], (ck, "f_kHz"))
        res[tag]["friedman"] = {k: v for k, v in fr.items() if k != "task_matrix"}; res[tag]["friedman_posthoc"] = posthoc(fr, "br_gp")
        if tag == "D1":
            res[tag]["law"] = law(rows, ["copy", "gp", "bss", "br", "br_gp"])
            res[tag]["by_freq"] = {f: summary_table([r for r in rows if r["f_kHz"] == f], ["bss", "bss_best", "bss_gp", "br", "br_gp", "gp", "str_or", "dtw_or"]) for f in sorted({r["f_kHz"] for r in rows})}
            res[tag]["by_split"] = {s: summary_table([r for r in rows if r["split"] == s], ["bss", "bss_best", "bss_gp", "br", "br_gp", "gp"]) for s in sorted({r["split"] for r in rows})}
            res[tag]["by_freq_tests"] = {f: compare([r for r in rows if r["f_kHz"] == f], "br_gp", ["bss_best", "bss_best_gp"], ck) for f in sorted({r["f_kHz"] for r in rows}) if "bss_best" in rows[0]["m"]}
            res[tag]["cfg_br_gp"] = dict(collections.Counter(json.dumps(r["cfg"]["br_gp"], sort_keys=True) for r in rows))
    json.dump(res, open(a.out, "w"), indent=1)
    for tag in res:
        print(tag, "units", res[tag]["n_units"])
        for m, v in res[tag]["summary"].items():
            print(f"   {m:8s} mean {v['mean']:.3f} ± {v['std']:.3f}  median {v['median']:.3f}")
        for b, v in res[tag]["ours_vs"].items():
            print(f"   br_gp vs {b:7s} gain {v['median_rel_gain']:+.3f} CI {v['ci95'][0]:+.3f}..{v['ci95'][1]:+.3f} wins {v['wins']}/{v['n_units']} p_holm {v['p_holm']:.2g}")
        for b, v in res[tag]["br_vs"].items():
            print(f"   br    vs {b:7s} gain {v['median_rel_gain']:+.3f} CI {v['ci95'][0]:+.3f}..{v['ci95'][1]:+.3f} wins {v['wins']}/{v['n_units']} p_holm {v['p_holm']:.2g}")
        print("   coverage min block:", {k: round(v["min_block"], 3) for k, v in res[tag]["coverage"].items()})
        print("   selection (group):", {k: round(v["mean_regret"], 4) for k, v in res[tag]["selection_group"]["rules"].items()})
