#!/usr/bin/env python3
"""Variant B (amendment 10): verdict on the reserve half of D1. Reads, for one TAG (default reserve):
  results/e2_rows_<tag>.jsonl   (copy, bss, bss_tr, br, br_gp with the corrected protocol; bss_best chosen by validation)
  results/b_phys_rows_<tag>.jsonl  (physics index E_phys per path at d_ref = 15 C)
  results/e4b_rows_<tag>.jsonl     (training-data index e_mix per path, at d_ref and averaged over the test distances)
S1: Spearman rho over units between each index and the measured excess of stretch (bss_best - br_gp).
S2: task regret (mean error of the chosen method minus the better of br_gp and bss_best, over the 66 paths) for
    R_phys (median E_phys > theta), R_train (median e_mix_dref > theta), stress-block rule on {br_gp, bss_best},
    always br_gp, always bss_best; theta = 0.10 (decision) and 0.05 / 0.20 (sensitivity); paired Wilcoxon over tasks, Holm.
Output: results/b_eval_<tag>.json"""
import collections, json, os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import analyze as A

R = os.path.join(os.path.dirname(__file__), "..", "results")
TAG = os.environ.get("TAG", "reserve")
THETA, SENS = 0.10, (0.05, 0.20)


def main():
    rows = [json.loads(l) for l in open(os.path.join(R, f"e2_rows_{TAG}.jsonl"))]
    A.add_best_bss(rows)
    ph = {(r["split"], r["f_kHz"], r["path"]): r for r in (json.loads(l) for l in open(os.path.join(R, f"b_phys_rows_{TAG}.jsonl")))}
    tr = {(r["split"], r["f_kHz"], r["path"]): r for r in (json.loads(l) for l in open(os.path.join(R, f"e4b_rows_{TAG}.jsonl")))}
    key = lambda r: (r["split"], r["f_kHz"], r["path"])
    U = [r for r in rows if key(r) in ph and key(r) in tr]
    exc = np.array([r["m"]["bss_best"]["err"] - r["m"]["br_gp"]["err"] for r in U])
    Ep = np.array([ph[key(r)]["E_phys"]["15.0"] for r in U]); Et = np.array([tr[key(r)]["e_mix_dref"] for r in U])
    Etm = np.array([tr[key(r)]["e_mix"] for r in U])
    ok = np.isfinite(Ep) & np.isfinite(Et)
    s1 = {"n_units": int(ok.sum()),
          "rho_phys": float(spearmanr(Ep[ok], exc[ok]).correlation),
          "rho_train_dref": float(spearmanr(Et[ok], exc[ok]).correlation),
          "rho_train_mean_over_test_d": float(spearmanr(Etm[ok], exc[ok]).correlation)}
    s1["S1_pass"] = bool(s1["rho_phys"] >= s1["rho_train_dref"] and s1["rho_phys"] >= 0.53)
    # tasks
    G = collections.defaultdict(list)
    for r in U:
        G[(r["split"], r["f_kHz"])].append(r)
    tasks = []
    for g, rs in sorted(G.items()):
        e_br = np.mean([r["m"]["br_gp"]["err"] for r in rs]); e_bs = np.mean([r["m"]["bss_best"]["err"] for r in rs])
        v_br = np.median([r["m"]["br_gp"]["e_stress"] for r in rs])      # as R3 in analyze.selection: median over the task's units
        v_bs = np.median([r["m"]["bss_best"]["e_stress"] for r in rs])
        tasks.append({"split": g[0], "f_kHz": g[1], "err_br_gp": float(e_br), "err_bss_best": float(e_bs), "best": float(min(e_br, e_bs)),
                      "I_phys": float(np.nanmedian([ph[key(r)]["E_phys"]["15.0"] for r in rs])),
                      "I_train": float(np.nanmedian([tr[key(r)]["e_mix_dref"] for r in rs])),
                      "stress_pick": "br_gp" if v_br <= v_bs else "bss_best"})
    def regrets(rule):
        out = []
        for t in tasks:
            pick = rule(t); e = t["err_br_gp"] if pick == "br_gp" else t["err_bss_best"]; out.append(e - t["best"])
        return np.array(out)
    rules = {}
    for th in (THETA,) + SENS:
        rules[f"phys_theta{th}"] = regrets(lambda t, th=th: "br_gp" if t["I_phys"] > th else "bss_best")
        rules[f"train_theta{th}"] = regrets(lambda t, th=th: "br_gp" if t["I_train"] > th else "bss_best")
    rules["stress"] = regrets(lambda t: t["stress_pick"])
    rules["always_br_gp"] = regrets(lambda t: "br_gp"); rules["always_bss_best"] = regrets(lambda t: "bss_best")
    main_ = rules[f"phys_theta{THETA}"]; comp = ["stress", f"train_theta{THETA}", "always_br_gp", "always_bss_best"]
    ps = [float(wilcoxon(main_, rules[c]).pvalue) if np.any(main_ != rules[c]) else 1.0 for c in comp]
    s2 = {"n_tasks": len(tasks), "mean_regret": {k: float(v.mean()) for k, v in rules.items()},
          "zero_regret_share": {k: float((v < 1e-12).mean()) for k, v in rules.items()},
          "wilcoxon_phys_vs": {c: {"p": p, "p_holm": float(ph_)} for c, p, ph_ in zip(comp, ps, A.holm(ps))},
          "phys_picks_registration": int(sum(t["I_phys"] > THETA for t in tasks)),
          "train_picks_registration": int(sum(t["I_train"] > THETA for t in tasks))}
    s2["S2_pass"] = bool(s2["mean_regret"][f"phys_theta{THETA}"] <= s2["mean_regret"]["stress"] and
                         s2["mean_regret"][f"phys_theta{THETA}"] <= s2["mean_regret"][f"train_theta{THETA}"])
    verdict = "success" if (s1["S1_pass"] and s2["S2_pass"]) else ("partial" if (s1["S1_pass"] or s2["S2_pass"]) else "failure")
    by_f = {}
    for fk in sorted({t["f_kHz"] for t in tasks}):
        tt = [t for t in tasks if t["f_kHz"] == fk]
        by_f[fk] = {"I_phys": float(np.median([t["I_phys"] for t in tt])), "I_train": float(np.median([t["I_train"] for t in tt])),
                    "err_br_gp": float(np.median([t["err_br_gp"] for t in tt])), "err_bss_best": float(np.median([t["err_bss_best"] for t in tt]))}
    # third independent replication of the main comparison
    rep = A.compare(rows, "br_gp", ["bss_best", "bss", "bss_tr", "br", "copy"], "split")
    out = {"tag": TAG, "S1": s1, "S2": s2, "verdict": verdict, "by_freq": by_f, "tasks": tasks, "replication_main_claim": rep,
           "replication_by_freq": {fk: A.compare([r for r in rows if r["f_kHz"] == fk], "br_gp", ["bss_best"], "split")["bss_best"]
                                   for fk in sorted({r["f_kHz"] for r in rows})},
           "coverage": A.coverage(rows, "split", methods=("br_gp",))}
    json.dump(out, open(os.path.join(R, f"b_eval_{TAG}.json"), "w"), indent=1)
    print(json.dumps({"S1": s1, "S2": {k: v for k, v in s2.items()}, "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
