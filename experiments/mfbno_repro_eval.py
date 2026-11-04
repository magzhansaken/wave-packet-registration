#!/usr/bin/env python3
"""Appendix B: reproduction of MF-B-NO on its authors' data (You et al. 2026), CPU only.
Run 1: authors' training loop logic, checkpoint by test loss (as in the published code) and last epoch, 1 seed,
20 epochs x 379 optimizer steps (64 % of the authors' ~11,800 steps), 20 MC passes.
Run 2: checkpoint by a validation split of 4 of the 20 training designs, step-matched budget (39 epochs x 303 = 11,817
steps), 100 MC passes, 1 thread; MF-B-NO seed 1 and the HF-only model evaluated on the test set (the test evaluation of seed 2 did not finish).
Reads the run folders of baselines/mfbno_run; writes results/mfbno_repro_eval.json."""
import csv, json, os
B = os.environ.get("MFB", "baselines/mfbno_authors/MF-B-NO_forSpinodal")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "mfbno_repro_eval.json")
j = lambda p: json.load(open(os.path.join(B, p)))
def hist(p):
    p = os.path.join(B, p)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
s4m, s4h = j("out_step/repro_summary.json"), j("out_step_hf/repro_summary.json")
o = {"paper_reduction_pct": 84.1, "paper_steps": 11800, "paper_seeds": 5, "paper_mc": 1000,
     "s4": {"seeds": 1, "epochs": s4m["budget"]["epochs"], "steps": s4m["budget"]["epochs"] * s4m["budget"]["optimizer_steps_per_epoch"],
            "mc": s4m["budget"]["mc_samples"], "lf_mse": s4m["test_mse_LF_only"],
            "mf_test_by_test": s4m["mf"]["best_by_test"]["test_mse_mc_mean"], "mf_test_last": s4m["mf"]["last"]["test_mse_mc_mean"],
            "hf_test_by_test": s4h["hf"]["best_by_test"]["test_mse_mc_mean"], "hf_test_last": s4h["hf"]["last"]["test_mse_mc_mean"],
            "mf_cov_last": s4m["mf"]["last"]["coverage95"], "hf_cov_last": s4h["hf"]["last"]["coverage95"],
            "cpu_s_mf": s4m["seconds_total"], "cpu_s_hf": s4h["seconds_total"]}}
o["s4"]["steps_share_pct"] = 100 * o["s4"]["steps"] / o["paper_steps"]
o["s4"]["red_by_test_pct"] = 100 * (1 - o["s4"]["mf_test_by_test"] / o["s4"]["hf_test_by_test"])
o["s4"]["red_last_pct"] = 100 * (1 - o["s4"]["mf_test_last"] / o["s4"]["hf_test_last"])
s5 = j("out_val/repro_summary_seed1.json"); hm, hh = hist("out_val/repro_mf_history.csv"), hist("out_val/repro_hf_history.csv")
o["s5"] = {"seeds_done_mf": 1, "epochs": s5["budget"]["epochs"], "steps": s5["budget"]["optimizer_steps_total"], "mc": s5["budget"]["mc_samples"],
           "val_designs": len(s5["budget"]["validation_designs"]), "mf_test_by_val": s5["mf"]["best_by_val"]["test_mse_mc_mean"],
           "mf_cov_by_val": s5["mf"]["best_by_val"]["coverage95"], "mf_best_epoch": s5["mf"]["best_epoch_by_val"],
           "mf_val_last": float(hm[-1]["val_mse_eval_mode"]), "hf_val_last": float(hh[-1]["val_mse_eval_mode"]) if hh else None,
           "hf_epochs_done": len(hh), "cpu_s_mf_train": sum(float(r["sec"]) for r in hm), "cpu_s_hf_train": sum(float(r["sec"]) for r in hh),
           "hf_test": s5.get("hf", {}).get("best_by_val", {}).get("test_mse_mc_mean"),
           "red_test_pct": 100 * s5["reduction_vs_HFonly_best_by_val"] if "reduction_vs_HFonly_best_by_val" in s5 else None}
o["s5"]["red_val_pct"] = 100 * (1 - o["s5"]["mf_val_last"] / o["s5"]["hf_val_last"])
o["s5"]["steps_share_pct"] = 100 * o["s5"]["steps"] / o["paper_steps"]
json.dump(o, open(OUT, "w"), indent=1); print(json.dumps(o, indent=1))
