#!/usr/bin/env python3
"""Diagnostic for Section 4.8 (not part of the recorded test): how visible is the disc of dam_D16 on the 12 fixed paths
when the baseline is a MEASURED intact record at the nearest temperature (no extrapolation)? Positives: dam_D16 signals
against the nearest-temperature signal of the main half; negatives: every second main-half signal against its nearest
other main-half signal. Also the median prediction error of the registration in the AUC test (e13_auc_rows.jsonl).
Output: results/e13_diag.json"""
import json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw
from surrogates import core
from experiments.e2_run import subset_paths
from experiments.e13_auc import MAIN, DAM, R

out = {}
for fk in (60, 80, 100):
    D = ogw.load_freq(fk, root=MAIN); Q = ogw.load_freq(fk, root=DAM)
    S = D["S"].astype(float); S -= S.mean(-1, keepdims=True); T = D["T"]
    SD = Q["S"].astype(float); SD -= SD.mean(-1, keepdims=True); TD = Q["T"]
    au, dp, dn = [], [], []
    for i in subset_paths(D["channels"]):
        pos = [core.rel_err(S[np.argmin(np.abs(T - t)), i][None], SD[j, i][None])[0] for j, t in enumerate(TD)]
        neg = []
        for j in range(0, len(T), 2):
            k = np.argsort(np.abs(T - T[j])); k = k[k != j][0]; neg.append(core.rel_err(S[k, i][None], S[j, i][None])[0])
        au.append(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))); dp.append(np.median(pos)); dn.append(np.median(neg))
    out[str(fk)] = {"auc_median": float(np.median(au)), "auc_max": float(np.max(au)), "di_pos_median": float(np.median(dp)), "di_neg_median": float(np.median(dn))}
rows = [json.loads(l) for l in open(os.path.join(R, "e13_auc_rows.jsonl"))]
band = [r for r in rows if 60 <= r["f_kHz"] <= 100]
out["extrap_di_neg_median_br_gp"] = float(np.median([np.median(r["m"]["br_gp"]["di_neg"]) for r in band]))
out["extrap_di_neg_median_bss_best"] = float(np.median([np.median(r["m"]["bss_best"]["di_neg"]) for r in band]))
json.dump(out, open(os.path.join(R, "e13_diag.json"), "w"), indent=1); print(json.dumps(out, indent=1))
