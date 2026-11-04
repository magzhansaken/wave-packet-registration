"""A2 diagnosis (post-hoc analysis on the two D1 halves): why does br break on some (split, frequency) tasks?
Edge signal quality, neighbour correlations, tracked lags against the edge versus chained lags."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw
from experiments.e2_run import masks
from surrogates import core

def edge_info(root, fk, split):
    D = ogw.load_freq(fk, root=root); S = D["S"].astype(float); S -= S.mean(-1, keepdims=True); T = D["T"]
    side, tr, te, val, inner = masks(T, split)
    idx = np.where(tr)[0]; Ttr = T[idx]
    order = idx[np.argsort(-Ttr)] if side == "up" else idx[np.argsort(Ttr)]
    ie = order[0]
    def corr(a, b):  # mean over paths of normalised correlation
        num = (S[a] * S[b]).sum(-1); den = np.linalg.norm(S[a], axis=-1) * np.linalg.norm(S[b], axis=-1)
        return float(np.median(num / den))
    nb = [corr(order[k], order[k + 1]) for k in range(min(8, len(order) - 1))]
    return dict(fk=fk, split=split, n_tr=int(tr.sum()), T_edge=float(T[ie]), dTs_edge=float(np.ptp(D["Ts"][ie])),
                T_next=[round(float(T[o]), 2) for o in order[1:5]], ts_edge=float(D["ts"][ie]),
                corr_edge_next=[round(c, 5) for c in nb[:4]], corr_neigh_med=round(float(np.median(nb[1:])), 5))

if __name__ == "__main__":
    out = []
    for half, root in (("main", "data/d1"), ("reserve", "data/d1_reserve")):
        for fk in (40, 60, 140, 200):
            for sp in ("S1_le30_ge35", "S2_le35_ge40", "S3_le40_ge45", "S4_ge40_le35", "S5_ge45_le40", "S6_ge50_le45"):
                r = edge_info(root, fk, sp); r["half"] = half; out.append(r); print(json.dumps(r), flush=True)
