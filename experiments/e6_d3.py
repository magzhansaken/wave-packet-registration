#!/usr/bin/env python3
"""E6 - transfer to another drift: NASA CFRP fatigue coupons (D3). Condition x = log10(1 + cycles).

Per coupon: 'traction free' measurements only (no load, unclamped), exact duplicates (MD5) removed; cycle levels sorted;
split: training = first half of the levels, one level skipped, test = the remaining levels (extrapolation in cycles).
Validation block = outer 25 % of the training levels; inner part ends one level before it.
Window: from the end of the excitation burst (burst centre from the actuator signal + 2.5 periods) to 0.83 ms.
Packet windows for br/br_gp: there are no sensor coordinates in the NASA description, so packets are the envelope peaks of
the edge reference signal (above 10 % of its maximum) instead of ray arrivals; one label for all windows.
Rows: results/e6_rows.jsonl (unit = coupon x frequency x path)."""
import glob, hashlib, json, os, sys, time
import numpy as np
from scipy.signal import hilbert, find_peaks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from surrogates import core
from experiments.e2_run import GRIDS

SL = os.environ.get("D3_SLICES", "data/d3")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
FS = 1.2e6


def load_coupon(c):
    """Traction-free records of one coupon without exact duplicates, with their cycle levels."""
    fl = sorted(glob.glob(os.path.join(SL, c, "*.npz")))
    S, X, keep, seen, act = [], [], [], set(), None
    for f in fl:
        z = np.load(f)
        if "act0" in z.files:
            act = (z["act0"], z["act_freqs"])
        cond = " ".join(str(z["condition"]).lower().split())
        if cond != "traction free":
            continue
        h = json.loads(str(z["check"]))["md5_raw"]
        if h in seen:
            continue
        seen.add(h)
        S.append(z["sensor"]); X.append(np.log10(1.0 + float(z["cycles"]))); keep.append(os.path.basename(f))
        meta = (z["actuator"], z["sensor_id"], z["frequency_khz"])
    return np.stack(S), np.array(X), keep, meta, act


def burst_centre(act, fk):
    """Time of the envelope maximum of the excitation record at fk."""
    a0, fr = act
    a = a0[list(fr).index(fk)].astype(float)
    return float(np.argmax(np.abs(hilbert(a))) / FS)


def env_packets(t, s, fk):
    """Packet table from envelope peaks (no sensor coordinates are available for D3)."""
    e = np.abs(hilbert(s)); burst = 5.0 / (fk * 1e3)
    pk, _ = find_peaks(e, height=0.1 * e.max(), distance=max(1, int(0.5 * burst * FS)))
    return [("A0", "direct" if n == 0 else "env", 0.0, 0.0, float(t[p])) for n, p in enumerate(pk)]


def default_act(c):
    """Excitation record for a coupon whose slices carry none (L1_S11_F): the record of another coupon of the same layup
    (the NASA programme used one actuation waveform per frequency); recorded in the row as act_source."""
    for d in sorted(os.listdir(SL)):
        if d == c or not d.startswith(c.split("_")[0]) or not os.path.isdir(os.path.join(SL, d)):
            continue
        for f in sorted(glob.glob(os.path.join(SL, d, "*.npz"))):
            z = np.load(f)
            if "act0" in z.files:
                return (z["act0"], z["act_freqs"]), d
    return None, None


def task(c):
    """All surrogates on every path and frequency of one coupon; returns the unit rows."""
    S_all, X, files, (act_id, sen_id, fr), act = load_coupon(c)
    act_source = c
    if act is None:
        act, act_source = default_act(c)
    lev = np.unique(X)
    if len(lev) < 6:
        return []
    k = len(lev) // 2
    tr = X <= lev[k - 1]; te = X >= lev[k + 1]
    trl = lev[:k]; nv = max(1, int(round(0.25 * k)))
    val = tr & (X >= trl[k - nv]); inner = tr & (X <= trl[max(0, k - nv - 2)])
    t_full = np.arange(S_all.shape[2]) / FS
    rows = []
    flist = [int(x) for x in os.environ["D3_FREQS"].split(",")] if os.environ.get("D3_FREQS") else sorted(set(fr.tolist()))
    for fk in flist:
        tc = burst_centre(act, fk); i0 = int((tc + 2.5 / (fk * 1e3)) * FS)
        t = t_full[i0:]
        paths = np.where(fr == fk)[0]
        if os.environ.get("D3_MAXPATHS"):
            paths = paths[: int(os.environ["D3_MAXPATHS"])]
        S = S_all[:, paths, i0:].astype(np.float64); S -= S.mean(-1, keepdims=True)
        side = "up"
        # pooled per-window slopes (one label) from all paths of this coupon and frequency, training block
        # correction of 03.11 (amendment 8): fits evaluated on the validation block use windows and slopes of the inner part
        def pooled(mask, pk):
            """Pooled relative delay slope of all paths of the coupon from the given block."""
            rel = []
            for j in range(len(paths)):
                ie, xr, delta, cen, sl = core.br_laws(t, S[mask, j], X[mask], side, fk, pk[j], {}, lam=0.0, tc=tc)
                rel += [s_ / (cc - tc) for cc, s_ in zip(cen, sl) if cc - tc > 20e-6]
            k0 = float(np.median(rel)) if rel else 0.0
            return {"A0": k0, "pre": k0, "S0": k0}
        pk_all = [env_packets(t, S[np.argmax(np.where(tr, X, -1)), j], fk) for j in range(len(paths))]
        pk_in = [env_packets(t, S[np.argmax(np.where(inner, X, -1)), j], fk) for j in range(len(paths))]
        kappa = pooled(tr, pk_all); kappa_in = pooled(inner, pk_in)
        sub = list(range(0, len(paths), 3))
        best = {}
        for name, grid in GRIDS.items():
            if len(grid) == 1:
                best[name] = grid[0]; continue
            sc = []
            for cfg in grid:
                e = []
                for j in sub:
                    kw = dict(cfg)
                    if name in ("br", "br_gp"):
                        kw.update(fk=fk, packets=pk_in[j], kappa=kappa_in, tc=tc)
                    if name in ("br_gp",) and "lam" in kw and np.isfinite(kw["lam"]) and kw["lam"] > 0:
                        kw["lam"] = kw["lam"] / 1e4            # tau in (log10 cycles)^2: scale of x is ~1 instead of ~10 C
                    P, _ = core.METHODS[name](t, S[inner, j], X[inner], X[val], side, **kw)
                    e.append(np.mean(core.rel_err(P, S[val, j])))
                sc.append(np.median(e))
            best[name] = grid[int(np.argmin(sc))]
        edge_tr = X[tr].max(); edge_in = X[inner].max()
        d_test = np.abs(X[te] - edge_tr); d_cal = np.abs(X[val] - edge_in)
        rng = np.random.default_rng(fk)
        itr = np.where(tr)[0]; hold = np.zeros(len(itr), bool); hold[rng.permutation(len(itr))[: max(2, len(itr) // 5)]] = True
        mhold = np.zeros(len(X), bool); mhold[itr[~hold]] = True; kappa_hold = pooled(mhold, pk_all)
        for j, p in enumerate(paths):
            Str, Ste = S[tr, j], S[te, j]
            row = {"coupon": c, "act_source": act_source, "f_kHz": int(fk), "path": int(j), "act": int(act_id[p]), "sen": int(sen_id[p]),
                   "n_tr": int(tr.sum()), "n_te": int(te.sum()), "levels": int(len(lev)), "kappa": kappa,
                   "cfg": {k_: {kk: (str(v) if isinstance(v, float) and np.isinf(v) else v) for kk, v in cfgv.items()} for k_, cfgv in best.items()},
                   "dX_test": [float(x) for x in d_test], "m": {}, "iv": {}}
            for name in list(core.CANDIDATES) + ["dtw_or", "str_or"]:
                kw = dict(best.get(name, {}))
                if name in ("br", "br_gp"):
                    kw.update(fk=fk, packets=pk_all[j], kappa=kappa, tc=tc)
                    if "lam" in kw and np.isfinite(kw["lam"]) and kw["lam"] > 0:
                        kw["lam"] = kw["lam"] / 1e4
                if name in ("dtw_or", "str_or"):
                    kw["Sq"] = Ste
                want = name in ("gp", "bss_gp", "br_gp")
                if want:
                    kw["want_std"] = True
                t0 = time.time(); P, sd = core.METHODS[name](t, Str, X[tr], X[te], side, **kw); dt = time.time() - t0
                e = core.rel_err(P, Ste)
                m = {"err": float(np.mean(e)), "err_sig": [float(x) for x in e], "cost_s": float(dt)}
                if name in core.CANDIDATES:
                    kw2 = {k_: v for k_, v in kw.items() if k_ not in ("Sq", "want_std")}
                    if name in ("br", "br_gp"):
                        kw2.update(packets=pk_in[j], kappa=kappa_in)
                    Pv, _ = core.METHODS[name](t, S[inner, j], X[inner], X[val], side, **kw2)
                    m["e_stress"] = float(np.mean(core.rel_err(Pv, S[val, j])))
                    if name in ("br", "br_gp"):
                        kw2.update(packets=pk_all[j], kappa=kappa_hold)
                    Ph, _ = core.METHODS[name](t, Str[~hold], X[itr[~hold]], X[itr[hold]], side, **kw2)
                    m["e_in"] = float(np.mean(core.rel_err(Ph, Str[hold])))
                    if want:
                        rms = np.sqrt((Ste ** 2).mean(1))[:, None]; iv = {}
                        if sd is not None and np.all(np.isfinite(sd)):
                            iv["bayes"] = {"cov_sig": [float(x) for x in (np.abs(Ste - P) <= 1.96 * sd).mean(1)], "hw_rel": float(np.mean(1.96 * sd / rms))}
                        s_t, qz, kap, dmean = core.conformal_calibrate(S[val, j] - Pv, d_cal, S[val, j])
                        for lab, law in (("conf", False), ("law", True)):
                            hw = core.conformal_halfwidth(s_t, qz, kap, dmean, d_test, law=law)
                            iv[lab] = {"cov_sig": [float(x) for x in (np.abs(Ste - P) <= hw).mean(1)], "hw_rel": float(np.mean(hw / rms))}
                        row["iv"][name] = iv
                row["m"][name] = m
            rows.append(row)
    return rows


if __name__ == "__main__":
    import multiprocessing as mp
    out = os.path.join(OUT, "e6_rows.jsonl")
    done = {json.loads(l)["coupon"] for l in open(out)} if os.path.exists(out) else set()
    coupons = [c for c in sorted(os.listdir(SL)) if os.path.isdir(os.path.join(SL, c)) and c not in done]
    print(len(coupons), "coupons", flush=True); t0 = time.time()
    with mp.get_context("fork").Pool(int(os.environ.get("WORKERS", 2))) as pool:
        for rows in pool.imap_unordered(task, coupons):
            if not rows:
                continue
            with open(out, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            e = {k: np.mean([r["m"][k]["err"] for r in rows]) for k in rows[0]["m"]}
            cov = np.mean([np.mean(r["iv"]["br_gp"]["law"]["cov_sig"]) for r in rows if "br_gp" in r["iv"]])
            print(f"[{time.time() - t0:.0f}s] {rows[0]['coupon']}: " + " ".join(f"{k}={v:.3f}" for k, v in e.items()) + f" | law cov {cov:.3f}", flush=True)
