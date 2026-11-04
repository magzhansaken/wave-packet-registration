#!/usr/bin/env python3
"""Checks of the sliced data before any experiment (docs/PROTOCOL.md, section 4): sums, ranges, duplicates, gaps,
agreement on the overlap with earlier sources. Writes results/data_checks.json and prints a summary.

D1 OGW #1: per-frequency file counts against the selection plan, identical time stamps across frequencies, NaN/Inf,
MD5 duplicates of the raw catch arrays, ADC range (clipping), channel order, plate-temperature range, the two sensors'
difference, setpoint vs plate temperature, heating/cooling cycles; overlap: (a) the 40 kHz pair-1 subset used in the pilot analysis
(lorenzomie repository, normalised) - waveform correlation at matching temperatures, (b) A0 velocity and its thermal slope
on T4-T10 against the authors (1166 -> 1133 m/s, -0.825 m/s/C), (c) the global stretch law of the pilot analysis (c = -4.9e-4 1/C).
D3 NASA CFRP: coupons, files per coupon, cycle levels, conditions, paths and frequencies per file, NaN/Inf, MD5 duplicates.
"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bcs import ogw

SL = os.environ.get("SLICES", "data")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
PLAN = os.environ.get("D1_PLAN", "data/inventory/d1_plan.json")
D0 = os.environ.get("D0", "data/d0/ogw40k_pair1_standard.npz")


def last_by_raw(logp):
    """Last log entry per raw file of a slicing log."""
    rec = {}
    for line in open(logp):
        r = json.loads(line)
        key = os.path.basename(os.path.dirname(r["raw"])) + "/" + os.path.basename(r["raw"]) if "Composites" not in r["raw"] else r["raw"].split("/")[-1]
        rec[key] = r
    return rec


def check_d1():
    """Integrity checks of the D1 slices."""
    res = {}
    plan = json.load(open(PLAN)) if os.path.exists(PLAN) else None
    log = last_by_raw(os.path.join(SL, "d1_slice_log.jsonl"))
    ok = {k: v for k, v in log.items() if "error" not in v}
    res["log_entries"] = len(log); res["log_errors"] = sum("error" in v for v in log.values())
    per_f = collections.defaultdict(set)
    for k in ok:
        stamp, fn = k.split("/"); per_f[fn.replace("pc_f", "").replace("kHz.h5", "")].add(stamp)
    res["files_per_freq"] = {f: len(per_f[str(f)]) for f in ogw.FREQS}
    if plan:
        sel = set(plan["selected"]) if isinstance(plan["selected"][0], str) else None
        res["planned_per_freq"] = len(plan["selected"])
        if sel:
            res["missing_per_freq"] = {f: sorted(sel - per_f[str(f)])[:5] + (["..."] if len(sel - per_f[str(f)]) > 5 else []) for f in ogw.FREQS}
    stamps = [per_f[str(f)] for f in ogw.FREQS]
    common = set.intersection(*stamps) if stamps else set()
    res["timestamps_common_to_all_12"] = len(common)
    res["nan_total"] = int(sum(v["nan"] for v in ok.values())); res["inf_total"] = int(sum(v["inf"] for v in ok.values()))
    md5 = collections.Counter(v["md5_catch_raw"] for v in ok.values())
    res["md5_duplicates"] = int(sum(c - 1 for c in md5.values() if c > 1))
    am = np.array([v["absmax_raw"] for v in ok.values()])
    res["absmax_raw_V"] = {"min": float(am.min()), "median": float(np.median(am)), "max": float(am.max())}
    res["n_samples_raw"] = sorted({v["n_samples"] for v in ok.values()}); res["n_paths"] = sorted({v["n_paths"] for v in ok.values()})
    # per-file metadata from slices
    T, dT12, Tc, chan_ok, clip = [], [], [], True, 0
    ch0 = None
    meta = {}
    for f in ogw.FREQS:
        for p in sorted(x for x in glob.glob(os.path.join(SL, "d1", f"{f}kHz", "*.npz")) if not os.path.basename(x).startswith("_")):
            d = np.load(p)
            if ch0 is None:
                ch0 = d["channels"]
            chan_ok &= bool(np.array_equal(d["channels"], ch0))
            st = os.path.basename(p)[:-4]
            meta.setdefault(st, []).append((float(np.mean(d["T_plate"])), float(np.diff(d["T_plate"])[0]), float(d["T_ctc"]), float(d["timestamp"])))
    for st, v in meta.items():
        a = np.array(v); T.append(a[0, 0]); dT12.append(a[0, 1]); Tc.append(a[0, 2])
    T, dT12, Tc = np.array(T), np.array(dT12), np.array(Tc)
    stamps_sorted = sorted(meta); ts = np.array([meta[s][0][3] for s in stamps_sorted]); Ts = np.array([meta[s][0][0] for s in stamps_sorted])
    dTs = np.diff(Ts); turn = int(np.sum(np.diff(np.sign(dTs[np.abs(dTs) > 0.05])) != 0))
    res["channels_identical"] = chan_ok
    res["T_plate_mean_C"] = {"min": float(T.min()), "max": float(T.max()), "n_measurements": int(len(T))}
    res["T_sensor_difference_C"] = {"median_abs": float(np.median(np.abs(dT12))), "p95_abs": float(np.percentile(np.abs(dT12), 95)), "max_abs": float(np.abs(dT12).max())}
    res["T_plate_minus_setpoint_C"] = {"median": float(np.median(T - Tc)), "p95_abs": float(np.percentile(np.abs(T - Tc), 95))}
    res["T_direction_changes"] = turn
    res["T_spacing_C"] = {"median_abs_step_sorted": float(np.median(np.diff(np.sort(T))))}
    # same measurement: are the 12 frequencies of one time stamp at the same temperature? (sanity of the merge)
    spread = [np.ptp([x[0] for x in v]) for v in meta.values() if len(v) == 12]
    res["T_spread_within_timestamp_C"] = float(np.max(spread)) if spread else None
    return res, meta


def overlap_d1(meta):
    """Agreement of D1 with earlier sources on their overlap."""
    res = {}
    D = ogw.load_freq(40)
    # (a) pilot subset: 40 kHz, pair 1 (first channel row), amplitudes normalised by the repository authors
    if os.path.exists(D0):
        z = np.load(D0); S0 = z["signals"].astype(float); T0 = z["temperature"].astype(float); t0 = z["t"].astype(float)
        i = 0                                               # path (0, 1)
        corr, dTm = [], []
        for n in range(len(D["T"])):
            j = int(np.argmin(np.abs(T0 - D["T"][n]))); dTm.append(abs(T0[j] - D["T"][n]))
            a = np.interp(D["t"], t0, S0[j] - S0[j].mean()); b = D["S"][n, i].astype(float); b = b - b.mean()
            corr.append(float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))
        res["pilot_subset_correlation"] = {"median": float(np.median(corr)), "min": float(np.min(corr)), "max_T_mismatch_C": float(np.max(dTm))}
    # (b) A0 on T4-T10 (indices 3, 9): group velocity at the coldest file and thermal slope of the time of flight
    ch = D["channels"]; i = int([k for k in range(len(ch)) if set(ch[k].tolist()) == {3, 9}][0])
    L = 0.44; tc = 2.5 / 40e3
    o = np.argsort(D["T"]); ref = o[0]
    tp, _ = ogw.pick_peak(D["t"], D["S"][ref, i].astype(float), tc + L / 1600, tc + L / 900)
    lags = [0.0]; prev = 0.0
    w = (D["t"] >= tp - 80e-6) & (D["t"] <= tp + 80e-6)
    x = D["S"][ref, i].astype(float)
    for n in o[1:]:
        cands = prev + np.linspace(-10e-6, 10e-6, 201)
        vals = [np.dot(np.interp(D["t"][w] + c, D["t"], D["S"][n, i].astype(float)), x[w]) for c in cands]
        prev = cands[int(np.argmax(vals))]; lags.append(prev)
    lags = np.array(lags); Tsorted = D["T"][o]
    sl = np.polyfit(Tsorted, lags, 1)[0]
    v_ref = L / (tp - tc)
    res["A0_T4_T10"] = {"T_ref_C": float(D["T"][ref]), "group_velocity_ref_m_s": float(v_ref), "tof_slope_us_per_C": float(sl * 1e6),
                        "velocity_slope_m_s_per_C": float(-v_ref ** 2 / L * sl),
                        "authors": {"velocity_20C_m_s": 1166, "velocity_60C_m_s": 1133, "slope_m_s_per_C": -0.825, "tof_slope_us_per_C": 0.2749}}
    # (c) global stretch law on pair 1 at 40 kHz (pilot analysis: c = -4.9e-4 1/C)
    from scipy.signal import hilbert  # noqa
    i = 0; t = D["t"]; win = (t >= 80e-6) & (t <= 1.0e-3)
    ks = np.linspace(0.97, 1.03, 601); ref_s = D["S"][ref, i].astype(float)
    kk = []
    for n in range(len(D["T"])):
        y = D["S"][n, i].astype(float)[win]
        best = max(ks, key=lambda k: np.dot(np.interp(t[win] * k, t, ref_s), y) / (np.linalg.norm(np.interp(t[win] * k, t, ref_s)) + 1e-30))
        kk.append(best)
    c = np.polyfit(D["T"] - D["T"][ref], np.array(kk) - 1, 1)[0]
    res["global_stretch_pair1_40k"] = {"c_per_C": float(c), "pilot_c_per_C": -4.9e-4}
    return res


def check_d3():
    """Integrity checks of the D3 slices."""
    res = {}
    log = last_by_raw(os.path.join(SL, "d3_slice_log.jsonl"))
    ok = {k: v for k, v in log.items() if "error" not in v}
    res["log_entries_unique_raw"] = len(log); res["errors_remaining"] = sum("error" in v for v in log.values())
    res["nan_total"] = int(sum(v["nan"] for v in ok.values())); res["inf_total"] = int(sum(v["inf"] for v in ok.values()))
    md5 = collections.Counter(v["md5_raw"] for v in ok.values())
    res["md5_duplicates"] = int(sum(c - 1 for c in md5.values() if c > 1))
    res["paths_per_file"] = sorted({v["n_paths"] for v in ok.values()}); res["samples_per_file"] = sorted({v["n_samples"] for v in ok.values()})
    coupons = {}
    for d in sorted(glob.glob(os.path.join(SL, "d3", "*"))):
        cname = os.path.basename(d); files = sorted(glob.glob(os.path.join(d, "*.npz")))
        cyc, cond, freqs, fs = [], collections.Counter(), set(), set()
        for p in files:
            z = np.load(p, allow_pickle=False)
            cyc.append(float(z["cycles"])); cond[str(z["condition"])] += 1
            freqs |= set(np.unique(z["frequency_khz"]).tolist()); fs.add(int(z["fs"]))
        cyc = np.array(cyc)
        coupons[cname] = {"files": len(files), "cycle_levels": int(len(np.unique(cyc))), "cycles_max": float(np.nanmax(cyc)),
                          "conditions": dict(cond), "freqs_kHz": sorted(freqs), "fs_Hz": sorted(fs),
                          "files_per_level_max": int(max(collections.Counter(cyc.tolist()).values()))}
    res["coupons"] = coupons
    res["total_files"] = int(sum(c["files"] for c in coupons.values()))
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = {}
    out["D1"], meta = check_d1()
    print(json.dumps(out["D1"], indent=1, ensure_ascii=False)[:3000], flush=True)
    out["D1_overlap"] = overlap_d1(meta)
    print(json.dumps(out["D1_overlap"], indent=1), flush=True)
    out["D3"] = check_d3()
    print(json.dumps({k: v for k, v in out["D3"].items() if k != "coupons"}, indent=1), flush=True)
    for c, v in out["D3"]["coupons"].items():
        print(c, v)
    json.dump(out, open(os.path.join(OUT, "data_checks.json"), "w"), indent=1, ensure_ascii=False)
