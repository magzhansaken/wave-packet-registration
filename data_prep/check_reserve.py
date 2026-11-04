#!/usr/bin/env python3
"""Checks of the reserve half of D1 before it is used: file counts per frequency against the plan, identical
time stamps across frequencies, no overlap with the development half (time stamps and MD5 of the raw catch arrays), NaN/Inf,
plate-temperature range against the development half. Output: results/data_checks_reserve.json."""
import glob, json, os, sys
import numpy as np

RES = os.environ.get("D1_RESERVE", "data/d1_reserve")
DEV = os.environ.get("D1_DEV", "data/d1")
PLAN = "data/inventory/d1_reserve_plan.json"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "data_checks_reserve.json")
FREQS = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]


def scan(root):
    """Time stamps, checksums, temperatures and non-finite counts of all slices under root."""
    out = {}
    for fk in FREQS:
        fl = sorted(p for p in glob.glob(os.path.join(root, f"{fk}kHz", "*.npz")) if not os.path.basename(p).startswith("_"))
        st, md5, T, bad = [], [], [], 0
        for p in fl:
            z = np.load(p); c = z["catch"]
            bad += int((~np.isfinite(c)).sum()); st.append(os.path.basename(p)[:-4])
            md5.append(json.loads(str(z["check"]))["md5_catch_raw"]); T.append(float(np.mean(z["T_plate"])))
        out[fk] = {"stamps": st, "md5": md5, "T": T, "nonfinite": bad}
    return out


def main():
    """Compare the reserve half with the plan and with the development half; write the report."""
    plan = json.load(open(PLAN)); want = set(plan["reserve_timestamps"])
    r, d = scan(RES), scan(DEV)
    rep = {"per_freq": {}, "ok": True}
    for fk in FREQS:
        rs, ds = set(r[fk]["stamps"]), set(d[fk]["stamps"])
        e = {"n_reserve": len(rs), "missing_vs_plan": sorted(want - rs), "extra_vs_plan": sorted(rs - want),
             "stamp_overlap_with_dev": len(rs & ds), "md5_overlap_with_dev": len(set(r[fk]["md5"]) & set(d[fk]["md5"])),
             "md5_duplicates_within": len(r[fk]["md5"]) - len(set(r[fk]["md5"])), "nonfinite": r[fk]["nonfinite"],
             "T_range_reserve": [min(r[fk]["T"]), max(r[fk]["T"])] if r[fk]["T"] else None,
             "T_range_dev": [min(d[fk]["T"]), max(d[fk]["T"])] if d[fk]["T"] else None}
        e["ok"] = (not e["missing_vs_plan"]) and not e["extra_vs_plan"] and e["stamp_overlap_with_dev"] == 0 and e["md5_overlap_with_dev"] == 0 \
            and e["md5_duplicates_within"] == 0 and e["nonfinite"] == 0
        rep["per_freq"][fk] = e; rep["ok"] &= e["ok"]
    stamps = [set(r[fk]["stamps"]) for fk in FREQS]
    rep["same_stamps_all_freqs"] = all(s == stamps[0] for s in stamps)
    rep["ok"] &= rep["same_stamps_all_freqs"]
    json.dump(rep, open(OUT, "w"), indent=1)
    print(json.dumps({k: (v if k != "per_freq" else {f: {kk: vv for kk, vv in e.items() if kk in ("n_reserve", "ok", "T_range_reserve")} for f, e in v.items()}) for k, v in rep.items()}, indent=0))


if __name__ == "__main__":
    main()
