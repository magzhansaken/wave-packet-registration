#!/usr/bin/env python3
"""Integrity checks of dam_D16 (OGW #1 with the reversible damage at D16; amendment 11) before it is used: file counts per frequency against the plan, identical
time stamps across frequencies, no overlap with either half of D1 (udam) (time stamps and MD5 of the raw catch arrays), NaN/Inf,
plate-temperature range against D1; cycle structure (direction changes) and block sizes of S1-S6 for the inclusion rule of
amendment 11. No model is fitted. Output: results/data_checks_dam_D16.json."""
import glob, json, os, sys
import numpy as np

RES = os.environ.get("D1_DAM", "data/d1_dam")
DEVS = ["data/d1", "data/d1_reserve"]
PLAN = "data/inventory/d1_dam_plan.json"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "data_checks_dam_D16.json")
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
    """Compare dam_D16 with the plan and with both halves of D1; write the report."""
    plan = json.load(open(PLAN)); want = set(plan["stamps"])
    r = scan(RES); ds = [scan(p) for p in DEVS]
    d = {fk: {"stamps": sum((x[fk]["stamps"] for x in ds), []), "md5": sum((x[fk]["md5"] for x in ds), []), "T": sum((x[fk]["T"] for x in ds), [])} for fk in FREQS}
    rep = {"per_freq": {}, "ok": True}
    for fk in FREQS:
        rs, ds = set(r[fk]["stamps"]), set(d[fk]["stamps"])
        e = {"n_dam": len(rs), "missing_vs_plan": sorted(want - rs), "extra_vs_plan": sorted(rs - want),
             "stamp_overlap_with_udam": len(rs & ds), "md5_overlap_with_udam": len(set(r[fk]["md5"]) & set(d[fk]["md5"])),
             "md5_duplicates_within": len(r[fk]["md5"]) - len(set(r[fk]["md5"])), "nonfinite": r[fk]["nonfinite"],
             "T_range_dam": [min(r[fk]["T"]), max(r[fk]["T"])] if r[fk]["T"] else None,
             "T_range_udam": [min(d[fk]["T"]), max(d[fk]["T"])] if d[fk]["T"] else None}
        e["ok"] = (not e["missing_vs_plan"]) and not e["extra_vs_plan"] and e["stamp_overlap_with_udam"] == 0 and e["md5_overlap_with_udam"] == 0 \
            and e["md5_duplicates_within"] == 0 and e["nonfinite"] == 0
        rep["per_freq"][fk] = e; rep["ok"] &= e["ok"]
    stamps = [set(r[fk]["stamps"]) for fk in FREQS]
    rep["same_stamps_all_freqs"] = all(s == stamps[0] for s in stamps)
    rep["ok"] &= rep["same_stamps_all_freqs"]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from experiments.e2_run import masks, SPLITS
    T = np.array(r[40]["T"]); ts = [s for s in r[40]["stamps"]]
    o = np.argsort(ts); Tt = T[o]; turns = []; dirn = 0; ext = Tt[0]
    for v in Tt:                                     # direction changes with a 2 C hysteresis (sensor noise ignored)
        if dirn == 0:
            if abs(v - Tt[0]) > 2.0:
                dirn = 1 if v > Tt[0] else -1; ext = v
        elif dirn * (v - ext) > 0:
            ext = v
        elif abs(v - ext) > 2.0:
            turns.append(round(float(ext), 2)); dirn = -dirn; ext = v
    rep["n_direction_changes"] = len(turns); rep["turning_T"] = turns
    rep["splits"] = {}
    for sp in SPLITS:
        side, tr, te, val, inner = masks(T, sp)
        n = dict(n_tr=int(tr.sum()), n_te=int(te.sum()), n_val=int(val.sum()), n_inner=int(inner.sum()))
        n["included"] = n["n_tr"] >= 10 and n["n_te"] >= 5 and n["n_val"] >= 3 and n["n_inner"] >= 3
        rep["splits"][sp] = n
    json.dump(rep, open(OUT, "w"), indent=1)
    print(json.dumps({k: (v if k != "per_freq" else {f: {kk: vv for kk, vv in e.items() if kk in ("n_dam", "ok", "T_range_dam")} for f, e in v.items()}) for k, v in rep.items()}, indent=0))


if __name__ == "__main__":
    main()
