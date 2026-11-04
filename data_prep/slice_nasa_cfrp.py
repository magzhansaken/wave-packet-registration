#!/usr/bin/env python3
"""Slice NASA CFRP Composites (Saxena, Goebel, Larrosa, Chang; NASA Ames PCoE / Stanford SACL) PZT records.

Input : raw files <root>/2. Composites/Layup*/<coupon>/PZT*data/*.mat  (struct 'coupon')
Output: <out>/d3/<coupon>/<file>.npz with
        sensor  float32 (252, 1000)  received signals, first 1000 samples (0.83 ms at 1.2 MHz), path order of the file
        actuator, sensor_id, frequency_khz, amplitude, gain   int arrays (252,)
        cycles, load, condition, comment, fs
        check   dict: nan, inf, absmax_raw, md5_raw, n_paths, n_samples
        act0    float32 (n_freq, 2000)  actuator signals per frequency (only in the first file of a coupon)
Raw .mat deleted after a verified write (--delete)."""
import argparse, glob, hashlib, json, os, sys
import numpy as np, scipy.io as sio
NS = 1000

def scal(x):
    """Scalar field of the coupon struct -> float; empty field (stored as [] in some files, e.g. L1_S18_F) -> NaN."""
    a = np.asarray(x, dtype=float).ravel()
    return float(a[0]) if a.size == 1 else (float("nan") if a.size == 0 else float(a.mean()))

def slice_file(path, out_root, keep_act):
    """Slice one NASA coupon file: sensor signals, path metadata and checksums."""
    c = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["coupon"]
    pd = np.atleast_1d(c.path_data)
    S = np.stack([np.asarray(p.signal_sensor, dtype=np.float64) for p in pd])
    meta = {k: np.array([int(getattr(p, k)) for p in pd]) for k in ["actuator", "sensor", "frequency", "amplitude", "gain"]}
    fs = int(pd[0].sampling_rate)
    chk = {"nan": int(np.isnan(S).sum()), "inf": int(np.isinf(S).sum()), "absmax_raw": float(np.nanmax(np.abs(S))),
           "md5_raw": hashlib.md5(S.tobytes()).hexdigest(), "n_paths": int(S.shape[0]), "n_samples": int(S.shape[1])}
    coupon = path.replace("\\", "/").split("/")[-3]; od = os.path.join(out_root, "d3", coupon); os.makedirs(od, exist_ok=True)
    op = os.path.join(od, os.path.basename(path).replace(".mat", ".npz"))
    extra = {}
    if keep_act:
        fr = sorted(set(meta["frequency"].tolist()))
        extra["act0"] = np.stack([np.asarray(pd[int(np.where(meta["frequency"] == f)[0][0])].signal_actuator, dtype=np.float32) for f in fr])
        extra["act_freqs"] = np.array(fr)
    np.savez_compressed(op, sensor=S[:, :NS].astype(np.float32), actuator=meta["actuator"], sensor_id=meta["sensor"],
                        frequency_khz=meta["frequency"], amplitude=meta["amplitude"], gain=meta["gain"],
                        cycles=scal(c.cycles), load=scal(c.load),
                        condition=str(c.condition), comment=str(c.comment) if np.size(c.comment) else '', fs=fs, check=json.dumps(chk), **extra)
    z = np.load(op); assert z["sensor"].shape == (S.shape[0], NS)
    return op, chk

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--batch", help="JSON list of device paths: process only these files (safe for parallel runs)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, "2. Composites", "Layup*", "*", "PZT*data", "*.mat")))
    if a.batch:
        PFX = os.environ.get("RAW_PREFIX", "")
        want = [os.path.join(a.root, p[len(PFX):].replace("\\", "/")) for p in json.load(open(a.batch))]
        files = [p for p in want if os.path.exists(p)]
        missing = [p for p in want if not os.path.exists(p)]
        if missing: print("not staged:", len(missing), file=sys.stderr)
    logp = os.path.join(a.out, "d3_slice_log.jsonl"); n = 0
    for p in files:
        coupon = p.replace("\\", "/").split("/")[-3]; flag = os.path.join(a.out, "d3", coupon, "_act_saved")
        try:
            op, chk = slice_file(p, a.out, not os.path.exists(flag))
            if not os.path.exists(flag): open(flag, "w").write(os.path.basename(op))
            with open(logp, "a") as fh: fh.write(json.dumps({"raw": p.split("data/")[-1], "slice": op.split(a.out)[-1], **chk}, ensure_ascii=False) + "\n")
            if a.delete: os.remove(p)
            n += 1
        except Exception as e:
            with open(logp, "a") as fh: fh.write(json.dumps({"raw": p, "error": repr(e)}, ensure_ascii=False) + "\n")
            print("ERROR", p, e, file=sys.stderr)
    print(f"sliced {n} of {len(files)} files")
