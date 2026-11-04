#!/usr/bin/env python3
"""Slice Open Guided Waves #1 (OGW_CFRP_Temperature_udam, CC0, doi 10.6084/m9.figshare.8108297).

Input : raw files  <root>/OGW_CFRP_Temperature_udam/<timestamp>/pc_f<freq>kHz.h5   (read-only copies)
Output: <out>/d1/<freq>kHz/<timestamp>.npz  with
        catch   float32 (66, 2300)   received signals, window 80 us - 1.0 ms, decimated 10 MHz -> 2.5 MHz (polyphase FIR)
        T_plate float64 (2,)          two plate temperature sensors [C];  T_ctc, H_ctc: climate-chamber setpoint log values
        timestamp, channels (66,2) uint8, fs_raw, fs_out, t0 (start time of window, s), f_signal (Hz)
        check   dict: nan, inf, absmax_raw, md5_catch_raw, n_samples
        pitch0  float32 (13108,) excitation of path 0 (stored only for the first file of each frequency)
The raw file is deleted after its slice has been written and re-read successfully (--delete).
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np, h5py
from scipy.signal import resample_poly

I0, I1, DEC, PAD = 800, 10000, 4, 200          # 80 us .. 1.0 ms at 10 MHz; decimation 4 -> 2.5 MHz

def slice_file(path, out_root, keep_pitch):
    """Slice one raw OGW file: window, decimate, store float32 with checksums and temperatures."""
    with h5py.File(path, "r") as f:
        catch = f["pitchcatch/catch"][()]; fs = float(f["command/pitchcatch/sampling_frequency"][0])
        fsig = float(f["command/pitchcatch/signal_frequency"][0]); ch = f["command/pitchcatch/channels"][()]
        Tp = f["Temperature/values"][()].astype(float); Tc = float(f["CTC/Temperature"][0]); Hc = float(f["CTC/Humidity"][0])
        ts = float(f["timestamp"][0]); pitch0 = f["pitchcatch/pitch"][0].astype(np.float32) if keep_pitch else None
    chk = {"nan": int(np.isnan(catch).sum()), "inf": int(np.isinf(catch).sum()), "absmax_raw": float(np.abs(catch).max()),
           "md5_catch_raw": hashlib.md5(catch.tobytes()).hexdigest(), "n_samples": int(catch.shape[1]), "n_paths": int(catch.shape[0])}
    seg = catch[:, I0 - PAD: I1 + PAD]
    dec = resample_poly(seg, 1, DEC, axis=1)[:, PAD // DEC: (PAD + I1 - I0) // DEC].astype(np.float32)
    stamp = os.path.basename(os.path.dirname(path)); fk = os.path.basename(path).replace("pc_f", "").replace("kHz.h5", "")
    od = os.path.join(out_root, "d1", f"{fk}kHz"); os.makedirs(od, exist_ok=True); op = os.path.join(od, f"{stamp}.npz")
    np.savez_compressed(op, catch=dec, T_plate=Tp, T_ctc=Tc, H_ctc=Hc, timestamp=ts, channels=ch, fs_raw=fs, fs_out=fs / DEC,
                        t0=I0 / fs, f_signal=fsig, check=json.dumps(chk), **({"pitch0": pitch0} if keep_pitch else {}))
    z = np.load(op); assert z["catch"].shape == dec.shape and np.isfinite(z["catch"]).all()
    return op, chk

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--set", default="OGW_CFRP_Temperature_udam", help="data-set folder (udam, or dam_D16 for the clean test)")
    ap.add_argument("--batch", help="JSON list of device paths: process only these files (safe for parallel runs)")
    ap.add_argument("--pitch-only", nargs="*", help="raw files: store only their excitation signal as <freq>kHz/_pitch0_<stamp>.npz")
    a = ap.parse_args()
    if a.pitch_only:
        for p in a.pitch_only:
            with h5py.File(p, "r") as f:
                pitch0 = f["pitchcatch/pitch"][0].astype(np.float32); fs = float(f["command/pitchcatch/sampling_frequency"][0])
            stamp = os.path.basename(os.path.dirname(p)); fk = os.path.basename(p).replace("pc_f", "").replace("kHz.h5", "")
            np.savez_compressed(os.path.join(a.out, "d1", f"{fk}kHz", f"_pitch0_{stamp}.npz"), pitch0=pitch0, fs_raw=fs)
            if a.delete:
                os.remove(p)
            print("pitch saved", fk, stamp)
        sys.exit(0)
    files = sorted(glob.glob(os.path.join(a.root, a.set, "*", "pc_f*kHz.h5")))
    if a.batch:
        PFX = os.environ.get("RAW_PREFIX", "")
        want = [os.path.join(a.root, p[len(PFX):].replace("\\", "/")) for p in json.load(open(a.batch))]
        files = [p for p in want if os.path.exists(p)]
        missing = [p for p in want if not os.path.exists(p)]
        if missing: print("not staged:", len(missing), file=sys.stderr)
    logp = os.path.join(a.out, "d1_slice_log.jsonl"); n = 0
    for p in files:
        fk = os.path.basename(p).replace("pc_f", "").replace("kHz.h5", "")
        keep = not os.path.exists(os.path.join(a.out, "d1", f"{fk}kHz", "_pitch_saved"))
        try:
            op, chk = slice_file(p, a.out, keep)
            if keep: open(os.path.join(a.out, "d1", f"{fk}kHz", "_pitch_saved"), "w").write(os.path.basename(op))
            with open(logp, "a") as fh: fh.write(json.dumps({"raw": p.split("data/")[-1], "slice": op.split(a.out)[-1], **chk}, ensure_ascii=False) + "\n")
            if a.delete: os.remove(p)
            n += 1
        except Exception as e:
            with open(logp, "a") as fh: fh.write(json.dumps({"raw": p, "error": repr(e)}, ensure_ascii=False) + "\n")
            print("ERROR", p, e, file=sys.stderr)
    print(f"sliced {n} of {len(files)} files")
