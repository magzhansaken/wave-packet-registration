"""Open Guided Waves #1 plate: geometry, slice loading, packet picking (shared by E1 and the surrogate experiments)."""
import glob, os
import numpy as np
from scipy.signal import hilbert

# transducer coordinates, m (Moll et al., Sci. Data 6, 191, 2019, Table 1); plate 500 x 500 x 2 mm; disks 10 mm
XY = np.array([[450, 470], [370, 470], [290, 470], [210, 470], [130, 470], [50, 470],
               [450, 30], [370, 30], [290, 30], [210, 30], [130, 30], [50, 30]], float) / 1e3
PLATE = (0.5, 0.5)
R_PZT = 5e-3
FREQS = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
MAIN = [40, 60, 100, 140, 180, 220]
CONFIRM = [80, 120, 160, 200, 240, 260]
SLICES = os.environ.get("D1_SLICES", "data/d1")


def load_freq(fk, root=SLICES, max_files=None):
    """All slices of one frequency, sorted by time stamp. Returns dict(S (n, 66, m) float32, T (n,), Ts (n, 2), ts (n,),
    t (m,), channels (66, 2), files)."""
    fl = sorted(p for p in glob.glob(os.path.join(root, f"{fk}kHz", "*.npz")) if not os.path.basename(p).startswith("_"))
    if max_files:
        fl = fl[:max_files]
    S, T, Ts, ts = [], [], [], []
    for f in fl:
        d = np.load(f)
        S.append(d["catch"]); Ts.append(d["T_plate"]); T.append(float(np.mean(d["T_plate"]))); ts.append(float(d["timestamp"]))
        ch = d["channels"]; t = float(d["t0"]) + np.arange(d["catch"].shape[1]) / float(d["fs_out"])
    return dict(S=np.stack(S), T=np.array(T), Ts=np.array(Ts), ts=np.array(ts), t=t, channels=ch, files=fl)


def path_geometry(channels):
    """Length (m) and direction angle (rad) of every actuator-sensor path."""
    a, b = channels[:, 0], channels[:, 1]
    d = XY[b] - XY[a]
    return np.linalg.norm(d, axis=1), np.arctan2(d[:, 1], d[:, 0])


def envelope(s):
    """Hilbert envelope of a signal."""
    return np.abs(hilbert(s))


def pick_peak(t, s, lo, hi):
    """Time of the envelope maximum in [lo, hi] with parabolic refinement (NaN if the window is empty)."""
    e = envelope(s); w = np.where((t >= lo) & (t <= hi))[0]
    if len(w) < 3:
        return np.nan, np.nan
    j = w[np.argmax(e[w])]
    if 0 < j < len(t) - 1:
        y0, y1, y2 = e[j - 1], e[j], e[j + 1]; den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        frac = 0.0
    return t[j] + frac * (t[1] - t[0]), e[j]


def xcorr_lag(t, ref, s, lo, hi, maxlag):
    """Delay of s relative to ref in the window [lo, hi] (Hann-tapered), sub-sample by parabola, seconds."""
    dt = t[1] - t[0]; w = (t >= lo) & (t <= hi); idx = np.where(w)[0]
    if len(idx) < 8:
        return np.nan
    tap = np.hanning(len(idx)); x = ref[idx] * tap; y = s[idx] * tap
    L = int(round(maxlag / dt)); cc = np.correlate(y, x, mode="full"); mid = len(x) - 1
    seg = cc[mid - L: mid + L + 1]; j = int(np.argmax(seg))
    frac = 0.0
    if 0 < j < len(seg) - 1:
        y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]; den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return (j - L + frac) * dt
