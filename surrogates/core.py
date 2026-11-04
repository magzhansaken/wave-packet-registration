"""Surrogates for guided-wave signals under a drifting operating condition (temperature, fatigue cycles).

Unit of prediction: one path (actuator-sensor pair) at one frequency; training signals S (n, m) at conditions T (n,);
predict the signal at query conditions Tq. All signals share the time axis t (s, absolute: burst starts at t = 0).

Methods
  copy      nearest training condition (the edge of the training block for extrapolation)
  knn       mean of the k nearest training signals
  gp        PCA + Gaussian-process regression over the condition (data only)
  bss       baseline signal stretch (Croxford et al. 2010): edge reference, stretch and gain laws fitted on training
  bss_gp    bss + PCA-GP of the residual in physical time (additive multi-fidelity, the MF-B-NO structure with a GP head)
  br        ablation of ours: registration only, data-fitted window laws, no residual
  br_gp     ours: bicharacteristic registration (packet windows from ray arrivals of the plate model, per-window linear
            delay laws shrunk to the pooled travel-time law of each mode) + PCA-GP of the residual in registered time
  dtw_or    dynamic time warping of the edge reference onto the MEASURED target (Douglass & Harley 2018) - oracle-type
  str_or    best global stretch and gain fitted on the MEASURED target - oracle-type
"""
import time
import numpy as np
from numba import njit
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

KS = np.linspace(0.97, 1.03, 241)


def rel_err(P, Y):
    return np.linalg.norm(P - Y, axis=-1) / np.linalg.norm(Y, axis=-1)


def warp(t, s, k):
    return np.interp(t * k, t, s, left=0.0, right=0.0)


def shift(t, s, dl):
    """s(t - delta(t)); delta scalar or array over t."""
    return np.interp(t - dl, t, s, left=0.0, right=0.0)


def edge_index(T, side):
    return int(np.argmax(T)) if side == "up" else int(np.argmin(T))


# ---------------------------------------------------------------- data-only
def fit_copy(t, S, T, Tq, side, **kw):
    return S[np.argmin(np.abs(Tq[:, None] - T[None, :]), axis=1)], None


def fit_knn(t, S, T, Tq, side, k=3, **kw):
    k = min(k, len(T))
    idx = np.argsort(np.abs(Tq[:, None] - T[None, :]), axis=1)[:, :k]
    return S[idx].mean(1), None


def pca_gp(X, Y, Xq, n_pca=10, kernel="rbf", return_std=False, center=True):
    """PCA + independent GPs over the condition for the component scores.
    center=False (residual models): uncentred basis and zero-mean GPs, so that far from the data the residual reverts
    to zero, i.e. the prediction reverts to the low-fidelity / registered signal instead of the training-mean residual."""
    n = max(1, min(n_pca, len(X) - 1, Y.shape[1]))
    if center:
        p = PCA(n_components=n).fit(Y); Z = p.transform(Y); comps = p.components_; mean = p.mean_
    else:
        U, sv, Vt = np.linalg.svd(Y, full_matrices=False); comps = Vt[:n]; Z = Y @ comps.T; mean = np.zeros(Y.shape[1])
    zs = Z.std(0) + 1e-12 if center else np.sqrt((Z ** 2).mean(0)) + 1e-12
    base = RBF(5.0, (0.3, 300.0)) if kernel == "rbf" else Matern(5.0, (0.3, 300.0), nu=2.5)
    k = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-3, (1e-8, 1e0))
    g = GaussianProcessRegressor(kernel=k, normalize_y=center, n_restarts_optimizer=0, random_state=0).fit(X[:, None], Z / zs)
    if not return_std:
        return (g.predict(Xq[:, None]).reshape(len(Xq), -1) * zs) @ comps + mean, None
    mu, sd = g.predict(Xq[:, None], return_std=True)
    mu = mu.reshape(len(Xq), -1) * zs
    sd = np.atleast_2d(sd)
    if sd.shape[0] != len(Xq):
        sd = sd.T
    if sd.shape[1] == 1 and Z.shape[1] > 1:
        sd = np.repeat(sd, Z.shape[1], 1)
    sd = sd * zs
    trunc = np.sqrt(np.maximum(((Y - (Z @ comps + mean)) ** 2).mean(0), 0))
    std_t = np.sqrt((sd ** 2) @ (comps ** 2) + trunc[None, :] ** 2)
    return mu @ comps + mean, std_t


def fit_gp(t, S, T, Tq, side, n_pca=10, kernel="rbf", want_std=False, **kw):
    return pca_gp(T, S, Tq, n_pca, kernel, want_std)


# ---------------------------------------------------------------- stretch family
def stretch_fit(t, sref, s, w):
    best = (1e300, 1.0, 1.0); y = s[w]
    for k in KS:
        p = warp(t, sref, k)[w]; pp = p @ p
        g = (p @ y) / (pp + 1e-300); e = np.sum((g * p - y) ** 2)
        if e < best[0]:
            best = (e, k, g)
    return best[1], best[2]


def bss_laws(t, S, T, side, w=None):
    w = slice(None) if w is None else w
    ie = edge_index(T, side); sref = S[ie]; Tref = T[ie]
    sub = np.arange(len(T)) if len(T) <= 40 else np.linspace(0, len(T) - 1, 40).astype(int)
    kg = np.array([stretch_fit(t, sref, S[i], w) for i in sub]); dT = T[sub] - Tref
    c = np.polyfit(dT, kg[:, 0] - 1, 1) if len(sub) > 1 else np.array([0.0, 0.0])
    gl = np.polyfit(dT, kg[:, 1], 1) if len(sub) > 1 else np.array([0.0, 1.0])
    return sref, Tref, (lambda x: 1 + np.polyval(c, x - Tref)), (lambda x: np.polyval(gl, x - Tref))


def bss_laws_tracked(t, S, T, side, half=0.002, n=81):
    """BSS law with stretch factors tracked along the condition (from the edge inwards): each factor is searched in a
    narrow band around the previous one, so the estimate does not jump between correlation peaks one period apart."""
    ie = edge_index(T, side); sref = S[ie]; Tref = T[ie]
    order = np.argsort(-T) if side == "up" else np.argsort(T)
    k_prev = 1.0; ks = np.ones(len(T)); gs = np.ones(len(T))
    for i in order:
        if i == ie:
            continue
        cand = k_prev + np.linspace(-half, half, n); best = (-np.inf, 1.0, 1.0); y = S[i]
        for k in cand:
            p = warp(t, sref, k); c = (p @ y) / (np.linalg.norm(p) * np.linalg.norm(y) + 1e-300)
            if c > best[0]:
                best = (c, k, (p @ y) / (p @ p + 1e-300))
        ks[i], gs[i] = best[1], best[2]; k_prev = best[1]
    dT = T - Tref
    c = np.polyfit(dT, ks - 1, 1); gl = np.polyfit(dT, gs, 1)
    return sref, Tref, (lambda x: 1 + np.polyval(c, x - Tref)), (lambda x: np.polyval(gl, x - Tref))


def fit_bss_tr(t, S, T, Tq, side, **kw):
    """Stronger BSS baseline: same law, tracked stretch estimation."""
    sref, Tref, kf, gf = bss_laws_tracked(t, S, T, side)
    return np.array([gf(x) * warp(t, sref, kf(x)) for x in Tq]), None


def fit_bss_tr_gp(t, S, T, Tq, side, n_pca=10, want_std=False, **kw):
    sref, Tref, kf, gf = bss_laws_tracked(t, S, T, side)
    LFq = np.array([gf(x) * warp(t, sref, kf(x)) for x in Tq]); LFtr = np.array([gf(x) * warp(t, sref, kf(x)) for x in T])
    R, sd = pca_gp(T, S - LFtr, Tq, n_pca, "rbf", want_std, center=False)
    return LFq + R, sd


def fit_bss(t, S, T, Tq, side, **kw):
    sref, Tref, kf, gf = bss_laws(t, S, T, side)
    return np.array([gf(x) * warp(t, sref, kf(x)) for x in Tq]), None


def fit_bss_gp(t, S, T, Tq, side, n_pca=10, want_std=False, **kw):
    sref, Tref, kf, gf = bss_laws(t, S, T, side)
    LFq = np.array([gf(x) * warp(t, sref, kf(x)) for x in Tq]); LFtr = np.array([gf(x) * warp(t, sref, kf(x)) for x in T])
    R, sd = pca_gp(T, S - LFtr, Tq, n_pca, "rbf", want_std, center=False)
    return LFq + R, sd


# ---------------------------------------------------------------- bicharacteristic registration (ours)
def windows_from_packets(t, packets, fk, wf=1.0, t_lo=None, t_hi=None):
    """Window centres along the time axis and the mode that dominates each of them.
    Centres: ray arrivals of the packets (merged when closer than half a burst); gaps longer than 1.5 bursts are filled
    uniformly. Label of a window = the mode of the last DIRECT arrival before it: 'pre' before the S0 direct packet,
    'S0' between the S0 and the A0 direct arrivals (S0 and its reflections), 'A0' after the A0 direct arrival
    (A0 packets with the S0 reverberation). Returns centres (K,), half-width hw, labels."""
    burst = 5.0 / (fk * 1e3); hw = wf * max(burst, 20e-6) * 0.75
    lo = (t[0] if t_lo is None else t_lo) + hw; hi = (t[-1] if t_hi is None else t_hi) - hw
    t_s0 = min([tk for (m, name, L, psi, tk) in packets if m == "S0" and name == "direct"], default=np.inf)
    t_a0 = min([tk for (m, name, L, psi, tk) in packets if m == "A0" and name == "direct"], default=np.inf)
    pk = sorted([tk for (m, name, L, psi, tk) in packets if lo <= tk <= hi])
    cen = []
    for tk in pk:
        if cen and tk - cen[-1] < 0.5 * burst:
            continue
        cen.append(tk)
    allc = [lo] + cen + [hi]
    out = []
    for a, b in zip(allc[:-1], allc[1:]):
        out.append(a)
        gap = b - a
        if gap > 1.5 * burst:
            nfill = int(gap // burst)
            out += [a + j * gap / nfill for j in range(1, nfill)]
    out.append(hi)
    c = np.unique(np.array(out))
    lab = ["A0" if x >= t_a0 - 0.5 * burst else ("S0" if x >= t_s0 - 0.5 * burst else "pre") for x in c]
    return c, hw, lab


def track_lags(t, S, order, ref, lo, hi, f, rng_periods=0.4, n_c=41):
    """Delay of each signal against S[ref] in [lo, hi], tracked along `order` (from the reference outwards)."""
    w = np.where((t >= lo) & (t <= hi))[0]
    lags = np.zeros(len(S))
    if len(w) < 8:
        return lags
    tap = np.hanning(len(w)); x = S[ref, w] * tap; tw = t[w]
    prev = 0.0
    for n in order:
        if n == ref:
            prev = 0.0; continue
        cands = prev + np.linspace(-rng_periods / f, rng_periods / f, n_c)
        Y = np.stack([np.interp(tw + c, t, S[n]) for c in cands]) * tap
        v = Y @ x; j = int(np.argmax(v)); frac = 0.0
        if 0 < j < n_c - 1:
            den = v[j - 1] - 2 * v[j] + v[j + 1]
            frac = 0.5 * (v[j - 1] - v[j + 1]) / den if den != 0 else 0.0
        lags[n] = cands[j] + frac * (cands[1] - cands[0]); prev = lags[n]
    return lags


def br_laws(t, S, T, side, fk, packets, kappa, lam=1.0, wf=1.0, tc=None):
    """Delay field delta(t; T) from per-window laws. kappa: dict mode -> pooled relative travel-time slope (1/C) for the
    prior b0 = kappa * (t_c_window - t_c). lam: prior precision tau in C^2 (0 = data only, inf = prior only):
    slope = (sum x*lag + tau*b0) / (sum x^2 + tau), so the prior matters when the training block is narrow and fades as
    the block widens (sum x^2 grows)."""
    f = fk * 1e3; tc = 2.5 / f if tc is None else tc
    ie = edge_index(T, side); Tref = T[ie]
    cen, hw, lab = windows_from_packets(t, packets, fk, wf)
    order = np.argsort(-T) if side == "up" else np.argsort(T)          # from the edge inwards
    x = T - Tref; Sxx = float(np.sum(x * x)) + 1e-12
    slopes = np.zeros(len(cen))
    for j, (c, l) in enumerate(zip(cen, lab)):
        kap = kappa.get(l, kappa.get("A0", 0.0))
        b0 = kap * (c - tc)
        if np.isinf(lam):
            slopes[j] = b0; continue
        lg = track_lags(t, S, order, ie, c - hw, c + hw, f)
        Sxy = float(np.sum(x * lg))
        slopes[j] = (Sxy + lam * b0) / (Sxx + lam)
    def delta(Tv):
        return np.interp(t, cen, slopes * (Tv - Tref))
    return ie, Tref, delta, cen, slopes


def fit_br_gp(t, S, T, Tq, side, fk=40, packets=(), kappa=None, lam=1.0, wf=1.0, n_pca=10, resid="gp", want_std=False, tc=None, **kw):
    """Ours. resid='none': registration only (bicharacteristic low-fidelity law); resid='gp': plus a zero-mean PCA-GP of
    the residual in registered time."""
    kappa = kappa or {}
    ie, Tref, delta, cen, slopes = br_laws(t, S, T, side, fk, packets, kappa, lam, wf, tc=tc)
    sref = S[ie]
    Reg = np.array([shift(t, s, -delta(x)) for s, x in zip(S, T)])
    g = (Reg @ sref) / (sref @ sref + 1e-300)
    gl = np.polyfit(T - Tref, g, 1) if len(T) > 1 else np.array([0.0, 1.0])
    gf = lambda xv: np.polyval(gl, xv - Tref)
    Reg = Reg / gf(T)[:, None]
    if resid == "none":
        P = np.array([gf(x) * shift(t, sref, delta(x)) for x in Tq])
        return P, None
    R, sd = pca_gp(T, Reg - sref, Tq, n_pca, "rbf", want_std, center=False)
    P = np.array([gf(x) * shift(t, sref + r, delta(x)) for r, x in zip(R, Tq)])
    if sd is not None:
        sd = np.array([abs(gf(x)) * shift(t, s_, delta(x)) for s_, x in zip(sd, Tq)])
    return P, sd


def fit_br(t, S, T, Tq, side, **kw):
    """Ablation: registration only, data-fitted per-window laws (lam = 0), no residual model, no tuning."""
    kw = dict(kw); kw.update(lam=0.0, wf=1.0, resid="none")
    return fit_br_gp(t, S, T, Tq, side, **kw)


# ---------------------------------------------------------------- oracle-type references
@njit(cache=True)
def _dtw_path_map(a, b, band):
    """DTW of reference a onto target b (squared differences, Sakoe-Chiba band); returns for each target index the mean
    of the reference samples matched to it."""
    n = len(a); m = len(b); INF = 1e300
    D = np.full((n + 1, m + 1), INF); D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band); j1 = min(m, i + band)
        for j in range(j0, j1 + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            best = D[i - 1, j - 1]
            if D[i - 1, j] < best:
                best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            D[i, j] = c + best
    out = np.zeros(m); cnt = np.zeros(m)
    i = n; j = m
    while i > 0 and j > 0:
        out[j - 1] += a[i - 1]; cnt[j - 1] += 1
        d0 = D[i - 1, j - 1]; d1 = D[i - 1, j]; d2 = D[i, j - 1]
        if d0 <= d1 and d0 <= d2:
            i -= 1; j -= 1
        elif d1 <= d2:
            i -= 1
        else:
            j -= 1
    for k in range(m):
        if cnt[k] > 0:
            out[k] /= cnt[k]
    return out


def fit_dtw_or(t, S, T, Tq, side, Sq=None, band_frac=0.02, **kw):
    ie = edge_index(T, side); sref = S[ie]; band = max(2, int(band_frac * len(t)))
    P = []
    for y in Sq:
        p = _dtw_path_map(sref, y, band); g = (p @ y) / (p @ p + 1e-300); P.append(g * p)
    return np.array(P), None


def fit_str_or(t, S, T, Tq, side, Sq=None, **kw):
    ie = edge_index(T, side); sref = S[ie]
    P = []
    for y in Sq:
        k, g = stretch_fit(t, sref, y, slice(None)); P.append(g * warp(t, sref, k))
    return np.array(P), None


METHODS = {"copy": fit_copy, "knn": fit_knn, "gp": fit_gp, "bss": fit_bss, "bss_gp": fit_bss_gp, "br": fit_br, "br_gp": fit_br_gp,
           "dtw_or": fit_dtw_or, "str_or": fit_str_or, "bss_tr": fit_bss_tr, "bss_tr_gp": fit_bss_tr_gp}
CANDIDATES = ["copy", "knn", "gp", "bss", "bss_gp", "br", "br_gp"]       # surrogates eligible for selection (no oracles)


def timed(fn, *a, **kw):
    t0 = time.time(); out = fn(*a, **kw); return out, time.time() - t0


# ---------------------------------------------------------------- intervals
def conformal_calibrate(R_cal, d_cal, Y_cal, q=0.95):
    """Split conformal on a calibration block. R_cal: residuals (n, m); d_cal: distance of each calibration signal from
    the edge of the data the model was fitted on. Returns time profile s(t), quantile, growth slope kappa (1/C)."""
    s_t = np.maximum(R_cal.std(0), 1e-3 * np.abs(Y_cal).max())
    z = np.abs(R_cal) / s_t; qz = float(np.quantile(z.ravel(), q))
    rms = np.sqrt((R_cal ** 2).mean(1))
    if len(d_cal) > 2 and np.ptp(d_cal) > 0:
        b, a = np.polyfit(d_cal, rms, 1); kap = max(b / max(a + b * d_cal.mean(), 1e-12), 0.0)
    else:
        kap = 0.0
    return s_t, qz, float(kap), float(d_cal.mean())


def conformal_halfwidth(s_t, qz, kap, dmean, d_test, law=True):
    scale = (1 + kap * np.maximum(d_test - dmean, 0.0)) if law else np.ones_like(d_test)
    return qz * s_t[None, :] * scale[:, None]
