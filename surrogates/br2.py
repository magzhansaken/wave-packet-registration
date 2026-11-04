"""Variant A2: wave-packet registration with an intercept in every delay law.

Post-hoc diagnosis on the two D1 halves (`experiments/a2_diag.py`; `docs/PROTOCOL.md`, post-hoc analyses): the per-window delay law of br/br_gp
is fitted through the origin at the edge signal, slope = sum(x*lag)/sum(x^2), so the edge signal is treated as exact.
The training block holds measurements from the heating and the cooling legs of the cycle, and at the same plate
temperature the two legs differ by a small delay offset. When the edge signal comes from the other leg than most of the
block, the law through the origin is biased (D1 reserve half, 40 kHz, split S1: A0 slopes 3.0-6.1e-4 instead of 3.9-10e-4
fitted with an intercept). Here the law is lag = a_w + b_w x with a centred least-squares slope and the same prior:
b_w = (sum (x-xm)(lag-lm) + tau b0) / (sum (x-xm)^2 + tau), a_w = lm - b_w xm. Everything else is unchanged."""
import numpy as np
from surrogates import core


def br_laws2(t, S, T, side, fk, packets, kappa, lam=1.0, wf=1.0, tc=None):
    """Per-window delay laws with an intercept (centred least squares), shrunk to the pooled prior slope."""
    f = fk * 1e3; tc = 2.5 / f if tc is None else tc
    ie = core.edge_index(T, side); Tref = T[ie]
    cen, hw, lab = core.windows_from_packets(t, packets, fk, wf)
    order = np.argsort(-T) if side == "up" else np.argsort(T)
    x = T - Tref; xm = float(np.mean(x)); xc = x - xm; Sxx = float(np.sum(xc * xc)) + 1e-12
    slopes = np.zeros(len(cen)); icpt = np.zeros(len(cen))
    for j, (c, l) in enumerate(zip(cen, lab)):
        kap = kappa.get(l, kappa.get("A0", 0.0)); b0 = kap * (c - tc)
        if np.isinf(lam):
            slopes[j] = b0; continue
        lg = core.track_lags(t, S, order, ie, c - hw, c + hw, f)
        lm = float(np.mean(lg))
        slopes[j] = (float(np.sum(xc * (lg - lm))) + lam * b0) / (Sxx + lam)
        icpt[j] = lm - slopes[j] * xm
    def delta(Tv):
        """Delay field over the time axis at condition Tv."""
        return np.interp(t, cen, icpt + slopes * (Tv - Tref))
    return ie, Tref, delta, cen, slopes


def fit_br_gp2(t, S, T, Tq, side, fk=40, packets=(), kappa=None, lam=1.0, wf=1.0, n_pca=10, resid="gp", want_std=False, tc=None, **kw):
    """Registration with intercepts, plus the zero-mean PCA-GP residual in registered time (resid='gp')."""
    kappa = kappa or {}
    ie, Tref, delta, cen, slopes = br_laws2(t, S, T, side, fk, packets, kappa, lam, wf, tc=tc)
    sref = S[ie]
    Reg = np.array([core.shift(t, s, -delta(x)) for s, x in zip(S, T)])
    g = (Reg @ sref) / (sref @ sref + 1e-300)
    gl = np.polyfit(T - Tref, g, 1) if len(T) > 1 else np.array([0.0, 1.0])
    gf = lambda xv: np.polyval(gl, xv - Tref)
    Reg = Reg / gf(T)[:, None]
    if resid == "none":
        return np.array([gf(x) * core.shift(t, sref, delta(x)) for x in Tq]), None
    R, sd = core.pca_gp(T, Reg - sref, Tq, n_pca, "rbf", want_std, center=False)
    P = np.array([gf(x) * core.shift(t, sref + r, delta(x)) for r, x in zip(R, Tq)])
    if sd is not None:
        sd = np.array([abs(gf(x)) * core.shift(t, s_, delta(x)) for s_, x in zip(sd, Tq)])
    return P, sd


def fit_br2(t, S, T, Tq, side, **kw):
    """Registration with intercepts only: data-fitted laws, no residual model."""
    kw = dict(kw); kw.update(lam=0.0, wf=1.0, resid="none")
    return fit_br_gp2(t, S, T, Tq, side, **kw)
