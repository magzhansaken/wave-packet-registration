"""Tests of the baselines of amendment 13: windowed shift-and-stretch and the drift-aware conformal
intervals, plus the per-unit AUC."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import e13_run as E
from experiments.e13_auc import auc


def _burst(t, tc, f=60e3, n=5):
    """Cosine-squared windowed tone burst centred on tc."""
    w = np.where(np.abs(t - tc) < n / (2 * f), np.cos(np.pi * (t - tc) * f / n) ** 2, 0.0)
    return w * np.sin(2 * np.pi * f * (t - tc))


def test_sdws_recovers_linear_stretch():
    """Windowed shift-and-stretch recovers a stretch that is linear in temperature."""
    t = np.arange(0, 6e-4, 4e-7); T = np.linspace(20, 40, 21)
    S = np.array([_burst(t * (1 + 2e-4 * (x - 40)), 2e-4) for x in T])        # stretch linear in T, edge at 40
    P, _ = E.fit_sdws(t, S, T, np.array([45.0]), "up", fk=60, win="all", amp=True)
    y = _burst(t * (1 + 2e-4 * 5), 2e-4)
    assert np.linalg.norm(P[0] - y) / np.linalg.norm(y) < 0.05


def test_window_mask_starts_at_first_arrival():
    """The analysis window of sdws contains the first arrival."""
    t = np.arange(0, 6e-4, 4e-7); s = _burst(t, 2e-4)
    w = E.window_mask(t, s, 60, 3)
    assert w.any() and t[w][0] < 2e-4 < t[w][-1]


def test_weighted_quantile_matches_unweighted():
    """With equal weights the weighted quantile is the ordinary quantile."""
    rng = np.random.default_rng(0); z = rng.random(1000)
    assert abs(E.wquantile(z, np.ones_like(z), 0.9) - np.quantile(z, 0.9)) < 0.01


def test_intervals_cover_exchangeable_residuals():
    """Split conformal, nexCP and ACI cover exchangeable residuals at about 95 %."""
    rng = np.random.default_rng(1); m = 200
    Rcal = rng.normal(size=(20, m)); Ycal = rng.normal(size=(20, m)) * 5
    Ste = rng.normal(size=(30, m)) * 5; P = Ste - rng.normal(size=(30, m))
    out = E.intervals(Ste, P, Rcal, Ycal, np.linspace(0, 5, 20), np.linspace(6, 20, 30))
    for k in ("conf", "nexcp", "aci"):
        assert 0.9 < np.mean(out[k]["cov_sig"]) < 0.99


def test_auc():
    """AUC is 1 for separated scores and 0 for reversed ones."""
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([0.0]), np.array([1.0])) == 0.0
