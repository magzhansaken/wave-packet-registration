"""The synthetic two-mode example of examples/quickstart.py: one delay law per packet window beats a single stretch."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
import quickstart as q  # noqa: E402
from surrogates import core  # noqa: E402


def test_registration_beats_single_stretch_on_two_drift_rates():
    """Two modes drifting at different rates: the registration error at 45 C is below half the stretch error."""
    t = np.arange(0, 6e-4, 1 / q.FS); T = np.linspace(20.0, 35.0, 31)
    S = np.array([q.record(t, x) for x in T]); y = q.record(t, 45.0)[None]
    packets = [(m, k, 0.0, 0.0, tk) for m, k, tk, _ in q.PACKETS]
    p_bss, _ = core.fit_bss_tr(t, S, T, np.array([45.0]), "up")
    p_reg, _ = core.fit_br_gp(t, S, T, np.array([45.0]), "up", fk=q.FK, packets=packets, lam=0.0, resid="none", tc=0.0)
    assert core.rel_err(p_reg, y)[0] < 0.5 * core.rel_err(p_bss, y)[0]
