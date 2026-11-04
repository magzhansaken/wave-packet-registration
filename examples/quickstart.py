#!/usr/bin/env python3
"""Quick start without downloading data: two wave packets that drift at different rates with temperature.

A synthetic pitch-catch record holds fast S0-like packets and slow A0-like packets (direct arrivals and edge
reflections). Their travel times grow with temperature at different relative rates, which is the situation in which a single stretch of the baseline fails.
Both predictors are fitted on records at 20-35 C and predict the record at 45 C:

    tracked baseline signal stretch (one stretch factor for the whole record)   -> surrogates.core.fit_bss_tr
    wave-packet registration (one delay law per packet window)                  -> surrogates.core.fit_br_gp

Run from the repository root:  python3 examples/quickstart.py   (a few seconds on one CPU core)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogates import core  # noqa: E402

FK = 60                                   # centre frequency, kHz
F = FK * 1e3
FS = 2.5e6                                # sampling rate, Hz
SLOPE = {"S0": 1.0e-4, "A0": 6.0e-4}     # relative travel-time slopes d ln t / dT, 1/C
# (mode, kind, arrival time at 20 C in s, amplitude): direct packets and edge reflections of both modes
PACKETS = [("S0", "direct", 120e-6, 0.5), ("S0", "edge", 200e-6, 0.3), ("S0", "edge", 270e-6, 0.25),
           ("A0", "direct", 330e-6, 1.0), ("A0", "edge", 450e-6, 0.6), ("A0", "edge", 520e-6, 0.4)]


def burst(t, tc, ncyc=5):
    """Hann-windowed tone burst of ncyc cycles at F centred on tc."""
    w = np.where(np.abs(t - tc) < ncyc / (2 * F), np.cos(np.pi * (t - tc) * F / ncyc) ** 2, 0.0)
    return w * np.sin(2 * np.pi * F * (t - tc))


def record(t, temp):
    """Sum of the packets; the arrival time of each grows with temperature at the slope of its mode."""
    d = temp - 20.0
    return sum(a * burst(t, tk * (1 + SLOPE[m] * d)) for m, _, tk, a in PACKETS)


def main():
    """Fit both predictors on 20-35 C, predict 45 C, print the relative L2 error of each."""
    t = np.arange(0, 6e-4, 1 / FS)
    T_train = np.linspace(20.0, 35.0, 31)
    S = np.array([record(t, x) for x in T_train])
    T_query = np.array([45.0])
    y = record(t, 45.0)

    # packet table in the format of bcs.rays.packet_table: (mode, kind, path length, ray angle, arrival time)
    packets = [(m, k, 0.0, 0.0, tk) for m, k, tk, _ in PACKETS]
    p_bss, _ = core.fit_bss_tr(t, S, T_train, T_query, "up")
    p_reg, _ = core.fit_br_gp(t, S, T_train, T_query, "up", fk=FK, packets=packets, lam=0.0, resid="none", tc=0.0)

    for name, p in (("single stretch (tracked BSS)", p_bss), ("wave-packet registration", p_reg)):
        print(f"{name:30s} relative L2 error at 45 C: {core.rel_err(p, y[None])[0]:.3f}")


if __name__ == "__main__":
    main()
