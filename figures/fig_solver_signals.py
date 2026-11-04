#!/usr/bin/env python3
"""Figure 4: simulated and measured signals; panels (c),(d) the difference measured - calibrated simulation.
(a) error against grid step on exact plane waves (A0, S0, SH0; Strang and Yoshida), (b) energy change with free and
periodic edges, (c) simulated (calibrated) and measured signals for actuator T4 at 21 C on a short and a long path."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL, panel
from bcs import ogw

R = os.path.join(os.path.dirname(__file__), "..", "results")
num = json.load(open(os.path.join(R, "e1_numerics.json")))
fig = plt.figure(figsize=(W_FULL, 3.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.6], hspace=0.18, wspace=0.32)
z = np.load(os.path.join(R, "e1_solver_T4_40k.npz")); ts = z["t"]
D = ogw.load_freq(40); ch = D["channels"]; cold = int(np.argmin(D["T"]))
for j, (rcv, title) in enumerate(((2, "T4 to T3, 80 mm"), (9, "T4 to T10, 440 mm"))):
    ax = fig.add_subplot(gs[0, j]); panel(ax, "(a)" if j == 0 else "(b)")
    axr = fig.add_subplot(gs[1, j], sharex=ax); panel(axr, "(c)" if j == 0 else "(d)")
    i = [k for k in range(len(ch)) if set(ch[k].tolist()) == {3, rcv}][0]
    m = D["S"][cold, i].astype(float); m = m / np.abs(m).max()
    for name, col, lab in (("nominal", C["base"], "solver, nominal"), ("calibrated_cold", C["ours"], "solver, calibrated")):
        x = np.interp(D["t"], ts, z[name][rcv]); x = x / np.abs(x).max()
        ax.plot(D["t"] * 1e6, x, color=col, lw=0.9, label=lab)
        if name == "calibrated_cold":
            axr.plot(D["t"] * 1e6, m - x, color=C["ours"], lw=0.8)
    ax.plot(D["t"] * 1e6, m, color=C["ink"], lw=0.7, label="measured, 21 °C")
    ax.set_xlim(80, 300 if rcv == 2 else 600); ax.tick_params(labelbottom=False)
    if j == 0:
        ax.set_ylabel("Normalised signal"); axr.set_ylabel("Difference")
    ax.set_title(title, fontsize=7.5, color=C["ink"], loc="left")
    axr.set_ylim(-1.6, 1.6); axr.set_xlabel(r"Time ($\mu$s)")
    if j == 0:
        ax.legend(fontsize=6, loc="lower right", borderaxespad=0.2)
out = os.path.join(os.path.dirname(__file__), "fig3_signals.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print(out)
