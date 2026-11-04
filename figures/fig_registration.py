#!/usr/bin/env python3
"""Figure 2: wave-packet registration on one measured path (D1 main half, 60 kHz, split S2: training <= 35 C, test >= 40 C).
(a) edge record with its packet windows, coloured by mode label; (b) tracked delays of the training records in four windows
(filled) and their linear delay laws (Equation 3, tau = 0) extended into the test range, with the delays of the test records
tracked by the same procedure (open markers; not used in the fit). Prints the numbers shown in the figure."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL, panel
from bcs import ogw, rays
from surrogates import core
from experiments.e2_run import masks, calib_s

FK, SPLIT = 60, "S2_le35_ge40"
D = ogw.load_freq(FK); t = D["t"]; S = D["S"].astype(np.float64); T = D["T"]; S -= S.mean(-1, keepdims=True)
side, tr, te, val, inner = masks(T, SPLIT); ch = D["channels"]; L, _ = ogw.path_geometry(ch)
i = int(np.argsort(np.abs(L - np.median(L)))[0])                       # path of median length
pk = rays.packet_table(ogw.XY[ch[i][0]], ogw.XY[ch[i][1]], FK, modes=("S0", "A0"), order=1, tmax=t[-1], s=calib_s(SPLIT))
cen, hw, lab = core.windows_from_packets(t, pk, FK)
Str, Ttr = S[tr, i], T[tr]; ie = core.edge_index(Ttr, side); Te = Ttr[ie]
f = FK * 1e3; tc = 2.5 / f
both = np.concatenate([Str, S[te, i]]); Tb = np.concatenate([Ttr, T[te]]); ref = ie
order = np.argsort(np.abs(Tb - Te))                                     # from the edge outwards, both sides
LAB = {"pre": ("#c9c8c3", "before S0"), "S0": (C["muted"], "S0 label"), "A0": (C["ink2"], "A0 label")}
pick = [j for j in np.linspace(1, len(cen) - 2, 4).astype(int)]
cols = [C["ours"], C["base"], C["third"], C["fourth"]]

fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.4), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axs[0]; panel(ax, "(a)")
y = Str[ie] / np.abs(Str[ie]).max()
ax.plot(t * 1e6, y, color=C["ink"], lw=0.6)
for k, (c, l) in enumerate(zip(cen, lab)):                              # windows as staggered bars, centre ticks
    yb = 1.18 + 0.14 * (k % 2)
    ax.plot([(c - hw) * 1e6, (c + hw) * 1e6], [yb, yb], color=LAB[l][0], lw=2.2, solid_capstyle="butt")
    ax.plot([c * 1e6, c * 1e6], [yb - 0.06, yb + 0.06], color=C["ink2"], lw=0.6)
    if k in pick:
        ax.plot(c * 1e6, yb, "o", color=cols[pick.index(k)], ms=3.2, zorder=5)
for l, (col, name) in LAB.items():
    if l in lab:
        ax.plot([], [], color=col, lw=2.2, label=name)
ax.set_xlabel(r"Time ($\mu$s)"); ax.set_ylabel("Edge record (normalised)"); ax.set_xlim(t[0] * 1e6, t[-1] * 1e6)
ax.legend(loc="lower right", ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
ax.set_ylim(-1.3, 1.45)

ax = axs[1]; panel(ax, "(b)"); SL = []
x_tr = Ttr - Te; xx = np.linspace(min(x_tr.min(), 0), (T[te] - Te).max(), 50)
for j, col in zip(pick, cols):
    ord_tr = np.argsort(-Ttr) if side == "up" else np.argsort(Ttr)       # as in core.br_laws: from the edge inwards
    ltr = core.track_lags(t, Str, ord_tr, ie, cen[j] - hw, cen[j] + hw, f)
    ord_te = np.concatenate([[ref], len(Ttr) + np.argsort(np.abs(T[te] - Te))])   # display only: from the edge outwards
    lte = core.track_lags(t, both, ord_te, ref, cen[j] - hw, cen[j] + hw, f)[len(Ttr):]
    b = float(np.sum(x_tr * ltr) / np.sum(x_tr * x_tr))
    ax.plot(x_tr, ltr * 1e6, "o", color=col, ms=1.8, mew=0)
    ax.plot(T[te] - Te, lte * 1e6, "o", mfc="none", mec=col, ms=2.2, mew=0.5)
    ax.plot(xx, b * xx * 1e6, "-", color=col, lw=1.0, label=f"{cen[j] * 1e6:.0f} $\\mu$s ({lab[j]})")
    print(f"window {cen[j]*1e6:.0f} us ({lab[j]}): slope {b*1e9:.1f} ns/C"); SL.append({"c_us": float(cen[j] * 1e6), "label": lab[j], "slope_ns_per_C": b * 1e9})
ax.axvline(0, color=C["ink2"], lw=0.6); ax.axvspan(T[te].min() - Te, (T[te] - Te).max(), color=C["grid"], alpha=0.5, lw=0)
ax.set_xlabel(r"$\theta-\theta_e$ ($^\circ$C)"); ax.set_ylabel(r"Delay ($\mu$s)")
ax.legend(loc="upper left", fontsize=6, handlelength=1.2)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fig1_registration.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print(out, "path", i, "L", round(float(L[i]), 3), "edge T", round(float(Te), 2), "n_train", int(tr.sum()), "n_test", int(te.sum()), "windows", len(cen))
json.dump({"f_kHz": FK, "split": SPLIT, "path": i, "L_m": float(L[i]), "edge_T": float(Te), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "n_windows": len(cen), "windows": SL}, open(os.path.join(os.path.dirname(__file__), "..", "results", "fig1_registration.json"), "w"), indent=1)
