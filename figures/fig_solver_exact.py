#!/usr/bin/env python3
"""Figure A1 (Appendix C): verification of the bicharacteristic-splitting solver.
(a) error against grid step on exact plane waves (A0, S0, SH0; Strang and Yoshida), (b) energy change with free and
periodic edges, (c) simulated (calibrated) and measured signals for actuator T4 at 21 C on a short and a long path."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL, panel
from bcs import ogw

R = os.path.join(os.path.dirname(__file__), "..", "results")
num = json.load(open(os.path.join(R, "e1_numerics.json")))
fig = plt.figure(figsize=(W_FULL, 2.55))
gs = fig.add_gridspec(1, 2, wspace=0.32)
ax = fig.add_subplot(gs[0, 0]); panel(ax, "(a)")
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
sty = {"A0": ("o", C["ours"]), "S0": ("s", C["base"]), "SH0": ("^", C["third"])}
for key, rows in num["order"].items():
    br, sch = key.split("/")
    x = [r["ppw"] for r in rows]; e = [r["err"] for r in rows]
    mk, col = sty[br]
    ax.loglog(x, e, marker=mk, color=col, ls="-" if sch == "yoshida" else "--", lw=1.2, ms=4.5,
              mfc=col if sch == "yoshida" else "white", mew=1.0)
ax.set_xlabel("Grid points per wavelength"); ax.set_ylabel("Relative error (energy norm)")
ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 40])); ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
hm = [Line2D([], [], color=sty[b][1], marker=sty[b][0], lw=1.2, ms=4.5, label=b) for b in sty]
ax.add_artist(ax.legend(handles=hm, fontsize=6, loc="upper right", ncol=3, handlelength=1.8, borderaxespad=0.2, columnspacing=0.8))
ax.set_ylim(1e-5, 40)
eo = lambda sch: [r["eoc"] for k, rr in num["order"].items() if k.endswith(sch) for r in rr if "eoc" in r]
hd = [Line2D([], [], color=C["ink2"], ls="-", lw=1.2, label=f"Yoshida, observed order {min(eo('yoshida')):.1f}–{max(eo('yoshida')):.1f}"),
       Line2D([], [], color=C["ink2"], ls="--", lw=1.2, label=f"Strang, observed order {min(eo('strang')):.1f}–{max(eo('strang')):.1f}")]
ax.legend(handles=hd, fontsize=6, loc="lower left", handlelength=2.2, borderaxespad=0.2)
ax = fig.add_subplot(gs[0, 1]); panel(ax, "(b)")
E = num["energy"]
for kind, col in (("membrane", C["base"]), ("bending", C["ours"])):
    for edges, ls in (("free", "-"), ("periodic", ":")):
        rr = sorted([r for r in E if r["kind"] == kind and r["edges"] == edges and r["scheme"] == "yoshida"], key=lambda r: r["h_mm"])
        ax.loglog([r["h_mm"] for r in rr], [-r["dE_rel"] for r in rr], color=col, ls=ls, marker="o", ms=4.5, lw=1.2)
ax.set_xlabel("Grid step $h$ (mm)"); ax.set_ylabel(r"Energy loss at 60 $\mu$s, $-\Delta E/E_0$")
ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1.0])); ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(2e-5, 3.0)
hd = [Line2D([], [], color=C["base"], lw=1.2, marker="o", ms=4.5, label="membrane"),
      Line2D([], [], color=C["ours"], lw=1.2, marker="o", ms=4.5, label="bending"),
      Line2D([], [], color=C["ink2"], lw=1.2, ls="-", label="free edges"),
      Line2D([], [], color=C["ink2"], lw=1.2, ls=":", label="periodic edges")]
ax.legend(handles=hd, fontsize=6, loc="upper left", ncol=2, handlelength=2.2, borderaxespad=0.2, columnspacing=1.0)
out = os.path.join(os.path.dirname(__file__), "figC1_exact.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print(out)
