#!/usr/bin/env python3
"""Figure 6: coverage of the nominal 95 % bands of br_gp against the distance from the training edge on the three sets
of D1: drift-law conformal (solid) and split conformal (dashed). Data: results/coverage_eval.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL

R = os.path.join(os.path.dirname(__file__), "..", "results")
d = json.load(open(os.path.join(R, "coverage_eval.json")))
SER = [("main", "main half", C["ours"], "o"), ("reserve", "reserve half", C["base"], "s"), ("dam_D16", "damage D16", C["third"], "^")]
fig, ax = plt.subplots(figsize=(W_FULL, 2.5))
for key, name, col, mk in SER:
    for k, ls, lab in (("law", "-", "growth law"), ("conf", "--", "split conformal")):
        bd = d[key][k]["by_distance"]; xs = sorted(bd, key=float)
        ax.plot([float(x) + 1.25 for x in xs], [bd[x][0] for x in xs], ls=ls, marker=mk, color=col, ms=3.2, lw=1.1, label=f"{name}, {lab}")
ax.axhline(0.95, color=C["ink2"], lw=0.6); ax.axhline(0.90, color=C["ink2"], lw=0.6, ls=":")
ax.set_xlabel(r"Distance from training edge $d$ ($^\circ$C)"); ax.set_ylabel("Coverage of 95% band")
ax.set_ylim(0.55, 1.0); ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=5.8, ncol=2, handlelength=2.0, columnspacing=1.0)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fig5_coverage3.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200); print(out)
