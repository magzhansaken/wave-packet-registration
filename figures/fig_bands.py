#!/usr/bin/env python3
"""Figure 5: median relative gain of br_gp over the strongest stretch baseline by frequency on the three sets of D1
(95 % unit-bootstrap intervals), with the mode-mixing band shaded. Data: results/bands_eval.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL

R = os.path.join(os.path.dirname(__file__), "..", "results")
d = json.load(open(os.path.join(R, "bands_eval.json")))
SER = [("main", "main half", C["ours"], "o", -4), ("reserve", "reserve half", C["base"], "s", 0), ("dam_D16", "damage D16 (held out)", C["third"], "^", 4)]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(W_FULL, 2.7), gridspec_kw={"width_ratios": [2.3, 1]})
from figures.style import panel
ax.axvspan(50, 110, color=C["grid"], alpha=0.8, lw=0, zorder=0)
ax.text(80, 33.5, "mode-mixing band", ha="center", va="bottom", fontsize=7, color=C["ink2"])
ax.axhline(0, color=C["ink2"], lw=0.7)
for key, name, col, mk, dx in SER:
    pf = d[key]["per_freq"]; fs = sorted(int(f) for f in pf)
    m = np.array([pf[str(f)]["median_rel_pct"] for f in fs]); lo = np.array([pf[str(f)]["ci_pct"][0] for f in fs]); hi = np.array([pf[str(f)]["ci_pct"][1] for f in fs])
    x = np.array(fs) + dx
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=mk, color=col, ms=3.8, lw=0.9, capsize=1.5, label=name, zorder=3)
ax.set_xticks(range(40, 261, 20)); ax.set_xlim(28, 272); ax.set_ylim(-28, 37)
ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("Median gain over strongest stretch (%)")
ax.legend(loc="upper right", fontsize=6.5, ncol=1, handlelength=1.0)
panel(ax, "(a)"); panel(bx, "(b)")
# (b): distribution of the per-unit gain by band (violin), same units and baseline as Table 4
from experiments import analyze as A
P = lambda f: os.path.join(R, f)
SETS = {"main": lambda: A.load(P("e2_rows.jsonl"), [P("e2_rows_bss_tr.jsonl"), P("e2_rows_brgp_fix.jsonl")]) + A.load(P("e2_rows_confirm.jsonl")),
        "reserve": lambda: A.load(P("e2_rows_reserve.jsonl")), "dam_D16": lambda: A.load(P("e2_rows_dam.jsonl"))}
BANDS = [("40", lambda f: f == 40), ("60–100", lambda f: 60 <= f <= 100), ("120–260", lambda f: f >= 120)]
for j, (key, name, col, mk, dx) in enumerate(SER):
    rows = SETS[key](); A.add_best_bss([r for r in rows if "bss_best" not in r["m"]])
    rows = [r for r in rows if "bss_best" in r["m"] and "br_gp" in r["m"]]
    for b, (bn, fn) in enumerate(BANDS):
        g = np.array([100 * (r["m"]["bss_best"]["err"] - r["m"]["br_gp"]["err"]) / r["m"]["bss_best"]["err"] for r in rows if fn(r["f_kHz"])])
        v = bx.violinplot([np.clip(g, -60, 60)], positions=[b + (j - 1) * 0.27], widths=0.25, showextrema=False, showmedians=True)
        for pc in v["bodies"]:
            pc.set_facecolor(col); pc.set_edgecolor(col); pc.set_alpha(0.55)
        v["cmedians"].set_color(C["ink"]); v["cmedians"].set_linewidth(0.9)
bx.axhline(0, color=C["ink2"], lw=0.7); bx.set_xticks(range(3)); bx.set_xticklabels([b[0] for b in BANDS]); bx.set_xlabel("Band (kHz)")
bx.set_ylabel("Gain per unit (%)"); bx.set_ylim(-62, 62)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fig4_bands.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200); print(out)
