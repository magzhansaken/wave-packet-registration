#!/usr/bin/env python3
"""Figure 8: median mode-mixing index (training blocks only) against the median gain of the registration over the
strongest stretch baseline, by frequency, on the main and reserve halves. Data: results/discussion_eval.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL

R = os.path.join(os.path.dirname(__file__), "..", "results")
d = json.load(open(os.path.join(R, "discussion_eval.json")))
fig, ax = plt.subplots(figsize=(W_FULL, 2.6))
ax.axvspan(0.38, 0.75, color=C["grid"], alpha=0.6, lw=0, zorder=0)
ax.axhline(0, color=C["ink2"], lw=0.6)
for key, name, col, mk in (("main", "main half", C["ours"], "o"), ("reserve", "reserve half", C["base"], "s")):
    pf = d[key]["per_freq"]
    xs = [pf[f]["median_emix"] for f in pf]; ys = [pf[f]["gain_pct"] for f in pf]
    ax.plot(xs, ys, marker=mk, ls="none", color=col, ms=4.5, label=name, zorder=3)
    if key == "main":
        for f in pf:
            if int(f) in (40, 60, 80, 100):
                ax.annotate(f"{f} kHz", (pf[f]["median_emix"], pf[f]["gain_pct"]), xytext=(6, -2), textcoords="offset points",
                            fontsize=6.6, color=C["ink"])
ax.text(0.73, -18, "mixing band", ha="right", fontsize=6.6, color=C["ink2"])
ax.set_xlabel(r"Median mode-mixing index $e_{\mathrm{mix}}$ ($-$)"); ax.set_ylabel("Median gain over strongest stretch (%)")
ax.set_xlim(0.05, 0.75); ax.legend(loc="upper left", fontsize=6.8)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fig7_mixing.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200); print(out)
