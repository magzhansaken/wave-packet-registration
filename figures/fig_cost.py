#!/usr/bin/env python3
"""Figures 7 and A2: accuracy against cost on the 36 neural units. x: median CPU time per unit under equal load (one thread,
timing_eval.json; br and br_gp include their share of the per-task ray tables and slopes); y: median relative L2 error
on the 36 units of the main half (filled) and of the damaged set (open), neural_eval.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL

R = os.path.join(os.path.dirname(__file__), "..", "results")
tm = json.load(open(os.path.join(R, "timing_eval.json"))); ne = json.load(open(os.path.join(R, "neural_eval.json")))
pre = tm["prep_cpu_s_per_unit"]
M = [("copy", "copy", C["muted"], "o"), ("knn", "kNN", C["muted"], "s"), ("gp", "GP", C["muted"], "^"),
     ("bss", "stretch", C["base"], "o"), ("bss_tr", "tracked stretch", C["base"], "s"),
     ("bss_gp", "stretch + GP", C["base"], "^"), ("bss_tr_gp", "tracked stretch + GP", C["base"], "D"),
     ("br", "registration", C["ours"], "s"), ("br_gp", "registration + GP (ours)", C["ours"], "o"),
     ("mfbno", "MF-B-NO", C["third"], "D"), ("fvae", "f-VAE", C["fourth"], "v")]
ERR = {"mfbno": "mfbno_bss", "fvae": "fvae"}
def draw(ax, legend=False):
    for m, name, col, mk in M:
        x = tm["median_cpu_s"][m] + (pre if m in ("br", "br_gp") else 0.0)
        for st, fill in (("main", True), ("dam_D16", False)):
            y = ne[st][ERR[m]]["median"] if m in ERR else ne[st]["median_err_on_units"][m]
            ax.plot([x], [y], marker=mk, ms=5.5 if m == "br_gp" else 4.5, color=col, mfc=col if fill else "white", mew=1.1,
                    ls="none", zorder=4 if m == "br_gp" else 3, label=name if fill else None)
    ax.set_xscale("log"); ax.set_xlabel("CPU time per unit (s, one thread)"); ax.set_ylabel(r"Median relative $L_2$ error ($-$)")


def legend(fig, ax, ncol, y):
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", color=C["ink2"], ls="none", ms=4.5), plt.Line2D([], [], marker="o", color=C["ink2"], mfc="white", ls="none", ms=4.5)]
    l += ["main half", "damage D16"]
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=6.6, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, y, 1, 1))


# main text: the group of methods with errors below 0.36
fig, ax = plt.subplots(figsize=(W_FULL, 3.0)); draw(ax)
ax.set_xlim(0.06, 120); ax.set_ylim(0.305, 0.355); ax.minorticks_off()
for a in list(ax.lines):
    if a.get_ydata()[0] > 0.36:
        a.remove()
legend(fig, ax, 4, 0.2)
out = os.path.join(os.path.dirname(__file__), "fig6_cost.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200); print(out)
# Appendix C: all methods
fig, ax = plt.subplots(figsize=(W_FULL, 3.0)); draw(ax)
ax.set_yscale("log"); ax.set_yticks([0.3, 0.5, 0.7, 1.0, 1.3]); ax.set_yticklabels(["0.3", "0.5", "0.7", "1.0", "1.3"])
ax.minorticks_off(); ax.set_ylim(0.27, 1.6)
legend(fig, ax, 5, 0.17)
out = os.path.join(os.path.dirname(__file__), "figC2_cost_all.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200); print(out)
