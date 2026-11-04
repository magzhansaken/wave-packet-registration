#!/usr/bin/env python3
"""Figure 3: data. (a) plate with the 12 transducers, all 66 paths, the 12 fixed paths of tuning and ablations, and the path
of Figure 2; (b) plate temperature against acquisition time for the three sets of D1 (main half, reserve half, dam_D16)
with the training/test thresholds of splits S1-S6. Temperatures are read from the slices (mean of the two plate sensors).
Numbers for the text: results/fig2_data.json."""
import glob, json, os, sys
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL, panel
from bcs import ogw
from experiments.e2_run import subset_paths

ROOT = os.environ.get("M1_ROOT", ".")
SETS = [("main half", f"{ROOT}/data/d1", C["ours"], "o"), ("reserve half", f"{ROOT}/data/d1_reserve", C["base"], "s"),
        ("dam_D16 (held out)", f"{ROOT}/data/d1_dam", C["third"], "^")]


def temps(root, fk=40):
    out = []
    for f in sorted(glob.glob(os.path.join(root, f"{fk}kHz", "*.npz"))):
        if os.path.basename(f).startswith("_"):
            continue
        d = np.load(f); out.append((datetime.strptime(os.path.basename(f)[:15], "%Y%m%dT%H%M%S"), float(np.mean(d["T_plate"]))))
    return sorted(out)


ch = ogw.load_freq(40, max_files=1)["channels"]; sub = subset_paths(ch); L, _ = ogw.path_geometry(ch)
i_fig1 = int(np.argsort(np.abs(L - np.median(L)))[0])
fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw={"width_ratios": [1, 1.35]})
ax = axs[0]; panel(ax, "(a)")
ax.add_patch(plt.Rectangle((0, 0), 500, 500, fill=False, ec=C["ink2"], lw=0.8))
for k, (a, b) in enumerate(ch):
    pa, pb = ogw.XY[a] * 1e3, ogw.XY[b] * 1e3
    if k == i_fig1:
        st = dict(color=C["base"], lw=1.4, zorder=4)
    elif k in sub:
        st = dict(color=C["ours"], lw=0.9, zorder=3)
    else:
        st = dict(color="#cfcec9", lw=0.4, zorder=1)
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], **st)
for n, (x, y) in enumerate(ogw.XY * 1e3):
    ax.plot(x, y, "o", ms=4, mfc="white", mec=C["ink"], mew=0.8, zorder=5)
    ax.text(x, y + (42 if y > 250 else -42), f"T{n + 1}", ha="center", va="center", fontsize=6)
ax.plot([], [], color="#cfcec9", lw=0.8, label="66 paths"); ax.plot([], [], color=C["ours"], lw=0.9, label="12 fixed paths")
ax.plot([], [], color=C["base"], lw=1.4, label="path of Figure 2")
ax.set_xlim(-25, 525); ax.set_ylim(-70, 570); ax.set_aspect("equal"); ax.grid(False)
ax.set_xlabel("$x$ (mm)"); ax.set_ylabel("$y$ (mm)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=5.8, handlelength=1.2, columnspacing=0.8)

ax = axs[1]; panel(ax, "(b)"); info = {}
for name, root, col, mk in SETS:
    tt = temps(root); t0 = tt[0][0]
    h = np.array([(t - t0).total_seconds() / 3600 for t, _ in tt]); T = np.array([x for _, x in tt])
    ax.plot(h, T, mk, color=col, ms=1.8, mew=0, label=name)
    info[name] = {"n": len(T), "T_min": float(T.min()), "T_max": float(T.max()), "hours": float(h.max()),
                  "start": tt[0][0].isoformat(), "end": tt[-1][0].isoformat()}
for thr in (30, 35, 40, 45, 50):
    ax.axhline(thr, color=C["ink2"], lw=0.5, ls="--")
ax.set_xlabel("Time from first measurement (h)"); ax.set_ylabel(r"Plate temperature ($^\circ$C)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=5.8, markerscale=2, handlelength=0.8, columnspacing=0.8)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fig2_data.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
info["fixed_paths"] = [int(k) for k in sub]; info["fig1_path"] = i_fig1
json.dump(info, open(os.path.join(os.path.dirname(__file__), "..", "results", "fig2_data.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
