#!/usr/bin/env python3
"""Figure 1: overview of the method - what is computed from which data."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures.style import plt, C, W_FULL
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(W_FULL, 2.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
W, H = 17.0, 16.0
def box(x, y, title, sub, col, w=W):
    ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec=col, lw=1.2))
    ax.text(x + w / 2, y + H - 2.2, title, ha="center", va="top", fontsize=6.8, fontweight="bold", color=C["ink"])
    ax.text(x + w / 2, y + H - 6.6, sub, ha="center", va="top", fontsize=5.9, color=C["ink2"], linespacing=1.3)
def arrow(x0, y0, x1, y1, col=None):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=col or C["ink2"], shrinkA=0, shrinkB=0))
X = [1, 21, 41, 61, 81]; YT, YB = 29, 5
box(X[0], YT, "Training block", "records at known\nconditions;\nedge record", C["muted"])
box(X[0], YB, "Plate model", "bicharacteristic-\nsplitting solver", C["muted"])
box(X[1], YB, "Packet windows", "ray arrival times\nand mode labels", C["ours"])
box(X[1], YT, "Delay tracking", "delay of every\nwindow in every\ntraining record", C["ours"])
box(X[2], YT, "Delay laws", "one linear law in\nthe condition\nper window", C["ours"])
box(X[3], YT, "Prediction", "edge record moved\nalong the delay\nfield + residual", C["ours"])
box(X[2], YB, "Validation block", "fit on inner part;\nresidual growth\nlaw in distance", C["base"])
box(X[3], YB, "Intervals", "width grows with\nthe distance\nfrom the edge", C["base"])
box(X[4], 17, "Mixing index", "from training\ndata only: where\na single stretch\nfails", C["third"])
y1, y2 = YT + H / 2, YB + H / 2
arrow(X[0] + W + 0.4, y1, X[1] - 0.4, y1); arrow(X[0] + W + 0.4, y2, X[1] - 0.4, y2)
arrow(X[1] + W / 2, YB + H + 0.4, X[1] + W / 2, YT - 0.4); arrow(X[1] + W + 0.4, y1, X[2] - 0.4, y1)
arrow(X[2] + W + 0.4, y1, X[3] - 0.4, y1); arrow(X[2] + W + 0.4, y2, X[3] - 0.4, y2)
arrow(X[2] + W / 2, YT - 0.4, X[2] + W / 2, YB + H + 0.4); arrow(X[3] + W / 2, YT - 0.4, X[3] + W / 2, YB + H + 0.4)
ax.annotate("", (X[4] + W / 2, 17 + H + 0.4), (X[2] + W / 2 + 3, YT + H + 0.4),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C["third"], connectionstyle="arc3,rad=-0.25", shrinkA=0, shrinkB=0))
out = os.path.join(os.path.dirname(__file__), "fig1_pipeline.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200); print(out)
