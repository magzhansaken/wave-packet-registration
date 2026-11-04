"""One palette and one typography for all figures (MDPI single column 13.6 cm wide).
Categorical order (validated reference palette): blue = ours, orange = strongest baseline, aqua = third series."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"ours": "#2a78d6", "base": "#eb6834", "third": "#1baf7a", "fourth": "#eda100", "muted": "#8a8984",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
W_FULL = 13.6 / 2.54            # inches
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "mathtext.fontset": "dejavusans", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def panel(ax, s):
    """Panel letter at the top left of an axis."""
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=C["ink"])
