"""
Figure 1: distribution of test accuracy over random 70/30 splits for each leakage
pipeline (Random Forest, paper hyperparameters) against the paper's 97.78%.

Requires results/leakage_factorial_per_split.csv (run 02_... first).
Output: results/fig1_leakage_accuracy_ranges.png
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(HERE, "results", "legacy")
PAPER_ACC = 0.9778
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {False: "#2a78d6", True: "#eb6834"}  # categorical slots 1-2: time excluded / included

per = pd.read_csv(os.path.join(OUT, "leakage_factorial_per_split.csv"))
d = per[(per.model == "RF") & (per.test_subset == "all_test_rows")]
n_splits = d.seed.nunique()

groups = [(False, "train_only", "No outlier removal\nSMOTE on training fold only"),
          (True, "train_only", "IQR outlier removal (full data)\nSMOTE on training fold only"),
          (False, "global", "No outlier removal\nSMOTE on full data before split"),
          (True, "global", "IQR outlier removal (full data)\nSMOTE on full data before split")]

fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
yt, yl = [], []
for gi, (O, S, label) in enumerate(groups):
    for T in (False, True):
        y = gi * 3 + (0.55 if T else -0.55)
        a = d[(d.outlier_removal == O) & (d.smote == S) & (d.time_feature == T)].accuracy
        lo, hi = a.quantile([0.05, 0.95])
        ax.plot([lo, hi], [y, y], color=COLORS[T], lw=6, solid_capstyle="round", zorder=2)
        ax.scatter(a.mean(), y, s=70, color=SURFACE, edgecolor=COLORS[T], lw=2, zorder=3)
        ax.scatter(a.max(), y, marker="|", s=160, color=COLORS[T], lw=2, zorder=3)
    yt.append(gi * 3); yl.append(label)

ax.axvline(PAPER_ACC, color=INK, ls="--", lw=1.2, zorder=1)
ax.text(PAPER_ACC - 0.003, 10.3, "Paper reports 97.78%", ha="right", va="center", fontsize=9, color=INK)
ax.set_yticks(yt, yl, fontsize=8.5, color=INK2)
ax.set_ylim(-1.6, 10.9)
ax.set_xlim(0.62, 1.0)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel(f"Test accuracy over {n_splits:,} random 70/30 splits (Random Forest, paper hyperparameters)", fontsize=9, color=INK2)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, length=0)

from matplotlib.lines import Line2D
handles = [Line2D([], [], color=COLORS[False], lw=6, label="time excluded"),
           Line2D([], [], color=COLORS[True], lw=6, label="time included as feature (leak)"),
           Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, label="mean"),
           Line2D([], [], marker="|", ls="", color=INK2, markersize=12, label="best single split"),
           Line2D([], [], color=INK2, lw=6, alpha=0.35, label="5th-95th percentile")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=5, fontsize=8, frameon=False, labelcolor=INK2, handlelength=1.6, columnspacing=1.2)
ax.set_title("No combination of leaks reproduces the reported 97.78% accuracy", loc="left", fontsize=12, color=INK, pad=12)
plt.tight_layout()
path = os.path.join(OUT, "fig1_leakage_accuracy_ranges.png")
plt.savefig(path, dpi=300, facecolor=SURFACE)
print("saved", path)
