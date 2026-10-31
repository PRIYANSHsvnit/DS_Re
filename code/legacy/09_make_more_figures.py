"""
Figures 2 and 4 (need outputs of 06_external_validation.py).
  fig2_external_validation.png : discrimination internal (UCI CV) vs external (SUPPORT-CHF) vs within-SUPPORT ceiling,
                                 plus calibration (predicted vs observed 180-day mortality)
  fig4_hazard_ratios_uci_vs_support.png : shared-predictor Cox HRs in both cohorts
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(HERE, "results", "legacy")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


# ---------- Figure 2 ----------
s = pd.read_csv(os.path.join(OUT, "external_validation_summary.csv"), index_col=0)
order = ["CoxPH", "RSF", "GBSurv", "RF_classifier"]
labels = {"CoxPH": "Cox PH", "RSF": "Random survival forest", "GBSurv": "Gradient-boosted survival", "RF_classifier": "RF classifier"}
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.35, 1]))
style(a)
for i, m in enumerate(order):
    y = len(order) - 1 - i
    vals = [s.loc[m, "internal_harrell_c"], s.loc[m, "external_harrell_c"], s.loc[m, "within_support_cv_harrell_c"]]
    a.plot([min(vals), max(vals)], [y, y], color=GRID, lw=2, zorder=1)
    for v, c in zip(vals, (C1, C2, C3)):
        a.scatter(v, y, s=70, color=c, edgecolor=SURFACE, lw=1.5, zorder=3)
a.axvline(0.5, color=INK2, ls=":", lw=1)
a.text(0.502, len(order) - 0.45, "chance", fontsize=8, color=INK2)
a.set_yticks(range(len(order)), [labels[m] for m in order[::-1]], fontsize=9, color=INK2)
a.set_xlim(0.48, 0.72); a.set_ylim(-0.6, len(order) - 0.3)
a.set_xlabel("Harrell's C-index (age, sex, creatinine, sodium, diabetes)", fontsize=9, color=INK2)
a.set_title("A. Discrimination", loc="left", fontsize=11, color=INK)
a.legend(handles=[Line2D([], [], marker="o", ls="", color=C1, label="UCI internal (10x5 CV)"),
                  Line2D([], [], marker="o", ls="", color=C2, label="UCI model on SUPPORT-CHF"),
                  Line2D([], [], marker="o", ls="", color=C3, label="trained within SUPPORT (ceiling)")],
         loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=8, frameon=False, labelcolor=INK2)

style(b)
cal = ["CoxPH", "RSF", "GBSurv"]
for i, m in enumerate(cal):
    y = len(cal) - 1 - i
    obs, pred_ext, pred_in = s.loc[m, "external_observed_180d_mortality"], s.loc[m, "external_mean_predicted_180d_mortality"], s.loc[m, "within_support_cv_mean_predicted_180d_mortality"]
    b.plot([obs, pred_ext], [y, y], color=GRID, lw=2, zorder=1)
    b.scatter(obs, y, s=90, marker="|", color=INK, lw=2.5, zorder=3)
    b.scatter(pred_ext, y, s=70, color=C2, edgecolor=SURFACE, lw=1.5, zorder=3)
    b.scatter(pred_in, y, s=70, color=C3, edgecolor=SURFACE, lw=1.5, zorder=2)
    b.text(pred_ext + 0.008, y + 0.12, f"slope {s.loc[m, 'external_calibration_slope']:.2f}", fontsize=8, color=INK2)
b.set_yticks(range(len(cal)), [labels[m] for m in cal[::-1]], fontsize=9, color=INK2)
b.set_xlim(0.2, 0.52); b.set_ylim(-0.6, len(cal) - 0.3)
b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
b.set_xlabel("180-day mortality in SUPPORT-CHF", fontsize=9, color=INK2)
b.set_title("B. Calibration", loc="left", fontsize=11, color=INK)
b.legend(handles=[Line2D([], [], marker="|", ls="", color=INK, markersize=10, mew=2.5, label="observed"),
                  Line2D([], [], marker="o", ls="", color=C2, label="UCI model"),
                  Line2D([], [], marker="o", ls="", color=C3, label="within-SUPPORT model")],
         loc="upper center", bbox_to_anchor=(0.32, -0.17), ncol=3, fontsize=8, frameon=False, labelcolor=INK2, columnspacing=1.0)
fig.suptitle("External validation on 1,384 real heart-failure patients (SUPPORT-CHF)", x=0.01, ha="left", fontsize=12, color=INK)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_external_validation.png"), dpi=300, facecolor=SURFACE)

# ---------- Figure 4 ----------
h = pd.read_csv(os.path.join(OUT, "cox_hr_uci_vs_support.csv"))
names = {"age": "Age (per SD)", "sex": "Male sex", "log_creatinine": "Serum creatinine (per SD, log)", "serum_sodium": "Serum sodium (per SD)", "diabetes": "Diabetes"}
vars_ = ["age", "log_creatinine", "serum_sodium", "sex", "diabetes"]
fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE); style(ax)
for i, v in enumerate(vars_):
    for k, (coh, c, dy) in enumerate((("UCI", C1, 0.17), ("SUPPORT", C2, -0.17))):
        r = h[(h.variable == v) & h.cohort.str.startswith(coh)].iloc[0]
        y = len(vars_) - 1 - i + dy
        ax.plot([r.CI_low, r.CI_high], [y, y], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter(r.HR, y, s=55, color=c, edgecolor=SURFACE, lw=1.5, zorder=3)
ax.axvline(1, color=INK2, ls="--", lw=1)
ax.set_xscale("log"); ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6, 2]); ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6", "2"])
ax.set_yticks(range(len(vars_)), [names[v] for v in vars_[::-1]], fontsize=9, color=INK2)
ax.set_xlabel("Hazard ratio for death (95% CI), multivariable Cox", fontsize=9, color=INK2)
ax.set_title("Risk factors: development cohort vs independent cohort", loc="left", fontsize=12, color=INK)
ax.legend(handles=[Line2D([], [], color=C1, lw=2.2, marker="o", label="UCI Faisalabad (n = 299)"),
                   Line2D([], [], color=C2, lw=2.2, marker="o", label="SUPPORT-CHF (n = 1,384)")],
          loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, fontsize=8, frameon=False, labelcolor=INK2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_hazard_ratios_uci_vs_support.png"), dpi=300, facecolor=SURFACE)
print("saved fig2 and fig4")
