"""
Abstract-level analysis of published results on the UCI Heart Failure Clinical Records dataset (C6).

Input : results/literature/corpus_all_citing.csv (from 05_build_literature_corpus.py)
Coding: CODES below was coded by reading the abstract of every candidate that states a performance
        number (n = 90; AI-assisted, abstract only). Headline = best number the abstract reports for
        THIS dataset. Flags are only set when the abstract states them explicitly:
          time   'Y'  follow-up time is named as a model input / top predictor
          resamp 'Y'  SMOTE / oversampling / resampling is used (order vs split not knowable from abstracts)
          surv   'Y'  censoring-aware survival methods (Cox, RSF, C-index, time-dependent AUC)
        status: inc = included; d5000 = 5,000-row synthetic Kaggle version; dup = duplicate publication;
                exc = excluded (other dataset / multi-dataset average / not a prediction study / unclear)
Output: results/literature/abstract_coding.csv, results/literature/abstract_summary.csv,
        results/fig3_literature_vs_achievable.png
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIT = os.path.join(HERE, "results", "legacy", "literature")
OUT = os.path.join(HERE, "results", "legacy")

# id: (status, headline_acc, headline_auc, time, resamp, surv, note)
CODES = {
    "W7203480858": ("d5000", .997, .9998, "", "Y", "", "5,000-row version; SMOTE; chi2 feature selection"),
    "W4408537479": ("d5000", .992, None, "", "Y", "", "5,000-row version; SMOTE"),
    "W4210695379": ("inc", .991, None, "Y", "", "", "follow-up time among most significant features; XGBoost"),
    "W4413403817": ("exc", None, None, "", "", "", "headline numbers are for Cleveland"),
    "W4308409114": ("inc", .9889, None, "", "", "", "LEBoosting"),
    "W4288075586": ("exc", None, None, "", "", "", "two heart-disease datasets, not clearly this one"),
    "W4390604002": ("exc", None, None, "", "", "", "average over UCI/Faisalabad/Framingham"),
    "W4411996149": ("exc", None, None, "", "", "", "headline on a 1,025-record dataset"),
    "W4389319122": ("exc", None, None, "", "", "", "'Kaggle heart failure' ambiguous (likely 918-row set)"),
    "W7224308719": ("exc", None, None, "", "", "", "920-record UCI heart-disease data mislabelled"),
    "W7160044626": ("inc", .971, .986, "", "", "", "hybrid feature selection + ensemble"),
    "W7222408188": ("dup", .971, .986, "", "Y", "", "same results as W7160044626; class-imbalance correction"),
    "W3217225149": ("inc", .9667, None, "", "", "", "GA + ANFIS"),
    "W4366375058": ("inc", .9667, None, "", "", "", "SVM"),
    "W4411003116": ("inc", .962, .987, "", "", "", "RF with Cox-derived feature selection"),
    "W3203001841": ("inc", .96, None, "Y", "", "", "follow-up month used as key feature"),
    "W4200034720": ("inc", .95, None, "", "", "", "RF, optimised hyperparameters"),
    "W4407925212": ("inc", .9487, None, "", "", "", "multi-objective hybrid"),
    "W7222651756": ("exc", None, None, "", "", "", "dataset not identifiable from abstract"),
    "W3216962276": ("exc", None, None, "", "", "", "different cohort (303 CAD patients)"),
    "W4407360851": ("inc", .94, None, "", "Y", "", "SMOTE + chi2 selection + GBM"),
    "W3216223959": ("inc", .939, .925, "Y", "", "", "MLP; time most significant variable"),
    "W4403047362": ("inc", .939, None, "", "Y", "", "SMOTE + PSO + RF (88.3% without)"),
    "W7133187981": ("exc", None, None, "", "", "", "aggregated multiple datasets"),
    "W3213642092": ("inc", .9333, None, "", "", "", "k-means + SVM"),
    "W7166797777": ("inc", .93, None, "", "", "", "SVM-DEA-CSA"),
    "W3126592173": ("inc", .789, None, "", "", "", "BioExpDNN; 78.9% for this dataset"),
    "W4393028357": ("inc", .92, None, "", "", "", "RF"),
    "W4387697609": ("inc", .9111, None, "Y", "", "", "NN; time strongest risk factor"),
    "W4392082202": ("inc", .9111, None, "", "", "", "BPSO + MLP"),
    "W7138681251": ("inc", .9101, None, "Y", "", "", "stacking; follow-up time an input"),
    "W4407177677": ("exc", None, None, "", "", "", "four datasets"),
    "W4221074700": ("inc", .90, None, "", "Y", "", "SMOTE-ENN + RF"),
    "W4401322862": ("exc", None, None, "", "", "", "Cleveland"),
    "W4292651891": ("inc", .90, None, "", "Y", "", "RFE + SMOTE + SVM"),
    "W4389275724": ("inc", .90, None, "", "Y", "", "SMOTE + RF"),
    "W4416830620": ("inc", .90, None, "", "Y", "", "SMOTE + RF"),
    "W4311110841": ("inc", .896, None, "", "", "", "DT"),
    "W4317615006": ("inc", .89, None, "", "", "", "filter-wrapper selection + SVM"),
    "W7160696220": ("d5000", .882, None, "", "Y", "", "5,000-row version; RUS/SMOTE + CART"),
    "W4406530537": ("inc", .88, None, "", "", "", "AutoML; UCI HF diagnosis"),
    "W3201708182": ("inc", .8788, None, "", "Y", "", "KNN with SMOTE (71.6% CV without)"),
    "W4386753624": ("inc", .8721, None, "Y", "", "", "RF with AUC filter; time among rule attributes"),
    "W4406238075": ("inc", .87, None, "Y", "Y", "", "four predictors incl. follow-up time; balanced training set"),
    "W7216675827": ("inc", .87, None, "", "", "", "kNN"),
    "W3217760844": ("inc", .8662, .8301, "", "", "", "hybrid classifier"),
    "W4413557064": ("inc", .8644, None, "", "Y", "", "SMOTE + NN, 10-fold CV"),
    "W7203768738": ("inc", .8633, None, "", "Y", "", "SMOTE + DECORATE"),
    "W3122377581": ("inc", .9431, None, "", "Y", "", "RF with resample (85.8% with SMOTE)"),
    "W4285101279": ("inc", .85, .93, "", "", "", "LightGBM"),
    "W4316924427": ("inc", .8461, None, "", "", "", "hybrid FS + KNN, 10-fold CV"),
    "W4416336299": ("inc", .846, .916, "", "Y", "", "SMOTE + RF"),
    "W7161590067": ("inc", .8333, .879, "", "", "", "RF"),
    "W7196932547": ("inc", .8267, None, "", "", "", "DT/NB/SVM"),
    "W7168087432": ("inc", .8227, .74, "", "", "", "PSO + random tree"),
    "W4385059333": ("inc", .819, None, "", "", "", "NB, 3 features"),
    "W4413095987": ("inc", .8167, .8446, "Y", "", "", "mixture of experts; time top feature"),
    "W3183411741": ("inc", .80, None, "", "", "", "DT on EF + SC"),
    "W4417031350": ("inc", .80, .872, "Y", "", "", "logistic regression; follow-up time strongest"),
    "W4406470264": ("exc", None, None, "", "", "", "dataset size not stated"),
    "W7132893313": ("exc", None, None, "", "", "", "outlier-detection thesis"),
    "W4315498630": ("exc", None, None, "", "", "", "different cohort (500 patients)"),
    "W4223657649": ("inc", None, None, "", "", "Y", "IPCW survival ML; headline metric unclear"),
    "W4395070038": ("inc", None, .96, "", "Y", "", "SMOTE-ENN + RF"),
    "W4414544694": ("inc", None, .988, "", "", "Y", "RSF C-index 0.884, AUC 0.988"),
    "W4415363018": ("exc", None, None, "", "", "", "'UCI' data with NT-proBNP: not this dataset"),
    "W4416750830": ("inc", None, .79, "", "", "", "RF"),
    "W7140203826": ("inc", .80, .837, "", "", "", "LR / RF"),
    "W7167710409": ("exc", None, None, "", "", "", "synthetic data release, not a study"),
    "W7203489492": ("inc", None, .82, "", "", "Y", "survival + calibration study"),
    "W7203498345": ("inc", None, .752, "", "", "Y", "LR + Cox"),
    "W7215037997": ("inc", None, .789, "", "", "", "10x10 CV; simple models match ensembles"),
    "W7169717037": ("exc", None, None, "Y", "", "", "governance demo; v1 leaked time (AUC 0.910), corrected in v2"),
    "W4413933837": ("inc", None, .85, "Y", "", "", "Shiny app; follow-up time an input"),
    "W7197006858": ("inc", None, .73, "", "", "Y", "Rane et al. discrete-time survival"),
    "W7212013904": ("inc", None, .77, "", "", "", "ElasticNet / XGBoost"),
    "W7143409914": ("inc", None, .907, "", "", "", "ensemble, 5-fold CV"),
    "W7161965071": ("inc", None, .91, "", "", "", "RF pooled out-of-fold"),
    "W7223665190": ("exc", None, None, "", "", "", "uses this dataset only as external validation"),
    "W7221625498": ("exc", None, None, "", "", "", "HF headline not given"),
    "W7223218682": ("exc", None, None, "", "", "", "dataset not identifiable"),
    "W7160697621": ("inc", .80, .83, "Y", "", "", "LR / RF; follow-up time top predictor"),
    "W7167467963": ("inc", .91, .94, "", "Y", "", "Tomek links / SMOTE + RF"),
    "W4409225031": ("inc", .8455, .9134, "", "", "", "ensemble"),
    "W4396781314": ("exc", None, None, "", "", "", "SPECT amyloidosis"),
    "W4402042129": ("exc", None, None, "", "", "", "MIMIC-IV"),
    "W4409512086": ("exc", None, None, "", "", "", "community cohort"),
    "W7223549211": ("exc", None, None, "", "", "", "metabolomics cohort"),
    "W4401371609": ("exc", None, None, "", "", "", "CHD + hypertension cohort"),
    "W4225153219": ("exc", None, None, "", "", "", "AF nomogram"),
}
# Leakage-free reference: most favourable of 7 honest pipelines x 1,000 splits (08_honest_reference.py)
_ref = pd.read_csv(os.path.join(OUT, "honest_reference.csv"), index_col=0).loc["MOST FAVOURABLE (reference)"]
HONEST = dict(acc_mean=_ref.acc_mean, acc_p95=_ref.acc_p95, acc_max=_ref.acc_max, auc_mean=_ref.auc_mean, auc_p95=_ref.auc_p95, auc_max=_ref.auc_max)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_PLAIN, C_FLAG = "#2a78d6", "#eb6834"


def main():
    corpus = pd.read_csv(os.path.join(LIT, "corpus_all_citing.csv"))
    corpus["wid"] = corpus.openalex_id.str.split("/").str[-1]
    rows = [dict(wid=k, status=v[0], acc=v[1], auc=v[2], time_used=v[3] == "Y", resampling=v[4] == "Y", survival=v[5] == "Y", note=v[6])
            for k, v in CODES.items()]
    code = pd.DataFrame(rows).merge(corpus[["wid", "year", "title", "venue", "doi"]], on="wid", how="left")
    code.to_csv(os.path.join(LIT, "abstract_coding.csv"), index=False)

    inc = code[code.status == "inc"]
    acc, auc = inc.acc.dropna(), inc.auc.dropna()
    summ = {
        "abstracts_coded": len(code), "included_studies": len(inc), "excluded": int((code.status == "exc").sum()),
        "duplicates": int((code.status == "dup").sum()), "uses_5000_row_version": int((code.status == "d5000").sum()),
        "n_reporting_accuracy": len(acc), "median_accuracy": acc.median(), "iqr_accuracy": f"{acc.quantile(.25):.3f}-{acc.quantile(.75):.3f}",
        "n_accuracy_above_honest_p95": int((acc > HONEST["acc_p95"]).sum()), "share_accuracy_above_honest_p95": (acc > HONEST["acc_p95"]).mean(),
        "n_accuracy_above_best_honest_split": int((acc > HONEST["acc_max"]).sum()),
        "n_auc_above_best_honest_split": int((auc > HONEST["auc_max"]).sum()),
        "n_accuracy_ge_0.90": int((acc >= .90).sum()), "n_accuracy_ge_0.95": int((acc >= .95).sum()),
        "n_reporting_auc": len(auc), "median_auc": auc.median(),
        "n_auc_above_honest_p95": int((auc > HONEST["auc_p95"]).sum()), "share_auc_above_honest_p95": (auc > HONEST["auc_p95"]).mean(),
        "share_time_flag": inc.time_used.mean(), "share_resampling_flag": inc.resampling.mean(), "share_survival_methods": inc.survival.mean(),
        "median_acc_time_flag": inc[inc.time_used].acc.median(), "median_acc_resampling_flag": inc[inc.resampling].acc.median(),
        "median_acc_no_flag": inc[~inc.time_used & ~inc.resampling].acc.median(),
    }
    s = pd.Series(summ)
    s.to_csv(os.path.join(LIT, "abstract_summary.csv"), header=["value"])
    print(s.to_string())

    # Figure 3: published headline accuracy vs achievable
    d = inc.dropna(subset=["acc"]).sort_values("acc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=SURFACE); ax.set_facecolor(SURFACE)
    ax.axvspan(0.5, HONEST["acc_p95"], color="#c3c2b7", alpha=0.35, lw=0, zorder=0)
    ax.text(0.703, 0.93, "achievable without leakage\n(95th pct, best of 7 pipelines)", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK2, va="top")
    ax.axvline(HONEST["acc_max"], color=INK, lw=1.2, ls="--")
    ax.text(HONEST["acc_max"] + 0.003, 0.93, f"best of 7,000 honest splits {HONEST['acc_max']:.1%}", transform=ax.get_xaxis_transform(), fontsize=8, color=INK)
    ax.axvline(HONEST["acc_mean"], color=INK2, lw=1.2, ls=":")
    ax.text(HONEST["acc_mean"] + 0.004, 0.03, f"leakage-free mean {HONEST['acc_mean']:.0%}", transform=ax.get_xaxis_transform(), fontsize=8, color=INK2)
    flag = d.time_used | d.resampling
    rng = np.random.RandomState(0)
    y = rng.uniform(-0.35, 0.35, len(d))
    ax.scatter(d.acc[~flag], y[~flag], s=46, color=C_PLAIN, edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.scatter(d.acc[flag], y[flag], s=46, color=C_FLAG, edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.set_xlim(0.7, 1.0); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(f"Headline accuracy reported in the abstract (n = {len(d)} studies on the 299-patient dataset)", fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    n_above = int((d.acc > HONEST["acc_p95"]).sum())
    ax.set_title(f"{n_above} of {len(d)} published accuracies exceed the leakage-free range", loc="left",
                 fontsize=12, color=INK, pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_FLAG, label="abstract states time-as-input or resampling"),
                       Line2D([], [], marker="o", ls="", color=C_PLAIN, label="no such statement in abstract")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False, labelcolor=INK2)
    plt.tight_layout()
    p = os.path.join(OUT, "fig3_literature_vs_achievable.png")
    plt.savefig(p, dpi=300, facecolor=SURFACE)
    print("saved", p)


if __name__ == "__main__":
    main()
