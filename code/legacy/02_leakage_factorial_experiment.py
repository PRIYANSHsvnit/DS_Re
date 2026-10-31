"""
Factorial leakage experiment on the UCI heart-failure dataset (n = 299).

Question: which pipeline choices (alone or combined) are needed to reach the
97.78% accuracy reported by Ali et al. (Healthcare Analytics 2023)?

Pipeline factors (2 x 2 x 2 = 8 pipelines):
  O  outlier removal  : sequential 1.5*IQR filtering on the FULL data before splitting (paper) vs none
  S  SMOTE            : on the FULL data before splitting (paper) vs on the training fold only
  T  `time` feature   : included as a predictor vs excluded (it is follow-up length, i.e. part of the outcome)

Evaluation / reporting factors (computed for every pipeline):
  test set : all test rows vs only real (non-SMOTE) test rows
  report   : mean over R random 70/30 splits vs best single split (cherry-picking)

Usage:  python 02_leakage_factorial_experiment.py [--reps 200]
Outputs: results/leakage_factorial_per_split.csv, results/leakage_factorial_summary.csv,
         results/leakage_main_effects.csv
"""
import argparse, itertools, os, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(HERE, "data", "heart_failure_clinical_records_dataset.csv")
OUT = os.path.join(HERE, "results", "legacy")
TARGET = "DEATH_EVENT"
IQR_COLS = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]
PAPER_ACC = 0.9778

# Hyperparameters from Table 3 of the paper
MODELS = {
    "RF": lambda s: RandomForestClassifier(n_estimators=65, max_depth=8, random_state=s, n_jobs=1),
    "DT": lambda s: DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=s),
    "XGB": lambda s: XGBClassifier(learning_rate=0.59, random_state=s, n_jobs=1, eval_metric="logloss"),
}


def iqr_filter(df):
    for c in IQR_COLS:
        q1, q3 = df[c].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        df = df[(df[c] >= lo) & (df[c] <= hi)]
    return df


def smote(X, y, seed):
    """Returns resampled X, y and a boolean mask marking real (non-synthetic) rows."""
    Xr, yr = SMOTE(random_state=seed).fit_resample(X, y)
    real = np.zeros(len(Xr), dtype=bool)
    real[: len(X)] = True  # imblearn appends synthetic rows after the originals
    return Xr, yr, real


def one_split(df_raw, O, S, T, seed):
    df = iqr_filter(df_raw) if O else df_raw
    feats = [c for c in df.columns if c != TARGET and (T or c != "time")]
    X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(int)

    if S == "global":
        X, y, real = smote(X, y, seed)
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
        Xtr, ytr, Xte, yte, real_te = X[tr], y[tr], X[te], y[te], real[te]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        Xtr, ytr, _ = smote(Xtr, ytr, seed)
        real_te = np.ones(len(yte), dtype=bool)

    rows = []
    for mname, make in MODELS.items():
        m = make(seed).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        for subset, mask in (("all_test_rows", np.ones_like(real_te)), ("real_test_rows", real_te)):
            yt, pt = yte[mask], p[mask]
            yh = (pt >= 0.5).astype(int)
            rows.append(dict(model=mname, outlier_removal=O, smote=S, time_feature=T, seed=seed, test_subset=subset,
                             n_total=len(df), n_test=int(mask.sum()), accuracy=accuracy_score(yt, yh),
                             auroc=roc_auc_score(yt, pt), bal_acc=balanced_accuracy_score(yt, yh), f1=f1_score(yt, yh)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df_raw = pd.read_csv(DATA)

    grid = list(itertools.product([False, True], ["train_only", "global"], [False, True], range(args.reps)))
    res = Parallel(n_jobs=-1, verbose=0)(delayed(one_split)(df_raw, O, S, T, s) for O, S, T, s in grid)
    per = pd.DataFrame([r for rows in res for r in rows])
    per.to_csv(os.path.join(OUT, "leakage_factorial_per_split.csv"), index=False)

    keys = ["model", "outlier_removal", "smote", "time_feature", "test_subset"]
    summ = per.groupby(keys).agg(
        n_total=("n_total", "first"), n_test=("n_test", "median"),
        acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        acc_p95=("accuracy", lambda a: a.quantile(0.95)), acc_best_split=("accuracy", "max"),
        p_split_reaches_paper=("accuracy", lambda a: (a >= PAPER_ACC - 1e-9).mean()),
        auroc_mean=("auroc", "mean"), auroc_best_split=("auroc", "max"),
        bal_acc_mean=("bal_acc", "mean"), f1_mean=("f1", "mean"),
    ).reset_index().round(4)
    summ.to_csv(os.path.join(OUT, "leakage_factorial_summary.csv"), index=False)

    # Main effects (accuracy points gained by switching each factor to the leaky level), all test rows
    a = per[per.test_subset == "all_test_rows"]
    eff = []
    for mname in MODELS:
        d = a[a.model == mname]
        for fac, leaky, clean in (("outlier_removal", True, False), ("smote", "global", "train_only"), ("time_feature", True, False)):
            eff.append(dict(model=mname, factor=fac,
                            delta_accuracy=d[d[fac] == leaky].accuracy.mean() - d[d[fac] == clean].accuracy.mean(),
                            delta_auroc=d[d[fac] == leaky].auroc.mean() - d[d[fac] == clean].auroc.mean()))
    eff = pd.DataFrame(eff).round(4)
    eff.to_csv(os.path.join(OUT, "leakage_main_effects.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = summ[(summ.model == "RF")].drop(columns=["model"])
    print("=== Random Forest (paper hyperparameters), %d random 70/30 splits per pipeline ===" % args.reps)
    print(show.to_string(index=False))
    print("\n=== Main effects (all test rows) ===")
    print(eff.to_string(index=False))


if __name__ == "__main__":
    main()
