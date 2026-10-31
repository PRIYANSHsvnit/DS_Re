"""
Leakage-free reference distribution for comparing published accuracies (C6).

To keep the comparison conservative, several honest pipelines are run (no resampling, class weights,
SMOTE inside training only; RF, logistic regression, XGBoost, SVM) over 1,000 random stratified
70/30 splits each, without `time`. The reference used in the paper is the MOST FAVOURABLE pipeline's
95th percentile, i.e. the bar a published result must clear to be plausible without leakage.

Usage:  python 08_honest_reference.py
Output: results/honest_reference.csv
"""
import os, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(HERE, "data", "heart_failure_clinical_records_dataset.csv")
OUT = os.path.join(HERE, "results", "legacy")

PIPES = {
    "RF (paper params), no resampling": lambda s: RandomForestClassifier(n_estimators=65, max_depth=8, random_state=s, n_jobs=1),
    "RF 500 trees, no resampling": lambda s: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=s, n_jobs=1),
    "RF 500 trees, class weights": lambda s: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=s, n_jobs=1),
    "Logistic regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "XGBoost (default)": lambda s: XGBClassifier(random_state=s, n_jobs=1, eval_metric="logloss"),
    "SVM (RBF)": lambda s: make_pipeline(StandardScaler(), SVC(probability=True, random_state=s)),
    "RF (paper params), SMOTE in training only": "smote",
}


def run(X, y, seed):
    warnings.filterwarnings("ignore")  # joblib workers do not inherit the filter
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    out = []
    for name, make in PIPES.items():
        if make == "smote":
            Xs, ys = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
            m = RandomForestClassifier(n_estimators=65, max_depth=8, random_state=seed, n_jobs=1).fit(Xs, ys)
        else:
            m = make(seed).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        out.append(dict(pipeline=name, seed=seed, accuracy=accuracy_score(yte, (p >= .5).astype(int)), auroc=roc_auc_score(yte, p)))
    return out


def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["DEATH_EVENT", "time"]).to_numpy(float); y = df.DEATH_EVENT.to_numpy()
    res = pd.DataFrame([r for rows in Parallel(n_jobs=-1)(delayed(run)(X, y, s) for s in range(1000)) for r in rows])
    summ = res.groupby("pipeline").agg(acc_mean=("accuracy", "mean"), acc_p95=("accuracy", lambda a: a.quantile(.95)), acc_max=("accuracy", "max"),
                                       auc_mean=("auroc", "mean"), auc_p95=("auroc", lambda a: a.quantile(.95)), auc_max=("auroc", "max")).round(4)
    summ = summ.sort_values("acc_p95", ascending=False)
    summ.loc["MOST FAVOURABLE (reference)"] = [summ.acc_mean.max(), summ.acc_p95.max(), summ.acc_max.max(),
                                               summ.auc_mean.max(), summ.auc_p95.max(), summ.auc_max.max()]
    summ.to_csv(os.path.join(OUT, "honest_reference.csv"))
    print("Majority-class accuracy:", round(1 - y.mean(), 4))
    print(summ.to_string())


if __name__ == "__main__":
    main()
