"""
Extended leakage experiment: can ANY realistic combination of leaks reach 97.78%?

Raw data (299 patients, no outlier removal). Four binary leakage factors, 2^4 = 16 pipelines, RF with the
paper's hyperparameters unless tuned, N random stratified 70/30 splits each:
  T  follow-up `time` used as a predictor
  S  SMOTE on the full data before splitting (synthetic rows also enter the test set)
  F  feature selection (top-5 by mutual information) computed on ALL rows, before splitting
  H  hyper-parameter search where the configuration is chosen by TEST accuracy (30 random RF configurations);
     the clean alternative uses the paper's fixed hyper-parameters
Also reports a fifth, 'selective reporting' statistic: the best of K=10 random splits (a researcher re-running
until the number looks good), for the all-leaks pipeline.
A 2^4 factorial ANOVA on accuracy quantifies main effects and interactions.

Usage:  python 10_extended_leaks.py [--reps 300]
Outputs: results/extended_leaks_per_split.csv, results/extended_leaks_summary.csv, results/extended_leaks_anova.csv
"""
import argparse, itertools, os, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(HERE, "data", "heart_failure_clinical_records_dataset.csv")
OUT = os.path.join(HERE, "results", "legacy")
PAPER = 0.9778


def one(df, T, S, F, H, seed):
    warnings.filterwarnings("ignore")
    rng = np.random.RandomState(seed)
    cols = [c for c in df.columns if c != "DEATH_EVENT" and (T or c != "time")]
    X, y = df[cols].to_numpy(float), df.DEATH_EVENT.to_numpy()
    if F:  # selection on all rows (leak): keep the 5 most informative columns
        keep = np.argsort(mutual_info_classif(X, y, random_state=seed))[-5:]
        X = X[:, keep]
    if S:
        Xr, yr = SMOTE(random_state=seed).fit_resample(X, y)
        tr, te = train_test_split(np.arange(len(yr)), test_size=0.3, random_state=seed, stratify=yr)
        Xtr, ytr, Xte, yte = Xr[tr], yr[tr], Xr[te], yr[te]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if H:  # configuration picked by test accuracy
        best = (-1, None)
        for k in range(30):
            cfg = dict(n_estimators=int(rng.choice([30, 65, 100, 200, 400])), max_depth=int(rng.choice([3, 5, 8, 12, 20])),
                       min_samples_leaf=int(rng.choice([1, 2, 4])), max_features=rng.choice(["sqrt", "log2", None]))
            m = RandomForestClassifier(random_state=seed, n_jobs=1, **cfg).fit(Xtr, ytr)
            a = accuracy_score(yte, m.predict(Xte))
            if a > best[0]:
                best = (a, m)
        m = best[1]
    else:
        m = RandomForestClassifier(n_estimators=65, max_depth=8, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return dict(T=T, S=S, F=F, H=H, seed=seed, accuracy=accuracy_score(yte, (p >= .5).astype(int)), auroc=roc_auc_score(yte, p))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=300); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    grid = list(itertools.product([False, True], repeat=4))
    jobs = [(T, S, F, H, s) for (T, S, F, H) in grid for s in range(a.reps)]
    per = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(df, *j) for j in jobs))
    per.to_csv(os.path.join(OUT, "extended_leaks_per_split.csv"), index=False)

    g = per.groupby(["T", "S", "F", "H"]).accuracy
    summ = pd.DataFrame({"acc_mean": g.mean(), "acc_sd": g.std(), "acc_p95": g.quantile(.95), "acc_best": g.max(),
                         "share_reaching_97.78": g.apply(lambda x: (x >= PAPER - 1e-9).mean()),
                         "auroc_mean": per.groupby(["T", "S", "F", "H"]).auroc.mean()}).round(4).reset_index()
    # selective reporting: best of 10 random splits for each pipeline
    best10 = []
    for key, d in per.groupby(["T", "S", "F", "H"]):
        acc = d.accuracy.to_numpy(); r = np.random.RandomState(0)
        best10.append(np.mean([acc[r.choice(len(acc), 10, replace=False)].max() for _ in range(2000)]))
    summ["best_of_10_splits_mean"] = np.round(best10, 4)
    summ.to_csv(os.path.join(OUT, "extended_leaks_summary.csv"), index=False)

    d = per.assign(**{k: per[k].astype(int) for k in "TSFH"})
    model = ols("accuracy ~ C(T)*C(S)*C(F)*C(H)", data=d).fit()
    an = sm.stats.anova_lm(model, typ=2)
    an["pct_variance"] = (an.sum_sq / an.sum_sq.sum() * 100).round(2)
    an.round(5).to_csv(os.path.join(OUT, "extended_leaks_anova.csv"))
    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print("\nANOVA (share of accuracy variance):"); print(an[["F", "PR(>F)", "pct_variance"]].round(4).to_string())
    print("\nMain effects (accuracy points):")
    for k in "TSFH":
        print(f"  {k}: {100 * (d[d[k] == 1].accuracy.mean() - d[d[k] == 0].accuracy.mean()):+.1f}")


if __name__ == "__main__":
    main()
