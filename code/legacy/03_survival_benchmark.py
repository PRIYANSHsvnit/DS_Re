"""
Survival-correct re-analysis of the UCI heart-failure dataset (n = 299).

The paper (and most follow-up work) treats DEATH_EVENT as a plain binary label and
either uses `time` as a predictor (leakage) or ignores it (censoring bias: patients
lost to follow-up at day 12 are labelled "survived"). Here `time` is used for what it
is - the survival time - and models are compared with censoring-aware metrics.

Models : Cox PH (penalised), Random Survival Forest, Gradient-Boosted Survival,
         plus a binary RF classifier (risk = P(death)) to show what ignoring time costs.
Metrics: Harrell's C, Uno's C (tau = 240 d), time-dependent AUC at 30/90/180 d,
         integrated Brier score (30-180 d). Repeated stratified 5-fold CV (10 repeats).
Also   : full-cohort Cox hazard ratios with 95% CI + Schoenfeld PH test, compared to
         the paper's Table 8.

Usage:  python 03_survival_benchmark.py
Outputs: results/survival_cv_per_fold.csv, results/survival_cv_summary.csv,
         results/cox_hazard_ratios_299.csv
"""
import os, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

warnings.filterwarnings("ignore")
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(HERE, "data", "heart_failure_clinical_records_dataset.csv")
OUT = os.path.join(HERE, "results", "legacy")
FEATS = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure",
         "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking"]
LOG = ["creatinine_phosphokinase", "serum_creatinine"]  # right-skewed labs
TIMES = np.array([30, 90, 180])
IBS_GRID = np.arange(30, 181, 5)
TAU = 240


def prep(tr, te):
    tr, te = tr.copy(), te.copy()
    for c in LOG:
        tr[c], te[c] = np.log(tr[c]), np.log(te[c])
    sc = StandardScaler().fit(tr[FEATS])
    return sc.transform(tr[FEATS]), sc.transform(te[FEATS])


def models(seed):
    return {
        "CoxPH": CoxPHSurvivalAnalysis(alpha=0.1),
        "RSF": RandomSurvivalForest(n_estimators=500, min_samples_leaf=10, max_features="sqrt", random_state=seed, n_jobs=1),
        "GBSurv": GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed),
        "RF_binary_ignores_time": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=1),
    }


def fold(df, tr_idx, te_idx, rep, k):
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    Xtr, Xte = prep(tr, te)
    ytr = Surv.from_arrays(tr.DEATH_EVENT.astype(bool), tr.time)
    yte = Surv.from_arrays(te.DEATH_EVENT.astype(bool), te.time)
    # evaluation times must lie inside the test follow-up range
    t_ok = TIMES[(TIMES > te.time.min()) & (TIMES < te.time.max())]
    g_ok = IBS_GRID[(IBS_GRID > te.time.min()) & (IBS_GRID < te.time.max())]
    rows = []
    for name, m in models(rep * 10 + k).items():
        if name == "RF_binary_ignores_time":
            m.fit(Xtr, tr.DEATH_EVENT)
            risk = m.predict_proba(Xte)[:, 1]
            surv_fn = None
        else:
            m.fit(Xtr, ytr)
            risk = m.predict(Xte)
            surv_fn = m.predict_survival_function(Xte)
        r = dict(model=name, repeat=rep, fold=k,
                 harrell_c=concordance_index_censored(yte["event"], yte["time"], risk)[0],
                 uno_c=concordance_index_ipcw(ytr, yte, risk, tau=TAU)[0])
        aucs, _ = cumulative_dynamic_auc(ytr, yte, risk, t_ok)
        for t, a in zip(t_ok, aucs):
            r[f"auc_{t}d"] = a
        if surv_fn is not None:
            S = np.vstack([f(g_ok) for f in surv_fn])
            r["ibs_30_180"] = integrated_brier_score(ytr, yte, S, g_ok)
        rows.append(r)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    jobs = [(tr, te, i // 5, i % 5) for i, (tr, te) in enumerate(cv.split(df, df.DEATH_EVENT))]
    res = Parallel(n_jobs=-1)(delayed(fold)(df, tr, te, r, k) for tr, te, r, k in jobs)
    per = pd.DataFrame([x for rows in res for x in rows])
    per.to_csv(os.path.join(OUT, "survival_cv_per_fold.csv"), index=False)

    metrics = [c for c in per.columns if c not in ("model", "repeat", "fold")]
    summ = per.groupby("model")[metrics].agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.sort_values("harrell_c_mean", ascending=False)
    summ.to_csv(os.path.join(OUT, "survival_cv_summary.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("=== Repeated 5-fold CV (10 repeats), censoring-aware metrics ===")
    print(summ.to_string())

    # Full-cohort Cox model for risk-factor interpretation (per-SD HRs; log-transformed labs)
    d = df.copy()
    for c in LOG:
        d[c] = np.log(d[c])
    cont = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]
    d[cont] = (d[cont] - d[cont].mean()) / d[cont].std()
    cph = CoxPHFitter().fit(d[FEATS + ["time", "DEATH_EVENT"]], "time", "DEATH_EVENT")
    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    hr.columns = ["HR_per_SD_or_yes", "CI_low", "CI_high", "p"]
    ph = proportional_hazard_test(cph, d[FEATS + ["time", "DEATH_EVENT"]], time_transform="rank").summary["p"]
    hr = hr.assign(PH_test_p=ph).round(4).sort_values("p")
    hr.to_csv(os.path.join(OUT, "cox_hazard_ratios_299.csv"))
    print(f"\n=== Full-cohort Cox PH (concordance {cph.concordance_index_:.3f}); continuous vars per SD, CPK & creatinine log-scaled ===")
    print(hr.to_string())


if __name__ == "__main__":
    main()
