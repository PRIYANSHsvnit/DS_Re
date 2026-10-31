"""
External validation (contribution C5) and risk-factor replication (C3).

Development cohort : UCI Heart Failure Clinical Records, n = 299 (Faisalabad, 2015).
Validation cohort  : SUPPORT2 congestive-heart-failure subgroup, n = 1,387 (5 US centres, 1989-94),
                     publicly available from Vanderbilt Biostatistics (hbiostat.org/data).
Shared predictors  : age, sex, serum creatinine (log), serum sodium, diabetes.
Outcome            : all-cause death; SUPPORT follow-up administratively censored at 280 days to match
                     the UCI follow-up window (UCI max = 285 d). Nobody in SUPPORT-CHF is censored before.

Parts
  A. Transport: models fitted on all 299 UCI patients, evaluated on SUPPORT-CHF
     (Harrell C, Uno C, AUC(t), 180-day accuracy/AUROC, calibration) vs internal 10x5 CV.
  B. The paper-style pipeline (IQR filter + SMOTE before split): apparent internal accuracy vs external accuracy.
  C. Cox hazard ratios: UCI (11 predictors, 1,000-bootstrap 95% CI) and UCI vs SUPPORT on shared predictors.

Usage:  python 06_external_validation.py
Needs:  data/support2.csv  (download: https://hbiostat.org/data/repo/support2csv.zip)
Outputs: results/external_validation_summary.csv, results/external_paper_pipeline.csv,
         results/cox_bootstrap_hr_uci.csv, results/cox_hr_uci_vs_support.csv
"""
import os, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings("ignore")
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UCI = os.path.join(HERE, "data", "heart_failure_clinical_records_dataset.csv")
SUP = os.path.join(HERE, "data", "support2.csv")
OUT = os.path.join(HERE, "results", "legacy")
SHARED = ["age", "sex", "log_creatinine", "serum_sodium", "diabetes"]
ALL11 = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure",
         "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking"]
CENSOR_AT, H, TAU, TIMES = 280, 180, 240, np.array([30, 90, 180])


def load():
    u = pd.read_csv(UCI)
    u["log_creatinine"] = np.log(u.serum_creatinine)
    s = pd.read_csv(SUP, index_col=0)
    s = s[s.dzgroup == "CHF"].dropna(subset=["crea"]).copy()
    s = pd.DataFrame({"age": s.age, "sex": (s.sex == "male").astype(int), "log_creatinine": np.log(s.crea),
                      "serum_sodium": s.sod, "diabetes": s.diabetes.astype(int),
                      "time": np.minimum(s["d.time"], CENSOR_AT), "DEATH_EVENT": ((s.death == 1) & (s["d.time"] <= CENSOR_AT)).astype(int)})
    return u, s.reset_index(drop=True)


def surv_models(seed):
    return {"CoxPH": CoxPHSurvivalAnalysis(alpha=0.1),
            "RSF": RandomSurvivalForest(n_estimators=500, min_samples_leaf=10, max_features="sqrt", random_state=seed, n_jobs=1),
            "GBSurv": GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed)}


def evaluate(tr, te, seed, scaler_from=None):
    """Fit all models on tr (UCI rows), evaluate on te. Returns list of metric dicts."""
    sc = StandardScaler().fit(tr[SHARED])
    Xtr, Xte = sc.transform(tr[SHARED]), sc.transform(te[SHARED])
    ytr = Surv.from_arrays(tr.DEATH_EVENT.astype(bool), tr.time)
    yte = Surv.from_arrays(te.DEATH_EVENT.astype(bool), te.time)
    t_ok = TIMES[(TIMES > te.time.min()) & (TIMES < min(te.time.max(), tr.time.max()))]
    # 180-day binary label; patients censored before 180 d have unknown status and are excluded
    known = (te.time >= H) | (te.DEATH_EVENT == 1)
    y180 = ((te.DEATH_EVENT == 1) & (te.time <= H)).astype(int)
    rows = []
    risks = {}
    for name, m in surv_models(seed).items():
        m.fit(Xtr, ytr)
        risk = m.predict(Xte)
        surv = m.predict_survival_function(Xte)
        p180 = 1 - np.array([f(H) for f in surv])
        risks[name] = (risk, p180)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=1)
    rf.fit(Xtr, tr.DEATH_EVENT)
    p = rf.predict_proba(Xte)[:, 1]
    risks["RF_classifier"] = (p, None)
    for name, (risk, p180) in risks.items():
        r = dict(model=name, harrell_c=concordance_index_censored(yte["event"], yte["time"], risk)[0],
                 uno_c=concordance_index_ipcw(ytr, yte, risk, tau=TAU)[0])
        aucs, _ = cumulative_dynamic_auc(ytr, yte, risk, t_ok)
        r.update({f"auc_{t}d": a for t, a in zip(t_ok, aucs)})
        score = p180 if p180 is not None else risk
        r["auroc_180d"] = roc_auc_score(y180[known], score[known])
        if p180 is not None:
            r["accuracy_180d"] = accuracy_score(y180[known], (p180[known] >= 0.5).astype(int))
            r["observed_180d_mortality"] = y180[known].mean()
            r["mean_predicted_180d_mortality"] = p180[known].mean()
            lp = np.log(np.clip(p180[known], 1e-4, 1 - 1e-4) / (1 - np.clip(p180[known], 1e-4, 1 - 1e-4)))
            r["calibration_slope"] = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y180[known]).coef_[0, 0]
        else:
            r["accuracy_180d"] = accuracy_score(y180[known], (score[known] >= 0.5).astype(int))
        rows.append(r)
    return rows


def paper_pipeline(u, s, seed):
    """IQR outlier removal + SMOTE on full data + 70/30 split + RF(65, depth 8), shared predictors only."""
    d = u.copy()
    for c in ["age", "log_creatinine", "serum_sodium"]:
        q1, q3 = d[c].quantile([.25, .75]); k = 1.5 * (q3 - q1)
        d = d[(d[c] >= q1 - k) & (d[c] <= q3 + k)]
    X, y = SMOTE(random_state=seed).fit_resample(d[SHARED].to_numpy(float), d.DEATH_EVENT.to_numpy())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    m = RandomForestClassifier(n_estimators=65, max_depth=8, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    known = (s.time >= H) | (s.DEATH_EVENT == 1)
    y180 = ((s.DEATH_EVENT == 1) & (s.time <= H)).astype(int)[known]
    pe = m.predict_proba(s.loc[known, SHARED].to_numpy(float))[:, 1]
    return dict(seed=seed, internal_accuracy=accuracy_score(yte, m.predict(Xte)), internal_auroc=roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
                external_accuracy_180d=accuracy_score(y180, (pe >= 0.5).astype(int)), external_auroc_180d=roc_auc_score(y180, pe))


def cox_hr(d, cols, cont):
    d = d[cols + ["time", "DEATH_EVENT"]].copy()
    d[cont] = (d[cont] - d[cont].mean()) / d[cont].std()
    return CoxPHFitter().fit(d, "time", "DEATH_EVENT")


def boot_hr(u, cols, cont, b):
    idx = np.random.RandomState(b).randint(0, len(u), len(u))
    return np.exp(cox_hr(u.iloc[idx], cols, cont).params_)


def main():
    os.makedirs(OUT, exist_ok=True)
    u, s = load()
    print(f"UCI n={len(u)} deaths={u.DEATH_EVENT.sum()} | SUPPORT-CHF n={len(s)} deaths<= {CENSOR_AT}d={s.DEATH_EVENT.sum()} "
          f"| 180-d mortality UCI(known)={((u.DEATH_EVENT == 1) & (u.time <= H)).sum() / ((u.time >= H) | (u.DEATH_EVENT == 1)).sum():.3f} "
          f"SUPPORT={((s.DEATH_EVENT == 1) & (s.time <= H)).mean():.3f}")

    # A. internal 10x5 CV (shared predictors) vs external
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    res = Parallel(n_jobs=-1)(delayed(evaluate)(u.iloc[tr], u.iloc[te], i) for i, (tr, te) in enumerate(cv.split(u, u.DEATH_EVENT)))
    internal = pd.DataFrame([r for rows in res for r in rows]).groupby("model").mean(numeric_only=True).add_prefix("internal_")
    external = pd.DataFrame(evaluate(u, s, 0)).set_index("model").add_prefix("external_")
    # Reference ceiling: same models trained AND tested inside SUPPORT-CHF (10x5 CV). Separates
    # "these 5 predictors are weak in SUPPORT" from "models fitted on UCI do not transport".
    cvs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7)
    res_s = Parallel(n_jobs=-1)(delayed(evaluate)(s.iloc[tr], s.iloc[te], i) for i, (tr, te) in enumerate(cvs.split(s, s.DEATH_EVENT)))
    within = pd.DataFrame([r for rows in res_s for r in rows]).groupby("model").mean(numeric_only=True).add_prefix("within_support_cv_")
    summ = internal.join(external).join(within).round(4)
    summ.to_csv(os.path.join(OUT, "external_validation_summary.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n=== A. Shared-predictor models: internal 10x5 CV (UCI) vs external (SUPPORT-CHF) ===")
    print(summ[[c for c in summ.columns if any(k in c for k in ("harrell", "uno", "auroc_180", "accuracy_180", "calibration", "mortality"))]].T.to_string())

    # B. paper-style pipeline
    pp = pd.DataFrame(Parallel(n_jobs=-1)(delayed(paper_pipeline)(u, s, k) for k in range(200)))
    pp.to_csv(os.path.join(OUT, "external_paper_pipeline.csv"), index=False)
    print("\n=== B. Paper-style pipeline (IQR + SMOTE before split, RF), 200 seeds: mean (best) ===")
    for c in pp.columns[1:]:
        print(f"  {c:<24} {pp[c].mean():.3f}  (best {pp[c].max():.3f})")

    # C. hazard ratios
    cont11 = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]
    u11 = u.copy(); u11["creatinine_phosphokinase"] = np.log(u11.creatinine_phosphokinase); u11["serum_creatinine"] = np.log(u11.serum_creatinine)
    point = np.exp(cox_hr(u11, ALL11, cont11).params_)
    boots = pd.DataFrame(Parallel(n_jobs=-1)(delayed(boot_hr)(u11, ALL11, cont11, b) for b in range(1000)))
    bt = pd.DataFrame({"HR": point, "boot_CI_low": boots.quantile(0.025), "boot_CI_high": boots.quantile(0.975),
                       "share_boot_HR_gt_1": (boots > 1).mean()}).round(4).sort_values("HR")
    bt.to_csv(os.path.join(OUT, "cox_bootstrap_hr_uci.csv"))
    print("\n=== C1. UCI Cox HR (per SD for continuous; CPK & creatinine log), 1,000-bootstrap 95% CI ===")
    print(bt.to_string())

    contS = ["age", "log_creatinine", "serum_sodium"]
    rows = []
    for name, d in (("UCI (n=299)", u), ("SUPPORT-CHF (n=%d)" % len(s), s)):
        sm = cox_hr(d, SHARED, contS).summary
        for v in SHARED:
            rows.append(dict(cohort=name, variable=v, HR=np.exp(sm.loc[v, "coef"]), CI_low=sm.loc[v, "exp(coef) lower 95%"],
                             CI_high=sm.loc[v, "exp(coef) upper 95%"], p=sm.loc[v, "p"]))
    cmp_ = pd.DataFrame(rows).round(4)
    cmp_.to_csv(os.path.join(OUT, "cox_hr_uci_vs_support.csv"), index=False)
    print("\n=== C2. Shared-predictor Cox HRs: UCI vs SUPPORT-CHF (per SD for age, log creatinine, sodium) ===")
    print(cmp_.to_string(index=False))


if __name__ == "__main__":
    main()
