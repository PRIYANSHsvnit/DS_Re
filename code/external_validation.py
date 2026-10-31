"""Validate UCI models on a carefully harmonized external mortality cohort.

Input must already use the UCI column names and units and have actual event/censor
times in days from a compatible baseline. No guessed mappings or event dates.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

ROOT=Path(__file__).resolve().parent
FEATURES=['age','sex','serum_creatinine','serum_sodium','diabetes','ejection_fraction']
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('external',type=Path)
    ap.add_argument('--uci',type=Path,default=ROOT.parent/'data/heart_failure_clinical_records_dataset.csv')
    ap.add_argument('--output',type=Path,default=ROOT.parent/'results/external')
    ap.add_argument('--outcome-confirmed',action='store_true',help='Confirm compatible all-cause mortality, baseline, units and true event/censor times were verified against source documentation.')
    a=ap.parse_args()
    if not a.outcome_confirmed: raise SystemExit('Verify mortality outcome, time origin, measurement timing and units before adding --outcome-confirmed.')
    train=pd.read_csv(a.uci); test=pd.read_csv(a.external)
    for label,d in [('UCI',train),('external',test)]:
        missing=set(FEATURES+['time','DEATH_EVENT'])-set(d.columns)
        if missing: raise SystemExit(f'{label} missing required columns: {sorted(missing)}')
        for col in FEATURES+['time','DEATH_EVENT']: d[col]=pd.to_numeric(d[col],errors='raise')
        if d[['time','DEATH_EVENT']].isna().any().any() or (d.time<=0).any() or not d.DEATH_EVENT.isin([0,1]).all(): raise SystemExit(f'{label}: invalid time or event values')
        for col in ['sex','diabetes']:
            if not d[col].dropna().isin([0,1]).all(): raise SystemExit(f'{label}: {col} must be coded 0/1')
    if test.time.min()>=180 or test.time.max()<=180: raise SystemExit('External follow-up must span 180 days to estimate 180-day metrics.')
    ytr=Surv.from_arrays(train.DEATH_EVENT.astype(bool),train.time)
    yte=Surv.from_arrays(test.DEATH_EVENT.astype(bool),test.time)
    imp=SimpleImputer(strategy='median').fit(train[FEATURES]); scale=StandardScaler().fit(imp.transform(train[FEATURES]))
    xtr=scale.transform(imp.transform(train[FEATURES])); xte=scale.transform(imp.transform(test[FEATURES]))
    models={'CoxPH':CoxPHSurvivalAnalysis(alpha=.1), 'RSF':RandomSurvivalForest(n_estimators=500,min_samples_leaf=10,max_features='sqrt',random_state=0,n_jobs=1), 'GBSurv':GradientBoostingSurvivalAnalysis(n_estimators=200,learning_rate=.05,max_depth=2,subsample=.8,random_state=0)}
    # External censoring distribution: inverse censoring weights for known 180-day outcomes.
    ct,cg=kaplan_meier_estimator(~test.DEATH_EVENT.astype(bool).to_numpy(),test.time.to_numpy())
    def G(t, left=False):
        idx=np.searchsorted(ct,t,side='left' if left else 'right')-1
        return np.where(idx>=0,cg[np.maximum(idx,0)],1.)
    time=test.time.to_numpy(); event=test.DEATH_EVENT.to_numpy().astype(bool)
    cases=event & (time<=180); controls=time>180; known=cases|controls
    weights=np.zeros(len(test)); weights[cases]=1/G(time[cases],left=True); weights[controls]=1/G(np.full(controls.sum(),180))
    if not cases.any() or not controls.any() or not np.isfinite(weights[known]).all(): raise SystemExit('Insufficient cases, controls, or censoring support at 180 days.')
    rows=[]
    for name,model in models.items():
        model.fit(xtr,ytr); risk=model.predict(xte)
        p=np.clip(1-np.array([s(180) for s in model.predict_survival_function(xte)]),1e-6,1-1e-6)
        row={'model':name,'n':len(test),'harrell_c':concordance_index_censored(event,time,risk)[0], 'mean_predicted_180d':p.mean()}
        try: row['auc_180d']=cumulative_dynamic_auc(yte,yte,p,np.array([180.]))[0][0]
        except ValueError as exc: row['auc_180d_error']=str(exc)
        try:
            cal=LogisticRegression(C=1e6,solver='lbfgs').fit(np.log(p[known]/(1-p[known])).reshape(-1,1),cases[known].astype(int),sample_weight=weights[known])
            row['calibration_slope_ipcw']=cal.coef_[0,0]; row['calibration_intercept_ipcw']=cal.intercept_[0]
        except ValueError as exc: row['calibration_error']=str(exc)
        rows.append(row)
    a.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output/'metrics.csv',index=False)
    (a.output/'cohort_summary.txt').write_text(f'UCI n={len(train)}; external n={len(test)}; external events by 180d={cases.sum()}; controls beyond 180d={controls.sum()}; censored by 180d={(~known).sum()}\nCalibration uses external-cohort IPCW, logistic slope; AUC uses external-cohort censoring distribution. Estimates assume independent censoring.\n',encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
