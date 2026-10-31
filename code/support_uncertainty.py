"""External-only patient bootstrap for three fixed UCI-trained survival models."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT=Path(__file__).resolve().parent; cols=['age','sex','log_creatinine','serum_sodium','diabetes']
def main():
    u=pd.read_csv(ROOT.parent/'data/heart_failure_clinical_records_dataset.csv'); u['log_creatinine']=np.log(u.serum_creatinine)
    s=pd.read_csv(ROOT.parent/'data/support2.csv',index_col=0); s=s[(s.dzgroup=='CHF') & s.crea.notna()].copy()
    s=pd.DataFrame({'age':s.age,'sex':(s.sex=='male').astype(int),'log_creatinine':np.log(s.crea),'serum_sodium':s.sod,'diabetes':s.diabetes,'time':np.minimum(s['d.time'],280),'event':((s.death==1)&(s['d.time']<=280)).astype(int)}).reset_index(drop=True)
    assert not ((s.event==0)&(s.time<180)).any()
    scaler=StandardScaler().fit(u[cols]); xtr=scaler.transform(u[cols]); xte=scaler.transform(s[cols]); y=Surv.from_arrays(u.DEATH_EVENT.astype(bool),u.time)
    event=s.event.to_numpy(bool); time=s.time.to_numpy(); outcome=(event & (time<=180)).astype(int)
    models={'CoxPH':CoxPHSurvivalAnalysis(alpha=.1),'RSF':RandomSurvivalForest(n_estimators=500,min_samples_leaf=10,max_features='sqrt',random_state=0,n_jobs=1),'GBSurv':GradientBoostingSurvivalAnalysis(n_estimators=200,learning_rate=.05,max_depth=2,subsample=.8,random_state=0)}
    out=ROOT.parent/'results/support_bootstrap';out.mkdir(parents=True,exist_ok=True); rng=np.random.RandomState(2026); indices=rng.randint(0,len(s),(1000,len(s))); rows=[]; predictions=pd.DataFrame({'time':time,'event':event,'event_180':outcome})
    for name,model in models.items():
        model.fit(xtr,y); risk=model.predict(xte); p=np.array([1-f(180) for f in model.predict_survival_function(xte)]); predictions[name+'_p180']=p
        def metrics(ix):
            e=event[ix]; t=time[ix]; actual=outcome[ix]; pred=p[ix]; r=risk[ix]
            return {'harrell_c':concordance_index_censored(e,t,r)[0], 'auc_180':roc_auc_score(actual,pred),'brier_180':brier_score_loss(actual,pred),'mean_predicted':pred.mean(),'observed_180':actual.mean(),'mean_prediction_minus_observed':pred.mean()-actual.mean()}
        point=metrics(np.arange(len(s))); boot=pd.DataFrame([metrics(ix) for ix in indices])
        for k,v in point.items(): rows.append({'model':name,'metric':k,'estimate':v,'ci_low':boot[k].quantile(.025),'ci_high':boot[k].quantile(.975),'bootstrap_samples':len(indices)})
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);predictions.to_csv(out/'predictions.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
