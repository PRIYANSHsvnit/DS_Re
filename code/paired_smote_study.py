"""Compare training-only vs deliberately contaminated SMOTE on identical original test patients.

The contaminated arm draws synthetic points generated using the whole cohort,
including the original test patients. Both arms contain the same original training
patients and have the same class counts. This is a controlled leakage simulation,
not a direct reproduction of a study that also puts synthetic records in its test set.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
def one(df,seed,time_feature):
    cols=[c for c in df if c!='DEATH_EVENT' and (time_feature or c!='time')]
    X=df[cols].to_numpy(float); y=df.DEATH_EVENT.to_numpy(int)
    train,test=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
    xt,yt=SMOTE(random_state=seed).fit_resample(X[train],y[train])
    xall,yall=SMOTE(random_state=seed).fit_resample(X,y)
    synthetic=xall[len(y):]; needed=len(yt)-len(train)
    chosen=np.random.RandomState(seed).choice(len(synthetic),needed,replace=False)
    xl=np.vstack([X[train],synthetic[chosen]]); yl=np.r_[y[train],yall[len(y):][chosen]]
    assert len(yt)==len(yl) and np.array_equal(np.bincount(yt),np.bincount(yl))
    row={'seed':seed,'time_feature':time_feature,'test_n':len(test),'training_n':len(yt)}
    for label,xfit,yfit in [('train_only',xt,yt),('global_contaminated',xl,yl)]:
        model=RandomForestClassifier(n_estimators=65,max_depth=8,random_state=seed,n_jobs=1).fit(xfit,yfit)
        p=model.predict_proba(X[test])[:,1]; pred=(p>=.5).astype(int)
        for metric,fn in [('accuracy',accuracy_score),('balanced_accuracy',balanced_accuracy_score),('auc',roc_auc_score)]:
            row[label+'_'+metric]=float(fn(y[test],p if metric=='auc' else pred))
    return row
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=300); ap.add_argument('--jobs',type=int,default=2); ap.add_argument('--output',type=Path,default=ROOT.parent/'results/paired_smote'); args=ap.parse_args()
    df=pd.read_csv(ROOT.parent/'data/heart_failure_clinical_records_dataset.csv')
    rows=Parallel(n_jobs=args.jobs)(delayed(one)(df,s,t) for t in [False,True] for s in range(args.reps))
    data=pd.DataFrame(rows); out=args.output;out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'per_split.csv',index=False)
    summary=[]
    for t,group in data.groupby('time_feature'):
        for metric in ['accuracy','balanced_accuracy','auc']:
            a=group['train_only_'+metric]; b=group['global_contaminated_'+metric]; delta=b-a
            summary.append({'time_feature':t,'metric':metric,'splits':len(group),'train_only_mean':a.mean(),'contaminated_mean':b.mean(),'paired_difference_mean':delta.mean(),'paired_difference_p025':delta.quantile(.025),'paired_difference_p975':delta.quantile(.975)})
    result=pd.DataFrame(summary); result.to_csv(out/'summary.csv',index=False);print(result.to_string(index=False))
if __name__=='__main__': main()
