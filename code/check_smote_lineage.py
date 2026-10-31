"""Trace test-patient endpoints in the paired SMOTE simulation."""
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

ROOT=Path(__file__).resolve().parent
class TracedSMOTE(SMOTE):
    def _make_samples(self,X,y_dtype,y_type,nn_data,nn_num,n_samples,step_size=1.0,y=None):
        rng=check_random_state(self.random_state); ix=rng.randint(0,nn_num.size,size=n_samples)
        row=ix//nn_num.shape[1]; col=ix%nn_num.shape[1]
        self.endpoints=np.column_stack([row,nn_num[row,col]])
        return super()._make_samples(X,y_dtype,y_type,nn_data,nn_num,n_samples,step_size,y)
def main():
    d=pd.read_csv(ROOT.parent/'data/heart_failure_clinical_records_dataset.csv');y=d.DEATH_EVENT.to_numpy(int)
    rows=[]
    for tf in [False,True]:
        cols=[c for c in d if c!='DEATH_EVENT' and (tf or c!='time')];X=d[cols].to_numpy(float)
        for seed in range(300):
            tr,te=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
            a=SMOTE(random_state=seed);_,yt=a.fit_resample(X[tr],y[tr]);needed=len(yt)-len(tr)
            b=TracedSMOTE(random_state=seed);xb,yb=b.fit_resample(X,y)
            orig=SMOTE(random_state=seed);xo,yo=orig.fit_resample(X,y)
            assert np.array_equal(xb,xo) and np.array_equal(yb,yo)
            chosen=np.random.RandomState(seed).choice(len(b.endpoints),needed,replace=False)
            minority=np.flatnonzero(y==1); endpoints=minority[b.endpoints[chosen]]
            uses_test=np.isin(endpoints,te).any(axis=1)
            binary=[cols.index(c) for c in ['anaemia','diabetes','high_blood_pressure','sex','smoking']]
            synthetic=xb[len(y):][chosen][:,binary]; fractional=((synthetic>1e-10)&(synthetic<1-1e-10)).any(axis=1)
            rows.append({'time_feature':tf,'seed':seed,'synthetic_train_n':needed,'synthetic_with_test_endpoint':uses_test.sum(),'fraction_with_test_endpoint':uses_test.mean(),'fraction_with_fractional_binary':fractional.mean(),'test_minority_patients':sum(y[te]==1),'distinct_test_endpoints':np.unique(endpoints[np.isin(endpoints,te)]).size})
    a=pd.DataFrame(rows);out=ROOT.parent/'results/paired_smote';a.to_csv(out/'lineage_per_split.csv',index=False)
    summary=a.groupby('time_feature').agg(mean_fraction=('fraction_with_test_endpoint','mean'),min_fraction=('fraction_with_test_endpoint','min'),max_fraction=('fraction_with_test_endpoint','max'),mean_fraction_with_fractional_binary=('fraction_with_fractional_binary','mean'),mean_distinct_test_endpoints=('distinct_test_endpoints','mean'),mean_test_minority=('test_minority_patients','mean'))
    summary.to_csv(out/'lineage_summary.csv');print(summary.to_string())
if __name__=='__main__':main()
