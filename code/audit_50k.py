"""Recheck the 50k cohort without treating descriptive signals as provenance proof."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data/cardiovascular_hf_dataset-3.csv'
OUT = ROOT.parent / 'results/audit_50k'
def main():
    d = pd.read_csv(DATA)
    out = {'rows': len(d), 'columns': len(d.columns), 'columns_list': d.columns.tolist(),
           'duplicate_full_rows': int(d.duplicated().sum()), 'missing_rates': d.isna().mean().round(5).to_dict()}
    out['source_counts'] = d.Source.value_counts(dropna=False).to_dict()
    out['source_means'] = d.groupby('Source')[['NT_proBNP_pgmL','Ejection_Fraction_pct','HF_Risk_Binary']].mean().round(4).to_dict(orient='index')
    m = d.loc[:, d.isna().any()].isna().astype(int)
    corr = m.corr().to_numpy(); out['missingness_indicator_count'] = m.shape[1]
    out['max_abs_missingness_correlation'] = float(np.nanmax(np.abs(corr[np.triu_indices(m.shape[1],1)])))
    s = d[['Creatinine_mgdL','Age','Gender','eGFR_mLmin']].dropna()
    s=s[s.Gender.isin(['Female','Male']) & (s.Creatinine_mgdL>0)]
    female=s.Gender.eq('Female').to_numpy(); scr=s.Creatinine_mgdL.to_numpy()/np.where(female,.7,.9)
    calc=142*np.minimum(scr,1)**np.where(female,-.241,-.302)*np.maximum(scr,1)**-1.2*.9938**s.Age.to_numpy()*np.where(female,1.012,1)
    out['egfr_ckd_epi_2021']={'complete_n':len(s),'pearson_r':float(np.corrcoef(calc,s.eGFR_mLmin)[0,1]),'median_abs_difference':float(np.median(np.abs(calc-s.eGFR_mLmin)))}
    for x,y in [('Age','eGFR_mLmin'),('Age','HF_Risk_Binary'),('Statin','LDL_mgdL'),('Diuretic','Peripheral_Edema')]:
        z=d[[x,y]].dropna(); out[f'spearman_{x}_{y}']={'n':len(z),'r':float(spearmanr(z[x],z[y]).statistic)}
    bp=pd.to_numeric(d.SBP_mmHg,errors='coerce').dropna(); out['systolic_bp_terminal_zero_share']=float((bp.astype(int)%10==0).mean())
    out['outcome_columns_unique_counts']={c:int(d[c].nunique()) for c in ['HF_Risk_Binary','Future_Hospitalization_Risk','Survival_Probability_12mo']}
    excluded=['HF_Risk_Binary','Future_Hospitalization_Risk','Survival_Probability_12mo','Patient_ID']
    features=[c for c in d.columns if c not in excluded]
    cat=[c for c in features if not pd.api.types.is_numeric_dtype(d[c])]; num=[c for c in features if c not in cat]
    tr,te=train_test_split(d,test_size=.2,random_state=0,stratify=d.HF_Risk_Binary)
    def prep(dense):
        return ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('categorical',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=not dense)),cat)])
    models={'logistic_regression':make_pipeline(prep(False),LogisticRegression(max_iter=3000)), 'gradient_boosting':make_pipeline(prep(True),HistGradientBoostingClassifier(max_iter=400,learning_rate=.05,random_state=0))}
    out['random_split_auc']={}
    for name,model in models.items():
        model.fit(tr[features],tr.HF_Risk_Binary)
        out['random_split_auc'][name]=float(roc_auc_score(te.HF_Risk_Binary,model.predict_proba(te[features])[:,1]))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['columns_list','missing_rates']},indent=2))
if __name__=='__main__': main()
