"""Nested candidate-feature sets, with train-only preprocessing and paired test bootstrap.

Feature sets are hypotheses about availability, not verified prediction-time eligibility.
The source-label tests remain within one file and are not external validation.
Run from the repository root with: python code/feature_group_experiment.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/feature_groups'
OUT.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(ROOT / 'data/cardiovascular_hf_dataset-3.csv')
exclude = ['Patient_ID','Source','HF_Risk_Binary','Future_Hospitalization_Risk','Survival_Probability_12mo']
all_features = [c for c in data if c not in exclude]
demo = ['Age','Gender','Ethnicity','BMI','Activity_Level','SES_Score','SES_Category','Smoking_Status','Alcohol_Units_Week','Physical_Activity_Min_Week','Sleep_Hours','Diet_Score','Stress_Score']
comorb_vitals = ['Hypertension','Diabetes','CAD','Prior_MI','Arrhythmia','CKD','Family_HF_History','Heart_Rate_bpm','SBP_mmHg','DBP_mmHg','Respiratory_Rate','SpO2_pct']
labs = ['BNP_pgmL','NT_proBNP_pgmL','Troponin_I_ngmL','HbA1c_pct','LDL_mgdL','HDL_mgdL','Triglycerides_mgdL','Creatinine_mgdL','eGFR_mLmin','Sodium_mEqL','Potassium_mEqL','CRP_mgL']
tests = ['QTc_Interval_ms','QRS_Duration_ms','ST_Abnormality','LVH_Flag','AF_Presence','Ejection_Fraction_pct','LV_Wall_Thickness_mm','Cardiomegaly','Echo_Abnormalities_Score']
sets = [('Demographics and lifestyle', demo), ('Plus comorbidities and vitals',demo+comorb_vitals), ('Plus labs, ECG and echo',demo+comorb_vitals+labs+tests), ('All candidate predictors',all_features)]
assert all(len(set(cols)) == len(cols) and set(cols)<=set(all_features) for _,cols in sets)
y=data.HF_Risk_Binary
tr,te=train_test_split(np.arange(len(data)),test_size=.2,random_state=42,stratify=y)
splits=[('Random 80/20',tr,te)] + [('Held out: '+g,np.flatnonzero(data.Source!=g),np.flatnonzero(data.Source==g)) for g in sorted(data.Source.unique())]
rows=[]; delta=[]
for split,tr,te in splits:
    probs={}; yt=y.iloc[te].to_numpy(); rng=np.random.default_rng(20261101)
    samples=rng.integers(0,len(te),size=(500,len(te)))
    for name,features in sets:
        cats=[c for c in features if data[c].dtype=='object']; nums=[c for c in features if c not in cats]
        p=ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=True),nums),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats)])
        model=make_pipeline(p,HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=31,l2_regularization=1,random_state=42))
        model.fit(data.iloc[tr][features],y.iloc[tr]); prob=model.predict_proba(data.iloc[te][features])[:,1];probs[name]=prob
        auc=roc_auc_score(yt,prob)
        row={'split':split,'features':name,'n_features':len(features),'n_train':len(tr),'n_test':len(te),'test_prevalence':float(yt.mean()),'auroc':auc,'average_precision':average_precision_score(yt,prob),'brier':brier_score_loss(yt,prob)}
        rows.append(row);print(json.dumps(row),flush=True)
    reference=probs['Plus comorbidities and vitals']
    for name in ['Plus labs, ECG and echo','All candidate predictors']:
        contrast=roc_auc_score(yt,probs[name])-roc_auc_score(yt,reference)
        boot=[roc_auc_score(yt[ix],probs[name][ix])-roc_auc_score(yt[ix],reference[ix]) for ix in samples]
        delta.append({'split':split,'comparison':name+' minus comorbidities and vitals','auroc_difference':contrast,'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'bootstraps':len(boot)})
pd.DataFrame(rows).to_csv(OUT/'feature_group_metrics.csv',index=False)
pd.DataFrame(delta).to_csv(OUT/'feature_group_paired_differences.csv',index=False)
