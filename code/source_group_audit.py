"""Exploratory random-split and leave-one-source-label-out benchmark.

Run from the repository root: python code/source_group_audit.py.
Source labels are not verified hospitals or independent data sources.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/feature_groups'
OUT.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(ROOT / 'data/cardiovascular_hf_dataset-3.csv')
excluded = ['Patient_ID', 'Source', 'HF_Risk_Binary', 'Future_Hospitalization_Risk', 'Survival_Probability_12mo']
columns = [c for c in data.columns if c not in excluded]
demographics = ['Age', 'Gender', 'Ethnicity', 'BMI', 'Activity_Level', 'SES_Score', 'SES_Category', 'Smoking_Status', 'Alcohol_Units_Week', 'Physical_Activity_Min_Week', 'Sleep_Hours', 'Diet_Score', 'Stress_Score']
X, y = data[columns], data.HF_Risk_Binary
train, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=y)
splits = [('Random 80/20', train, test)]
for group in sorted(data.Source.unique()):
    splits.append(('Held out: ' + group, np.flatnonzero(data.Source != group), np.flatnonzero(data.Source == group)))
rows = []
for label, tr, te in splits:
    for subset, features in [('All candidate predictors', columns), ('Demographics and lifestyle', demographics)]:
        num = [c for c in features if X[c].dtype != 'object']
        cat = [c for c in features if X[c].dtype == 'object']
        preprocess = ColumnTransformer([('num', SimpleImputer(strategy='median', add_indicator=True), num), ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), cat)])
        model = make_pipeline(preprocess, HistGradientBoostingClassifier(max_iter=150, learning_rate=.08, max_leaf_nodes=31, l2_regularization=1, random_state=42))
        model.fit(X.iloc[tr][features], y.iloc[tr])
        prob = model.predict_proba(X.iloc[te][features])[:, 1]
        yt = y.iloc[te].to_numpy()
        # Conditional test-row bootstrap: model and split are held fixed.
        rng = np.random.default_rng(2026)
        boot=[]
        for _ in range(500):
            ix=rng.integers(0,len(te),len(te))
            if len(np.unique(yt[ix])) == 2: boot.append(roc_auc_score(yt[ix],prob[ix]))
        row = {'split':label,'features':subset,'n_train':len(tr),'n_test':len(te),'test_prevalence':yt.mean(),'auroc':roc_auc_score(yt,prob),'auroc_boot_low':np.quantile(boot,.025),'auroc_boot_high':np.quantile(boot,.975),'average_precision':average_precision_score(yt,prob),'brier':brier_score_loss(yt,prob)}
        rows.append(row)
        print(json.dumps(row),flush=True)
pd.DataFrame(rows).to_csv(OUT / 'source_group_benchmark.csv',index=False)
