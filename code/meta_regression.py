"""Analyze manually extracted full-text study data; never infer flags from abstracts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT.parent / 'results/legacy/literature/screening_sheet.csv'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=DEFAULT)
    ap.add_argument('--output', type=Path, default=ROOT.parent / 'results/literature')
    args = ap.parse_args()
    df = pd.read_csv(args.input, dtype=str).fillna('')
    cols = {'include':'SCREEN_include(Y/N)', 'dataset':'FT_dataset_version(299/5000/other)',
            'time':'FT_time_used_as_feature(Y/N/unclear)', 'smote':'FT_resampling_before_split(Y/N/unclear/none)',
            'accuracy':'FT_reported_accuracy', 'notes':'FT_notes'}
    missing = set(cols.values()) - set(df.columns)
    if missing: raise SystemExit(f'Missing required columns: {sorted(missing)}')
    def norm(s): return s.astype(str).str.strip().str.lower()
    included = df[norm(df[cols['include']]).isin(['y','yes']) & (norm(df[cols['dataset']]) == '299')].copy()
    for name in ['time','smote']:
        d = norm(included[cols[name]])
        included[name] = d.map({'y':1, 'yes':1, 'n':0, 'no':0, 'none':0})
    included['accuracy'] = pd.to_numeric(included[cols['accuracy']], errors='coerce')
    invalid = included.accuracy.notna() & ~included.accuracy.between(0,1)
    if invalid.any(): raise SystemExit('Accuracy must be recorded on the 0–1 scale; correct the extraction sheet.')
    eligible = included.dropna(subset=['accuracy','time','smote']).copy()
    args.output.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.output / 'eligible_studies.csv', index=False)
    lines = [f'Screening records: {len(df)}', f'Included 299-row studies: {len(included)}',
             f'Complete cases for regression: {len(eligible)}',
             'Flags must be supported by full-text quotations and page numbers in FT_notes.',
             'Associations are observational; differences in validation and selection may confound them.']
    if len(eligible) < 10 or eligible.time.nunique() < 2 or eligible.smote.nunique() < 2:
        lines.append('Regression not estimated: fewer than 10 complete studies or a flag has no variation.')
    else:
        import statsmodels.api as sm
        X = sm.add_constant(eligible[['time','smote']].astype(float), has_constant='add')
        if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
            lines.append('Regression not estimated: flag columns are collinear.')
        else:
            fit = sm.OLS(eligible.accuracy.astype(float), X).fit(cov_type='HC3')
            ci = fit.conf_int()
            pd.DataFrame({'coefficient':fit.params, 'ci_low':ci[0], 'ci_high':ci[1], 'p_value':fit.pvalues}).to_csv(args.output / 'regression.csv')
            lines += ['Exploratory unweighted study-level OLS, HC3 standard errors; coefficients are accuracy proportions.', str(fit.summary())]
    (args.output / 'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__': main()
