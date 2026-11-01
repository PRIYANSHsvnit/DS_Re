# Heart failure prediction: evaluation, transport and feature groups

This repository accompanies **“A Multi-Cohort Evaluation of Heart Failure Prediction: Paired Resampling, Model Transport and Feature-Group Analysis.”** It contains the corrected IEEE-style paper, experimental code and saved results. This is methods research, not a clinical tool.

The paper combines a paired UCI SMOTE contamination experiment, UCI survival benchmarks, external assessment on SUPPORT-CHF and an exploratory feature-group analysis in a separate 50,000-row file. The new source-label holdouts are within that file; they are not verified independent hospitals. The 50,000-row file's origin, label construction and clinical prediction time remain unverified. Earlier analyses of follow-up-time leakage and survival exist, including [Rane et al.](https://arxiv.org/abs/2608.04140) and [Naeem et al.](https://thesesjournal.com/index.php/1/article/view/3986). This is an incremental combined case study; priority has not been established by a systematic review.

## Contents

- `paper/Research_Paper_IEEE.pdf` — corrected paper; the SUPPORT reference DOI was fixed after the earlier professor submission. `paper/Research_Paper_IEEE.docx` is the matching editable Word document. `paper/manuscript.md`, `paper/build_paper.py` and `paper/results/` are its source and figures.
- `code/` — paired SMOTE, endpoint tracing, SUPPORT bootstrap, 50k checks and feature-group experiments. `code/legacy/` contains the three earlier factorial and survival scripts used in the paper.
- `results/` — saved numerical results, including split-level UCI outcomes and aggregate SUPPORT and feature-group summaries.
- `data/` — UCI CSV and, in the local prepared copy, SUPPORT2 and the 50,000-row CSV. **Only the UCI file is included by Git.** The other two are excluded by `.gitignore` until their redistribution terms are confirmed.
- `reports/Plagiarism Report.pdf` and `reports/Plagiarism Report.docx` — the authors' formatted originality review, including the SciSpace result and a separate independent check. `reports/SciSpace Similarity Report.pdf` is the original 19-page SciSpace export, retained unchanged. `reports/SciSpace AI Detection Report.pdf` is a separate six-page AI detector export; its classifications do not establish authorship or negate the paper's AI-assistance disclosure. The similarity result is not a Turnitin/iThenticate report or a determination of plagiarism. `reports/FINAL_REVIEW.md` records research findings and limitations.
- `DATA_PROVENANCE.md` — dataset sources, attribution, access, and reuse limitations.

## Reproduce the results

Use Python 3.11. Create a virtual environment and install `pip install -r requirements.txt`. Commands below are run from the repository root. Outputs are written under `results/` and will overwrite existing files except for the first smoke test.

```powershell
python code/paired_smote_study.py --reps 3 --jobs 1 --output smoke_results/paired
python code/paired_smote_study.py --reps 300 --jobs 2
python code/check_smote_lineage.py
python code/support_uncertainty.py
python code/feature_group_experiment.py
python code/source_group_audit.py
python code/audit_50k.py
python code/make_figures.py
```

`support_uncertainty.py` needs `data/support2.csv`; after a fresh clone, `python code/download_support.py` downloads it from Vanderbilt. The 50k scripts need `data/cardiovascular_hf_dataset-3.csv`; obtain this file with appropriate permission and place it there. Its source-card match has not been established byte-for-byte. The new feature-group study uses 16 separate model fits and 500 paired bootstraps for selected contrasts. `make_figures.py` reads saved results and replaces the three figures under `paper/results/`.

The original factorial and survival experiments can be rerun using the scripts in `code/legacy/`, but they are computationally expensive and overwrite their saved outputs. The exact original package environment was not recorded, so numerical results may differ with versions. Some historical comments in those scripts contain superseded interpretations; use the corrected paper and provenance document for interpretation.

The current PDF was exported from the Word document. After changing the Word document, export it again in Word to retain a matching PDF. Alternatively, `python paper/build_paper.py` rebuilds an editable Word file and a separately typeset IEEE-style PDF from `paper/manuscript.md`; the two layout engines paginate differently. The exact venue template and page limit remain to be confirmed.

## Findings and limits

Across 300 dependent UCI splits, mean accuracy was 0.726 with training-only SMOTE and 0.750 in the deliberately contaminated comparison on the same original test patients. For the UCI-trained five-predictor Cox model, SUPPORT Harrell's C was 0.567 (external-patient fixed-model bootstrap interval 0.542–0.593). In the separate 50k file, the four nested feature groups gave random-split AUROCs from 0.748 to 0.962. These are conditional results, not a universal performance ceiling or evidence of clinical readiness. The earlier 50k survival notebook simulated events and is not used as observed mortality evidence.

If the earlier overlapping paper was submitted or published at a refereed venue, disclose and cite it as required and check the new venue's policies. Authors must verify author details, funding, conflicts, ethics, data permissions and the chosen venue's requirements. AI assisted coding and drafting; authors are responsible for reviewing all content and following disclosure policies. No project-wide software license has been chosen; no rights are granted beyond those specified by each underlying data source.
