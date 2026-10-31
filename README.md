# Heart failure prediction: leakage audit

A reproducible methodological case study of leakage, survival modeling and external validation. This is research code, not a clinical tool.

## Files

- `paper/` — the submitted six-page PDF and a visually matching Word copy. The Word copy consists of page images so its layout matches the submitted PDF; its text is not directly editable. For editing, use `code/Editable_Manuscript.docx` (same research content but a different Word layout) or `code/manuscript.typ` (the typesetting source used for the submitted PDF).
- `code/` — scripts for the paired SMOTE test, endpoint tracing, SUPPORT bootstrap, 50k audit, and related analyses. `code/legacy/` retains the original factorial, survival and literature scripts.
- `data/` — the 299-row UCI data plus local copies of SUPPORT2 and the 50k dataset. The latter two are Git-ignored pending verified redistribution rights; they will not be included by `git add .`.
- `results/` — saved numerical results and literature screening sheets; `results/legacy/` contains the prior factorial and survival results.

The submitted paper refers to the earlier folder names `phase3_leakage_audit/` and `next_experiments/`. In this cleaned repository, their scripts are under `code/legacy/` and `code/`, and their saved outputs are under `results/legacy/` and `results/` respectively. The submitted PDF itself has not been changed.

## Run the analyses

Use Python 3.11 and a virtual environment, then install `pip install -r requirements.txt`. Run commands from the repository root:

```powershell
# Fast check without overwriting saved results
python code/paired_smote_study.py --reps 3 --jobs 1 --output smoke_results/paired
# Full 300-split paired analysis and SMOTE endpoint tracing
python code/paired_smote_study.py --reps 300 --jobs 2
python code/check_smote_lineage.py
# Requires data/support2.csv; 1,000 external patient bootstraps
python code/support_uncertainty.py
# Requires data/cardiovascular_hf_dataset-3.csv
python code/audit_50k.py
# Counts completed full-text extractions; current regression is not estimated
python code/meta_regression.py
```

Full analyses overwrite their saved output files. The original 28,800 legacy runs used an older environment whose exact package versions were not recorded; results can vary with dependencies. Historical scripts in `code/legacy/` may contain superseded interpretations: the 50k checks do not prove the file is synthetic.

After a fresh clone, obtain SUPPORT2 using `python code/download_support.py`. Obtain the 50k file with the relevant permissions and place it in `data/`. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md). The local reconstructed OpenAlex abstracts and patient-level SUPPORT predictions are also Git-ignored.

To rebuild the PDF from the editable Typst source, install requirements, then run:

```powershell
python -c "import typst;typst.compile('code/manuscript.typ', output='paper/Research_Paper_IEEE.pdf', root='code')"
```

The PDF is IEEE-style; check the exact template required by the venue. The original author list and affiliation are retained. Generative AI assisted coding, screening and drafting; the authors are responsible for reviewing results and institutional disclosure requirements.

## Findings and limitations

On 300 dependent splits, mean accuracy was 0.726 with training-only SMOTE and 0.750 in a deliberately contaminated simulation, evaluated on the same original patients. In the contaminated arm, 51.5% of synthetic training rows used a held-out patient as an interpolation endpoint. External SUPPORT Cox Harrell C was 0.567 (fixed-model patient-bootstrap interval 0.542–0.593). These results do not establish a universal performance ceiling or prove a dataset's provenance. [NOVELTY_AND_LIMITATIONS.md](NOVELTY_AND_LIMITATIONS.md) discusses prior work and the scope of the contribution.

UCI data are licensed CC BY 4.0; see provenance for attribution. No project-wide software license has been selected.
