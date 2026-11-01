# Research and repository review

Reviewed 1 November 2026. The current corrected paper and its matching editable Word document are in `../paper/`. The SUPPORT reference DOI in the earlier five-page professor submission was corrected in the current paper. A later wording correction clarifies that the five transport predictors were chosen for shared availability and the UCI-fitted model was not retrained on SUPPORT. Three figures and three tables are present; the PDF was exported from Word and checked for content outside the page boundaries.

## Contributions and prior work

The study combines a paired same-patient SMOTE stress test with endpoint tracing, a five-predictor UCI-to-SUPPORT transport assessment, and an exploratory nested feature-group analysis with source-label holdouts in a separate 50,000-row file. It is an incremental empirical case study; priority for this combination is not established by a systematic review.

- [Rane et al. (2026)](https://arxiv.org/abs/2608.04140) previously analysed follow-up-time leakage and survival modelling on the same 299-patient cohort.
- [Naeem et al. (2026)](https://thesesjournal.com/index.php/1/article/view/3986) compared models with and without follow-up time using repeated nested cross-validation; both are cited in the paper.
- The [HeartGuard project](https://github.com/NvyGreen/HeartGuard) reports UCI-to-MIMIC external testing. Its claims were not independently audited here. Do not claim the first external validation of a UCI-trained model.

## Verified results and remaining limits

The paired 300-split summary agrees with its saved split-level records. In a fresh-clone simulation, a three-seed rerun matched the six corresponding saved rows, the complete SUPPORT bootstrap reproduced all 18 saved estimates and bounds, and all 16 feature-group results and eight paired differences reproduced exactly. These checks used the versions listed in `../requirements.txt` and local copies of the data; they do not prove generalizability.

The two 50k files in the original workspace have different byte hashes but identical parsed values and ordering in all 50,000 rows and 68 columns. Their provenance and label construction are unverified. Source-label holdouts are within the same file, not verified independent hospitals. The earlier 50k survival notebook simulated event times and outcomes and is not used as evidence of observed mortality.

The text-overlap report in this folder is a limited independent review, not an official Turnitin/iThenticate score or certificate of originality. Its document hash identifies the current repository paper. The SUPPORT DOI error in an earlier submitted copy was corrected in the current paper.

Before formal publication, authors must verify their names, order, affiliations, funding, conflicts, ethics and data reuse conditions, and comply with the selected venue's instructions. If the earlier overlapping paper is published or under review, disclose and cite it as required and check submission policies. The work is not a completed TRIPOD+AI checklist and does not establish clinical readiness.
