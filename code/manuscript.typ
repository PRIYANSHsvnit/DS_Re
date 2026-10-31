#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.65in), columns: 2, footer: context align(center, text(size: 8pt, counter(page).display())))

#set text(font: "Times New Roman", size: 9.5pt)

#set par(justify: true, leading: 0.48em, spacing: 0.55em)

#set place(clearance: 7pt)

#place(top, scope: "parent", float: true, clearance: 11pt)[

#align(center)[#text(size: 17pt, weight: "bold")[Reported Performance, Leakage and External Validation in Heart Failure Prediction: A Factorial and Paired Audit]
#v(7pt)
#text(size: 10pt)[\[Vivek Patel\], \[Umang Patel\], \[Jeetu Mehra\], \[Priyansh T\]]
#linebreak()
#text(size: 9pt)[\[DoAI, SVNIT, Surat, India\]  |  \[u24ai040\@aid.svnit.ac.in\]]]

]

#par[#text(weight: "bold")[Abstract—]Machine-learning studies using the 299-patient Heart Failure Clinical Records dataset report widely varying performance. We examine whether selected evaluation choices can explain some of that variation and assess transfer to an independent heart-failure cohort. Two existing factorial experiments varied follow-up time, resampling, outlier handling, feature selection and test-set tuning. A new paired experiment evaluated training-only and deliberately contaminated SMOTE on the same original patients across 300 splits. With follow-up time excluded, mean accuracy was 72.6% and 75.0%, respectively; 51.5% of synthetic training rows in the contaminated arm used a held-out patient as an interpolation endpoint. In 1,384 SUPPORT-CHF patients, a UCI-trained Cox model achieved Harrell C=0.567 (external-patient bootstrap 95% interval 0.542–0.593). Its mean predicted 180-day risk exceeded the observed rate by 17.5 percentage points (15.0–19.9). We also report an exploratory abstract-level literature comparison and a plausibility audit of a separate 50,000-row file. The results show sensitivity to evaluation choices and a calibration gap, but do not establish an attainable-performance ceiling or prove the provenance of the larger dataset.]

#par[#text(weight: "bold")[Index Terms—]heart failure, data leakage, SMOTE, survival analysis, external validation, calibration, data provenance.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[I. INTRODUCTION]]]

#par[Heart failure (HF) prognosis has become a common benchmark for clinical machine learning. The Heart Failure Clinical Records dataset comprises 299 patients with 96 recorded deaths and follow-up ranging from 4 to 285 days \[1\], \[2\]. Its accessibility supports reproducible methods research, but its small size and variable follow-up make evaluation choices especially consequential.]

#par[Reported results span a wide range; for example, Ali et al. reported 97.78% accuracy using a random forest \[3\]. A predictive model intended for use at baseline cannot use a patient’s eventual follow-up duration as an input. Likewise, applying SMOTE to a complete dataset before splitting can cause synthetic training observations to depend on test patients \[4\], \[5\], \[6\]. The magnitude of the resulting bias depends on the data, model and evaluation procedure.]

#par[These issues have been studied previously: Rane et al. quantify follow-up-time leakage on this dataset \[7\], and Srujana et al. examine methods for censoring \[8\]. Our work combines a factorial audit with a controlled same-patient resampling comparison and external validation. The separate 50,000-row dataset is assessed as a provenance case study; its origin has not been established. We do not claim priority for any individual technique or for external validation.]

#par[Contributions. We report (C1) factorial comparisons of selected pipeline choices, (C2) censoring-aware reference results, (C3) an exploratory abstract-level comparison, (C4) five-predictor transfer to SUPPORT-CHF with external-patient uncertainty intervals, (C5) a descriptive plausibility audit, and (C6) a paired SMOTE simulation that holds the original test patients fixed.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[II. RELATED WORK]]]

#par[Leakage in ML-based science. Kaufman et al. formalised leakage \[5\], and Kapoor and Narayanan documented leakage across multiple fields \[4\]. SMOTE generates minority-class observations by interpolation \[6\]; applying it before partitioning can connect training examples to held-out patients. Class-imbalance correction can also affect risk calibration, so its use should be evaluated rather than assumed beneficial \[9\]. TRIPOD+AI provides reporting guidance for prediction-model studies \[10\].]

#par[This dataset. Rane et al. quantified follow-up-time leakage \[7\], Srujana et al. studied censoring-aware methods \[8\], and Naeem et al. compared the same cohort with and without follow-up time using nested cross-validation \[11\]. These analyses establish substantial prior work on leakage in this cohort. Our contribution is the specific paired SMOTE simulation and its endpoint tracing, considered alongside a SUPPORT transport analysis. A targeted search cannot establish priority for the combined design.]

#par[Data provenance. Gibson et al. investigated uncertain provenance in clinical prediction datasets \[12\]. A public data card \[13\] describes a file with the same name, dimensions and source counts as our local 50,000-row file, and describes it ambiguously as “synthetic or de-identified.” Its gated data were not accessed and byte identity was not verified. Descriptive plausibility checks cannot determine origin or identify a generating mechanism.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[III. DATA AND METHODS]]]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[A. Data]]

#par[Development cohort: UCI Heart Failure Clinical Records \[1\], \[2\]: 299 patients (105 women, age 40–95), 96 deaths, follow-up 4–285 d, 11 baseline predictors (age, anaemia, CPK, diabetes, ejection fraction (EF), hypertension, platelets, serum creatinine, serum sodium, sex, smoking), plus time and DEATH\_EVENT. External cohort: the CHF subgroup of SUPPORT \[14\] (five US centres, 1989–94; public data from Vanderbilt Biostatistics): 1,384 of 1,387 patients with creatinine recorded. Follow-up was administratively censored at 280 d to match UCI; no patient was censored earlier. Five predictors are shared: age, sex, creatinine, sodium, diabetes. Audited dataset: a 50,000-row, 68-column file presented as three care settings with binary target HF\_Risk\_Binary.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[B. Factorial Leakage Experiments (C1)]]

#par[Experiment 1 crossed three factors: O, sequential 1.5×IQR outlier removal on the full data; S, SMOTE on the full data before splitting (versus inside the training set); T, time as a predictor. Each of the eight pipelines used the classifiers and hyper-parameters of \[3\] (random forest: 65 trees, depth 8; decision tree: Gini, depth 5; XGBoost: learning rate 0.59) on 1,000 random stratified 70/30 splits (24,000 evaluations). Experiment 2 crossed T, S, F (the five most informative features by mutual information selected on all rows) and H (the best of 30 random random-forest configurations selected by test accuracy), on raw data, with 300 splits per cell (2⁴ factorial; 4,800 evaluations) and a four-way ANOVA on accuracy. We report means, 95th percentiles, best single split, and the share of splits reaching 97.78%; we also compute the mean best of 10 random splits to emulate re-running until a favourable figure appears.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[B.1 Paired SMOTE Sensitivity Experiment (C6)]]

#par[For each of 300 stratified 70/30 splits, we compared two 65-tree random forests evaluated on the same 90 original test patients. The training-only arm applied SMOTE after the split. In the deliberately contaminated arm, SMOTE was fitted to all 299 records, after which synthetic examples were sampled into the original training partition. Both arms retained identical original training patients and equal training class counts. The test set contained no synthetic observations. The simulation was repeated with and without the follow-up-time feature; it is a stress test, not a reconstruction of any published implementation.]

#par[For split k, let Aₖ⁽ᶜ⁾ and Aₖ⁽ᵗ⁾ denote accuracy under contaminated and training-only SMOTE. We report the mean paired difference over K=300 splits:]

#par[ΔA = (1/K) × sum over k \[A(contaminated, k) − A(training-only, k)\].         (1)]

#par[The 2.5th and 97.5th percentiles of these paired differences describe split-to-split variability, not a population confidence interval: the splits reuse the same 299 people. We reconstructed SMOTE’s random draws to identify synthetic training samples whose interpolation endpoints included a test patient. Standard SMOTE was applied to unscaled variables, including binary fields; the resulting values may be fractional.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[C. Leakage-free Reference and Survival Benchmark (C2)]]

#par[Seven prespecified pipelines excluding follow-up time were each evaluated on 1,000 stratified 70/30 splits: random forests with and without class weights or training-only SMOTE, logistic regression, XGBoost, and an RBF support vector machine. The most favourable statistic across the seven is used only as a descriptive reference. For survival analysis, we compared penalised Cox regression \[15\], random survival forest \[16\], gradient-boosted survival and a binary random-forest classifier with 10×5-fold cross-validation, using Harrell C \[17\], Uno C at τ=240 days \[18\], time-dependent AUC and integrated Brier score. Hazard ratios were estimated using multivariable Cox regression, bootstrap intervals and Schoenfeld tests.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[D. External Validation (C4)]]

#par[Five-predictor models were fitted on all 299 UCI records and applied to SUPPORT-CHF using age, sex, log-creatinine, sodium and diabetes. UCI statistics were used for standardization. We report C-indices, 180-day performance and mean predicted versus observed 180-day mortality; the earlier analysis also calculated calibration slopes \[19\]. Internal UCI and within-SUPPORT repeated five-fold CV provide context, not performance ceilings. For external uncertainty, models were held fixed while 1,000 bootstrap samples of SUPPORT patients were drawn; percentile intervals reflect external-cohort sampling variability conditional on the fitted models.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[E. Literature Analysis (C3)]]

#par[OpenAlex \[20\] was queried for citations and keywords; the saved screening sheet contains 209 candidate records. A prior AI-assisted analysis coded 90 abstracts containing performance figures and classified 60 as eligible. This is exploratory abstract-level coding: the full-text fields remain incomplete, selection and extraction have not been independently checked, and no causal attribution to a specific pipeline is possible. We use the saved coding as a descriptive comparison, not a completed systematic review.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[F. Dataset Plausibility Audit (C5)]]

#par[The earlier 11-check audit examined distributions by care setting, missingness, eGFR agreement, predictor associations, terminal digits, outcome fields and model performance in the separate 50,000-row file. We independently recalculated key statistics. eGFR comparison uses the 2021 creatinine-based CKD-EPI equation \[21\] and assumes compatible units, measurements and timing; deviations do not identify dataset provenance.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[G. Assumptions and Approximations]]

#par[Random 70/30 splits reuse the same patients. Split-to-split variation is conditional on this sample and does not quantify uncertainty from drawing a new clinical cohort or prove that the original sample represents the broader Faisalabad population.]

#par[The most favourable statistics across seven specified pipelines serve as a descriptive reference, not as an upper bound on attainable performance. Selection of the most favourable statistic and differences across studies complicate direct comparisons.]

#par[The literature flags reflect what was coded from abstracts; an unmentioned method is unknown rather than absent. Full texts and independent coding are needed for stronger claims.]

#par[SUPPORT follow-up was truncated at 280 days; no selected SUPPORT patient was censored before day 180. UCI and SUPPORT differ in setting, era and available predictors. The UCI proportion among patients with known status at 180 days is not a censoring-adjusted estimate.]

#par[Leak mechanisms are implemented in the most direct form (e.g. mutual-information top-5 on all rows; best of 30 configurations by test accuracy); real studies may leak more or less.]

#par[Hazard-ratio interpretation depends on proportional hazards. The Schoenfeld test for EF gave p=0.024 in the saved analysis; this suggests a possible violation and warrants cautious interpretation, especially with a small sample and multiple tests.]

#par[The eGFR/CKD-EPI check assumes the recorded eGFR is intended to follow the standard equation.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[IV. RESULTS]]]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[A. Leakage Decomposition in the Tested Pipelines]]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#text(size: 8pt, weight: "bold")[TABLE I — RANDOM FOREST UNDER EIGHT PIPELINES, 1,000 SPLITS EACH]]
#v(3pt)
#set text(size: 8pt)
#table(columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr,), inset: 3pt, align: center, stroke: none, table.hline(stroke: 0.5pt),
[Leaks],
[Test n],
[Acc. mean±SD],
[AUROC],
[Acc. p95],
[Best],
[none],
[90],
[0.726±0.038],
[0.768],
[0.789],
[0.856],
[none+T],
[90],
[0.840±0.035],
[0.906],
[0.900],
[0.956],
[O],
[68],
[0.725±0.043],
[0.737],
[0.794],
[0.868],
[O+T],
[68],
[0.833±0.037],
[0.895],
[0.897],
[0.941],
[S],
[122],
[0.799±0.034],
[0.886],
[0.853],
[0.902],
[S+T],
[122],
[0.881±0.028],
[0.952],
[0.926],
[0.967],
[O+S],
[98],
[0.819±0.036],
[0.903],
[0.878],
[0.929],
[O+S+T],
[98],
[0.885±0.030],
[0.958],
[0.929],
[0.969],
table.hline(stroke: 0.5pt))
#v(2pt)
#text(size: 8pt)[O: outlier removal; S: SMOTE before split; T: time as predictor. None reached 97.78%.]
]

#par[The leakage-free pipeline averaged 72.6% (AUROC 0.768). Averaged over the other factors, T raised accuracy by 9.3 points (AUROC +0.104), S by 6.5 (+0.098) and O by 0.4. Effects were similar for the decision tree (T +8.7, S +4.6) and XGBoost (T +10.0, S +6.9). With O, S and T combined the mean was 88.5%; none of 24,000 evaluations reached 97.78% (best 96.9%).]

#par[The reported 97.78% is consistent with 88 correct predictions out of 90, but a rounded percentage does not uniquely identify the denominator. Our simulations did not reproduce that value under the tested configurations and seeds; this observation does not establish that another implementation is impossible or that a published study used undocumented steps.]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#image("figures/figure1.png", width: 100%, height: 2.25in, fit: "contain")]
#text(size: 8pt)[Fig. 1. Random-forest accuracy over 1,000 splits per tested pipeline. Segments show the 5th–95th percentiles, circles the means and ticks the best splits. Test populations differ when full-data SMOTE introduces synthetic test rows.]
]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[B. Extended Leaks: Feature Selection and Test-set Tuning]]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#text(size: 8pt, weight: "bold")[TABLE II — FOUR-FACTOR LEAKAGE GRID (RAW DATA, SELECTED CELLS)]]
#v(3pt)
#set text(size: 8pt)
#table(columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr,), inset: 3pt, align: center, stroke: none, table.hline(stroke: 0.5pt),
[Leaks],
[Acc. mean],
[p95],
[Best],
[Mean best-of-10],
[≥ 97.78%],
[none],
[0.735],
[0.789],
[0.833],
[0.785],
[0.0%],
[H],
[0.772],
[0.822],
[0.856],
[0.820],
[0.0%],
[F],
[0.743],
[0.800],
[0.844],
[0.801],
[0.0%],
[FH],
[0.777],
[0.833],
[0.878],
[0.830],
[0.0%],
[S],
[0.799],
[0.853],
[0.902],
[0.849],
[0.0%],
[T],
[0.843],
[0.889],
[0.933],
[0.891],
[0.0%],
[TS],
[0.882],
[0.926],
[0.967],
[0.925],
[0.0%],
[TSH],
[0.896],
[0.934],
[0.975],
[0.934],
[0.0%],
[TSF],
[0.873],
[0.918],
[0.943],
[0.916],
[0.0%],
[TSFH],
[0.889],
[0.926],
[0.951],
[0.927],
[0.0%],
table.hline(stroke: 0.5pt))
#v(2pt)
#text(size: 8pt)[T: time; S: SMOTE before split; F: feature selection on all rows; H: best of 30 configurations chosen on test accuracy.]
]

#par[In the extended grid, the observed marginal accuracy differences were +8.7 percentage points for time, +4.1 for pre-split SMOTE, −0.3 for full-data feature selection and +2.5 for selecting hyperparameters on the test set. These are descriptive averages across the tested settings; in particular, the SMOTE arms use different test populations. The highest single-split accuracy in this grid was 97.5%; no run reached 97.78%. The study cannot exclude other algorithms, split choices or data preparations.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[B.1. Paired SMOTE Results]]

#par[With follow-up time excluded, mean accuracy was 0.726 for training-only SMOTE and 0.750 in the contaminated simulation (ΔA=+0.024). Mean AUROC was 0.767 versus 0.809. The 2.5th–97.5th percentiles of paired accuracy differences were −0.033 to +0.078. Across splits, 51.5% of contaminated synthetic training samples used at least one test patient as an interpolation endpoint, and 95.9% had a fractional value in at least one binary field. With time included, mean accuracy was 0.838 versus 0.856. Results depend on the cohort, distance metric, unscaled predictors and simulation design.]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[C. Performance of Selected Baseline Pipelines]]

#par[Across the seven honest pipelines, mean accuracy ranged from 72.1% to 74.0% and mean AUROC from 0.733 to 0.788; the majority-class rate is 67.9%. The most favourable 95th percentiles were 80.0% accuracy and 0.851 AUROC; the best of 7,000 splits reached 86.7% and 0.906. Survival models reached C ≈ 0.73, only about 0.03 above Cox regression (Table III).]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#text(size: 8pt, weight: "bold")[TABLE III — CENSORING-AWARE BENCHMARK, UCI, 10×5-FOLD CV]]
#v(3pt)
#set text(size: 8pt)
#table(columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr,), inset: 3pt, align: center, stroke: none, table.hline(stroke: 0.5pt),
[Model],
[Harrell C],
[Uno C],
[AUC 90d],
[AUC 180d],
[IBS],
[GB survival],
[0.731±0.055],
[0.745],
[0.746],
[0.755],
[0.163],
[Random surv. forest],
[0.729±0.059],
[0.749],
[0.737],
[0.763],
[0.158],
[RF (ignores time)],
[0.722±0.060],
[0.745],
[0.728],
[0.756],
[-],
[Cox PH],
[0.700±0.059],
[0.712],
[0.698],
[0.717],
[0.163],
table.hline(stroke: 0.5pt))
]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[D. Exploratory Comparison with Published Abstracts]]

#par[Among the 90 AI-assisted coded abstracts, 60 were classified as eligible and 48 reported accuracy. The median reported accuracy was 89.8%; 43/48 exceeded the selected internal split reference of 80.0%. This screening is incomplete and unverified; the studies may differ in outcomes, populations, test sets and model selection. The comparison is descriptive and cannot determine how many published results were affected by leakage. Full-text extraction and independent review are required.]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#image("figures/figure2.png", width: 100%, height: 2.25in, fit: "contain")]
#text(size: 8pt)[Fig. 2. Accuracy values in 48 provisionally coded abstracts. The dashed line is the 95th percentile of one selected internal reference, not a boundary on possible performance. The coding and study eligibility have not been independently verified.]
]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[E. External Validation on SUPPORT-CHF]]

#par[For a model with predicted 180-day risks pᵢ and observed event indicators yᵢ in SUPPORT, we summarize average risk difference as:]

#par[Δrisk = (1/n) × sum over i (pᵢ − yᵢ).                                         (2)]

#par[Here all selected SUPPORT patients had known status through 180 days. A 1,000-sample patient-level bootstrap supplies percentile intervals for Δrisk and C-index, conditional on the fitted UCI model. This does not capture uncertainty from model training or future clinical settings.]

#par[In SUPPORT-CHF, 27.0% died within 180 days; UCI survival-model predictions averaged 44–46%. Internal UCI Harrell C was 0.67–0.69 across the four saved models, while external C was 0.55–0.57. Within-SUPPORT cross-validation gave a contextual comparison, not a ceiling. For Cox PH, external C was 0.567 (95% external bootstrap interval 0.542–0.593) and mean predicted minus observed 180-day mortality was 0.175 (0.150–0.199). Bootstrap intervals hold the UCI-trained model fixed and do not include development-sample uncertainty. The earlier UCI 46.1% known-status proportion excludes patients censored before 180 days and is not a censoring-adjusted estimate.]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#text(size: 8pt, weight: "bold")[TABLE IV — TRANSPORT OF SHARED-PREDICTOR MODELS TO SUPPORT-CHF]]
#v(3pt)
#set text(size: 8pt)
#table(columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr,), inset: 3pt, align: center, stroke: none, table.hline(stroke: 0.5pt),
[Model],
[C int.],
[C ext.],
[C within SUPPORT],
[Pred. 180d (obs 27.0%)],
[Cal. slope],
[Cox PH],
[0.681],
[0.567],
[0.573],
[44.5%],
[0.21],
[RSF],
[0.682],
[0.554],
[0.562],
[44.1%],
[0.25],
[GB surv.],
[0.688],
[0.552],
[0.570],
[46.4%],
[0.17],
[RF clf.],
[0.674],
[0.553],
[0.548],
[-],
[-],
table.hline(stroke: 0.5pt))
]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#image("figures/figure3.png", width: 100%, height: 2.25in, fit: "contain")]
#text(size: 8pt)[Fig. 3. Performance of five-predictor models internally, externally and within SUPPORT; observed and predicted 180-day mortality in SUPPORT. The cohorts differ in era, setting and predictor distributions.]
]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[F. Risk Factors]]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#text(size: 8pt, weight: "bold")[TABLE V — COX HAZARD RATIOS, UCI, BOOTSTRAP 95% CI (PER SD FOR CONTINUOUS)]]
#v(3pt)
#set text(size: 8pt)
#table(columns: (1fr, 1fr, 1fr,), inset: 3pt, align: center, stroke: none, table.hline(stroke: 0.5pt),
[Predictor],
[HR (95% CI)],
[CI excludes 1],
[Age],
[1.67 (1.32–2.27)],
[yes],
[Ejection fraction],
[0.61 (0.42–0.78)],
[yes],
[Creatinine (log)],
[1.55 (1.25–2.00)],
[yes],
[Hypertension],
[1.66 (1.08–2.68)],
[yes],
[Anaemia],
[1.60 (1.02–2.60)],
[yes],
[Sodium],
[0.86 (0.65–1.11)],
[no],
[CPK (log)],
[1.11 (0.86–1.43)],
[no],
[Platelets],
[0.96 (0.75–1.20)],
[no],
[Diabetes],
[1.15 (0.70–1.81)],
[no],
[Smoking],
[1.23 (0.71–2.15)],
[no],
[Male sex],
[0.77 (0.44–1.29)],
[no],
table.hline(stroke: 0.5pt))
]

#par[In the UCI Cox model, bootstrap intervals for age, EF, creatinine, hypertension and anaemia excluded one, while intervals for platelets and several other variables did not. These estimates are associations, not proof of causality or of absent effects. The EF proportional-hazards test was significant in this analysis; the corresponding hazard ratio should therefore be interpreted as an average over follow-up. EF was unavailable in SUPPORT, and continuous variables were standardized separately by cohort.]

#place(top, scope: "parent", float: true, clearance: 8pt)[
#align(center)[#image("figures/figure4.png", width: 100%, height: 2.05in, fit: "contain")]
#text(size: 8pt)[Fig. 4. Cox hazard ratios (95% CI) for five shared predictors in UCI and SUPPORT-CHF. Continuous predictors were standardized within each cohort, so a one-standard-deviation increase differs between cohorts.]
]

#block(above: 4pt, below: 2pt, breakable: false, sticky: true)[#text(weight: "bold", size: 9.5pt)[G. Fidelity of the 50,000-row Dataset]]

#par[The 50,000-row dataset has 68 columns. Recorded and calculated eGFR correlated at 0.467 among 40,604 complete records; mean EF was 50.6% for the inpatient-labelled group and 50.7% for the annual-checkup group. On a random split, logistic regression and gradient boosting achieved AUROC 0.952 and 0.965 for HF\_Risk\_Binary. The interpretation of this target as an observed clinical event is unverified. These findings motivate provenance checks but do not prove synthetic origin, a particular generator or clinical validity.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[V. DISCUSSION]]]

#par[Principal findings. Selected leakage-related choices materially changed apparent performance in the tested settings. Our paired SMOTE experiment shows a difference even when the original test patients are held fixed. These observations cannot establish a universal ceiling or attribute a particular literature result to a specific error.]

#par[Why external validation matters. Transferred models showed weak discrimination and overpredicted average 180-day mortality in SUPPORT. Differences in population, setting, observation time and measurement limit causal interpretation. A within-SUPPORT comparison provides context but does not define the best performance possible with these predictors.]

#par[Risk factors. The fitted Cox models identify associations in these cohorts, subject to sampling uncertainty, model assumptions and different predictor scales. EF is not available in SUPPORT, and failure to reject a null hypothesis for a predictor does not establish that it has no clinical effect.]

#par[Reporting checklist. Define the prediction time and outcome; keep resampling, preprocessing and feature selection within training data; report the distribution of performance across prespecified splits with suitable baselines; evaluate discrimination and calibration; validate externally where possible; and document data provenance. For fixed-horizon binary outcomes, early censoring requires an appropriate censoring-aware approach rather than an automatic assumption of survival.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[Threats to Validity]]]

#par[Internal validity: leak mechanisms are simulations and repeated splits of one cohort are dependent. Construct validity: the classification outcome is death recorded during variable follow-up, not a common fixed-horizon label; the literature coding is abstract-only and AI-assisted. External validity: SUPPORT shares five predictors, does not include EF and differs in setting and era. Conclusion validity: bootstrap intervals are conditional on the fitted UCI model; they do not account for model-development uncertainty or other hospitals.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[Limitations]]]

#par[The full-text review, double coding and a second external validation were not completed. Applying standard SMOTE to unscaled, mixed clinical variables may produce fractional binary values and is sensitive to feature scales. The 50,000-row file has unverified provenance and outcome definitions. Our clinical interpretation is limited by the small UCI cohort, censoring, model assumptions and differences between cohorts.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[VI. FUTURE WORK]]]

#par[Complete independent full-text screening and extraction of the 209 candidate records; perform any meta-regression only after sufficient verified coding and address study-level confounding.]

#par[Consider a second external cohort such as Zigong only after authorized access and verification of baseline timing, units and compatible mortality outcomes.]

#par[If a separate 5,000-row derivative is obtained and identified, examine its provenance and possible near-duplicate leakage.]

#par[A prospective reporting-checklist study: re-analysis of retracted or corrected ML papers to test whether the checklist would have flagged them.]

#par[Extending the factorial design to time-dependent covariate leakage and to other small public clinical benchmarks.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[VII. CONCLUSION]]]

#par[In this case study, selected leakage mechanisms changed observed performance, and models trained on UCI transferred with limited discrimination and miscalibration to SUPPORT-CHF. The paired experiment isolates the test-patient composition more carefully than the initial full-data SMOTE comparison. These findings support explicit prediction-time definitions, training-only preprocessing, censoring-aware evaluation and external validation; they do not establish a universal bound on accuracy.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[DATA AND CODE AVAILABILITY]]]

#par[UCI Heart Failure Clinical Records and SUPPORT2 are publicly available \[2\], \[14\]. SUPPORT2 data were obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics. Scripts, parameters and results accompany this project in phase3\_leakage\_audit/ and next\_experiments/; software includes scikit-learn, scikit-survival and lifelines \[22\], \[23\], \[24\]. The source and outcome definitions of the local 50,000-row file remain unverified.]

#block(above: 6pt, below: 3pt, breakable: false, sticky: true)[#align(center)[#text(weight: "bold", size: 9.5pt)[REFERENCES]]]

#par[#text(size: 8pt)[\[1\] T. Ahmad, A. Munir, S. H. Bhatti, M. Aftab, and M. A. Raza, “Survival analysis of heart failure patients: A case study,” PLoS ONE, vol. 12, no. 7, p. e0181001, 2017.]]

#par[#text(size: 8pt)[\[2\] D. Chicco and G. Jurman, “Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone,” BMC Med. Inform. Decis. Mak., vol. 20, p. 16, 2020.]]

#par[#text(size: 8pt)[\[3\] M. M. Ali et al., “A machine learning approach for risk factors analysis and survival prediction of heart failure patients,” Healthc. Anal., vol. 3, p. 100182, 2023.]]

#par[#text(size: 8pt)[\[4\] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, p. 100804, 2023.]]

#par[#text(size: 8pt)[\[5\] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” ACM Trans. Knowl. Discov. Data, vol. 6, no. 4, p. 15, 2012.]]

#par[#text(size: 8pt)[\[6\] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.]]

#par[#text(size: 8pt)[\[7\] A. Rane, A. Choudhari, S. Kant, and A. Deep, “Discrete-time survival analysis for heart failure mortality prediction,” arXiv:2608.04140, 2026.]]

#par[#text(size: 8pt)[\[8\] B. Srujana, D. Verma, and S. Naqvi, “Machine learning vs. survival analysis models: A study on right censored heart failure data,” Commun. Stat. Simul. Comput., 2022, doi: 10.1080/03610918.2022.2060510.]]

#par[#text(size: 8pt)[\[9\] R. van den Goorbergh, M. van Smeden, D. Timmerman, and B. Van Calster, “The harm of class imbalance corrections for risk prediction models,” J. Am. Med. Inform. Assoc., vol. 29, no. 9, pp. 1525–1534, 2022.]]

#par[#text(size: 8pt)[\[10\] G. S. Collins et al., “TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods,” BMJ, vol. 385, p. e078378, 2024.]]

#par[#text(size: 8pt)[\[11\]  A. Naeem, A. Ahmad, and K. Khan, “Leakage-controlled evaluation of machine learning for heart disease diagnosis and heart failure mortality prediction,” Spectrum of Engineering Sciences, vol. 4, no. 3, pp. 7526–7546, 2026. https://thesesjournal.com/index.php/1/article/view/3986.]]

#par[#text(size: 8pt)[\[12\]  A. D. Gibson, N. M. White, G. S. Collins, and A. G. Barnett, “Evidence of Unreliable Data and Poor Data Provenance in Clinical Prediction Model Research and Clinical Practice,” medRxiv, 2026, doi: 10.64898/2026.02.24.26347028 (preprint).]]

#par[#text(size: 8pt)[\[13\]  Auric Grid Laboratory, “Cardiovascular Heart Failure Dataset,” dataset card. https://huggingface.co/datasets/Auric-Grid/E1.L1-Cardiovascular-Heart-Failure-Risk-Prediction (accessed Oct. 31, 2026).]]

#par[#text(size: 8pt)[\[14\] W. A. Knaus et al., “The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults,” Ann. Intern. Med., vol. 122, no. 3, pp. 191–203, 1995.]]

#par[#text(size: 8pt)[\[15\] D. R. Cox, “Regression models and life-tables,” J. R. Stat. Soc. B, vol. 34, no. 2, pp. 187–220, 1972.]]

#par[#text(size: 8pt)[\[16\] H. Ishwaran, U. B. Kogalur, E. H. Blackstone, and M. S. Lauer, “Random survival forests,” Ann. Appl. Stat., vol. 2, no. 3, pp. 841–860, 2008.]]

#par[#text(size: 8pt)[\[17\] F. E. Harrell, R. M. Califf, D. B. Pryor, K. L. Lee, and R. A. Rosati, “Evaluating the yield of medical tests,” JAMA, vol. 247, no. 18, pp. 2543–2546, 1982.]]

#par[#text(size: 8pt)[\[18\] H. Uno, T. Cai, M. J. Pencina, R. B. D'Agostino, and L. J. Wei, “On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data,” Stat. Med., vol. 30, no. 10, pp. 1105–1117, 2011.]]

#par[#text(size: 8pt)[\[19\] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, “Calibration: The Achilles heel of predictive analytics,” BMC Med., vol. 17, p. 230, 2019.]]

#par[#text(size: 8pt)[\[20\] J. Priem, H. Piwowar, and R. Orr, “OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts,” arXiv:2205.01833, 2022.]]

#par[#text(size: 8pt)[\[21\] L. A. Inker et al., “New creatinine- and cystatin C-based equations to estimate GFR without race,” N. Engl. J. Med., vol. 385, pp. 1737–1749, 2021.]]

#par[#text(size: 8pt)[\[22\] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.]]

#par[#text(size: 8pt)[\[23\] S. Pölsterl, “scikit-survival: A library for time-to-event analysis built on top of scikit-learn,” J. Mach. Learn. Res., vol. 21, no. 212, pp. 1–6, 2020.]]

#par[#text(size: 8pt)[\[24\] C. Davidson-Pilon, “lifelines: Survival analysis in Python,” J. Open Source Softw., vol. 4, no. 40, p. 1317, 2019.]]
