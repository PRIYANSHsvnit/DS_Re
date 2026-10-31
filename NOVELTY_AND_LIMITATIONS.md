# Novelty check and limitations

Targeted search checked on 31 October 2026; it is not an exhaustive systematic review or a priority determination.

| Existing work | Relevance | Implication |
|---|---|---|
| [Rane et al., 2026](https://arxiv.org/abs/2608.04140) (preprint) | Same 299-patient cohort; explicitly quantifies follow-up-time leakage and studies survival modeling. | Time leakage and the survival framing are not new discoveries here. |
| [Naeem, Ahmad and Khan, 2026](https://thesesjournal.com/index.php/1/article/view/3986) | Compares the same cohort with and without follow-up time, including repeated nested CV. | Leakage-controlled evaluation and a time comparison are not novel on their own. |
| [Srujana et al.](https://doi.org/10.1080/03610918.2022.2060510) | Censoring-aware methods on this cohort. | Censoring-aware modeling itself is prior work. |
| [HeartGuard repository](https://github.com/NvyGreen/HeartGuard) | Public project reports an external UCI-to-MIMIC-IV evaluation and time-leakage analysis; claims have not been independently audited here. | Do not claim the first external validation of a UCI-trained model. |
| [Gibson et al., 2026](https://doi.org/10.64898/2026.02.24.26347028) (preprint) | Clinical dataset provenance concerns in other public datasets. | Plausibility checking is not a new general idea. |

Potentially useful contribution: this particular integrated case study uses a paired original-patient test comparison, traces held-out patients as SMOTE interpolation endpoints, and reports a fixed-model patient-bootstrap SUPPORT transport analysis. We have not established that every component or this combination is unique. The deliberately contaminated arm is not identical to a full-data-SMOTE-before-split implementation; test it as a stress test. The 2.5th–97.5th percentiles over splits are not population confidence intervals. The external bootstrap holds the fitted UCI model fixed and omits training-sample and between-hospital uncertainty. The original 50,000-row file's provenance and label definition remain uncertain. There was no Zigong evaluation or full-text systematic review.
