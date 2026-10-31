"""
Phase 1 - literature corpus for the meta-research arm (contribution C6).

Collects every work in OpenAlex that cites one of the dataset's source papers
(Chicco & Jurman 2020; Ahmad et al. 2017) or the audited paper (Ali et al. 2023),
plus works found by keyword search of titles/abstracts,
reconstructs abstracts, flags works that probably USE the 299-patient dataset,
and auto-extracts reported performance numbers and pipeline red flags from the
abstract. The output is a screening sheet: humans must confirm every row from
the full text before it enters the analysis.

Usage:  python 05_build_literature_corpus.py
Outputs: results/literature/corpus_all_citing.csv
         results/literature/screening_sheet.csv   (likely dataset users, sorted by reported accuracy)
"""
import json, os, re, time, urllib.parse, urllib.request
import pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(HERE, "results", "legacy", "literature")
SEEDS = {
    "W3012665221": "Chicco & Jurman 2020 (BMC MIDM)",
    "W2739315321": "Ahmad et al. 2017 (PLoS One)",
    "W4366831092": "Ali et al. 2023 (Healthcare Analytics)",
}
# Keyword searches catch papers that use the Kaggle/UCI copy without citing the source papers
SEARCHES = {
    'title_and_abstract.search:"heart failure clinical records"': "search: 'heart failure clinical records'",
    'title_and_abstract.search:"heart failure" "299 patients"': "search: 'heart failure' + '299 patients'",
    'title_and_abstract.search:"heart failure" "death event"': "search: 'heart failure' + 'death event'",
}
FIELDS = "id,doi,display_name,publication_year,type,primary_location,cited_by_count,abstract_inverted_index,authorships"

DATASET_TERMS = re.compile(r"\b299\b|heart failure clinical records|faisalabad|\bUCI\b|kaggle|death[ _]event|chicco", re.I)
CLINICAL_VARS = re.compile(r"serum creatinine|ejection fraction", re.I)
ML = re.compile(r"machine learning|deep learning|classif|random forest|xgboost|boosting|neural network|\bSVM\b|support vector|ensemble|"
                r"decision tree|logistic regression|naive bayes|k-nearest|\bKNN\b", re.I)
HF = re.compile(r"heart failure|\bHF\b|cardiac failure", re.I)
PCT = r"(\d{2}(?:\.\d+)?)\s?%"
ACC = re.compile(r"accura\w*[^.;%]{0,60}?" + PCT + r"|" + PCT + r"[^.;%]{0,25}accura\w*", re.I)
AUC = re.compile(r"(?:AUC|AUROC|ROC[- ]AUC|area under[^.;]{0,40}curve)[^.;]{0,40}?(0\.\d{2,4}|\d{2}(?:\.\d+)?\s?%)", re.I)
FLAGS = {
    "mentions_smote_or_oversampling": re.compile(r"SMOTE|oversampl|over-sampl|ADASYN|resampl", re.I),
    "mentions_time_or_followup": re.compile(r"follow[- ]?up (?:time|period|days)|\btime feature|\btime\b variable", re.I),
    "mentions_outlier_removal": re.compile(r"outlier", re.I),
    "mentions_augmentation_gan": re.compile(r"\bGAN\b|CTGAN|augment|synthetic data", re.I),
    "mentions_survival_methods": re.compile(r"\bCox\b|Kaplan|survival forest|C-index|concordance|censor", re.I),
    "mentions_cross_validation": re.compile(r"cross[- ]validation|k-fold|\bfold\b", re.I),
    "mentions_external_validation": re.compile(r"external(?:ly)? validat", re.I),
}


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def fetch(filt):
    # page-number paging (cursor paging needs an API key); fine for < 10,000 results
    rows, page = [], 1
    while True:
        q = urllib.parse.urlencode({"filter": filt, "per-page": 200, "page": page, "select": FIELDS})
        js = get(f"https://api.openalex.org/works?{q}")
        rows += js["results"]
        if not js["results"] or len(rows) >= js["meta"]["count"]:
            return rows
        page += 1
        time.sleep(0.2)


def nums(pattern, text):
    out = []
    for m in pattern.finditer(text):
        v = next(g for g in m.groups() if g)
        v = float(v.replace("%", "").strip())
        out.append(v / 100 if v > 1 else v)
    return sorted(set(round(x, 4) for x in out if 0.5 <= x <= 1.0))


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = {}
    sources = {f"cites:{k}": f"cites {v}" for k, v in SEEDS.items()} | {k: v for k, v in SEARCHES.items()}
    for filt, label in sources.items():
        works = fetch(filt)
        print(f"{label}: {len(works)} works")
        for w in works:
            r = recs.setdefault(w["id"], dict(
                openalex_id=w["id"], doi=w.get("doi"), title=w.get("display_name"), year=w.get("publication_year"),
                type=w.get("type"), venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                cited_by=w.get("cited_by_count"), n_authors=len(w.get("authorships") or []),
                abstract=abstract(w.get("abstract_inverted_index")), found_via=[]))
            r["found_via"].append(label.split(" (")[0])

    df = pd.DataFrame(recs.values())
    df = df[~df.openalex_id.isin(["https://openalex.org/" + k for k in SEEDS])]  # drop the seed papers themselves
    df["found_via"] = df.found_via.apply(lambda c: "; ".join(sorted(set(c))))
    text = (df.title.fillna("") + " . " + df.abstract.fillna(""))
    df["has_abstract"] = df.abstract.str.len() > 50
    df["is_hf"] = text.str.contains(HF)
    cites_source = df.found_via.str.contains("cites")
    # likely user = HF + ML + (>= 2017) + (dataset-specific term, or cites a source paper and names its key variables)
    df["likely_uses_dataset"] = (df.is_hf & text.str.contains(ML) & (df.year >= 2017)
                                 & (text.str.contains(DATASET_TERMS) | (cites_source & text.str.contains(CLINICAL_VARS))))
    df["reported_accuracy"] = text.apply(lambda t: nums(ACC, t))
    df["reported_auc"] = text.apply(lambda t: nums(AUC, t))
    df["max_reported_accuracy"] = df.reported_accuracy.apply(lambda v: max(v) if v else None)
    df["max_reported_auc"] = df.reported_auc.apply(lambda v: max(v) if v else None)
    for k, rx in FLAGS.items():
        df[k] = text.str.contains(rx)
    df = df.sort_values(["likely_uses_dataset", "max_reported_accuracy"], ascending=False)
    df.to_csv(os.path.join(OUT, "corpus_all_citing.csv"), index=False)

    # Screening sheet: columns the team fills from the FULL TEXT
    sheet = df[df.likely_uses_dataset].drop(columns=["abstract"]).copy()
    for col in ["SCREEN_include(Y/N)", "SCREEN_exclusion_reason", "FT_dataset_version(299/5000/other)", "FT_time_used_as_feature(Y/N/unclear)",
                "FT_resampling_before_split(Y/N/unclear/none)", "FT_outlier_removal_before_split(Y/N/none)", "FT_feature_selection_on_full_data(Y/N/none)",
                "FT_validation(single_split/CV/nested_CV/external)", "FT_test_size", "FT_best_model", "FT_reported_accuracy", "FT_reported_auc",
                "FT_censoring_handled(Y/N)", "FT_code_available(Y/N)", "FT_extractor_initials", "FT_notes"]:
        sheet[col] = ""
    sheet.to_csv(os.path.join(OUT, "screening_sheet.csv"), index=False)

    print(f"\nUnique citing works: {len(df)} | with abstract: {df.has_abstract.sum()} | heart-failure related: {df.is_hf.sum()} "
          f"| likely use the 299 dataset: {df.likely_uses_dataset.sum()}")
    u = df[df.likely_uses_dataset]
    acc = u.max_reported_accuracy.dropna()
    print(f"Likely users reporting an accuracy in the abstract: {len(acc)}; median {acc.median():.3f}; "
          f">=0.90: {(acc >= 0.90).sum()}; >=0.95: {(acc >= 0.95).sum()}; >=0.99: {(acc >= 0.99).sum()}")
    print("Abstract flags among likely users:", {k: int(u[k].sum()) for k in FLAGS})
    print("\nBy year:", u.year.value_counts().sort_index().to_dict())
    print("\nTop 15 by reported accuracy:")
    print(u.dropna(subset=["max_reported_accuracy"])[["year", "max_reported_accuracy", "venue", "title"]].head(15).to_string(index=False, max_colwidth=70))


if __name__ == "__main__":
    main()
