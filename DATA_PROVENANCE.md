# Dataset provenance and reuse

## Heart Failure Clinical Records (299 patients)

The included CSV contains the same 299 rows, columns and parsed values as the UCI source, verified on 31 October 2026. Its bytes differ from the UCI download, so it should not be described as a byte-identical copy. Cite: Heart Failure Clinical Records [Dataset], UCI Machine Learning Repository, 2020, https://doi.org/10.24432/C5Z89R. UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Chicco & Jurman (2020) and the original Ahmad et al. (2017) study. This repository does not imply their endorsement.

## SUPPORT2

The working folder contains a local copy of the publicly available SUPPORT2 CSV at `data/support2.csv`. The source is the Vanderbilt Department of Biostatistics: https://hbiostat.org/data/ and https://hbiostat.org/data/repo/support2csv.zip. Vanderbilt permits dataset use and requests citation of the original paper and the following acknowledgement: “Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.” UCI points readers to the linked source for licensing: https://archive.ics.uci.edu/dataset/880/support2. Explicit redistribution terms for this copy have not been established, so the CSV is Git-ignored; use `python code/download_support.py` after cloning.

## Local 50,000-row cardiovascular file

`data/cardiovascular_hf_dataset-3.csv` is present in the prepared local folder but Git-ignored. A public [Hugging Face dataset card](https://huggingface.co/datasets/Auric-Grid/E1.L1-Cardiovascular-Heart-Failure-Risk-Prediction) gives the same filename, dimensions, and source counts; access to the data is gated and no byte-level match was established. The card states “synthetic or de-identified”, which does not establish which description applies. Its redistribution license and the origin of the local copy are not verified. Do not commit it until the owner confirms reuse and redistribution rights. The audit cannot establish provenance.

## Literature metadata

OpenAlex search results, screened study metadata and AI-assisted abstract coding are incomplete and require independent verification. Reconstructed full abstracts are excluded from a public commit because their reuse terms may differ. Do not add article PDFs or gated source material without permission.
