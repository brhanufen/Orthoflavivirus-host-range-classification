# Genome-Scale Classification of *Orthoflavivirus* (Flavivirus) Host Range

Data and reproducibility materials for the manuscript:
**"Genome-Scale Classification of Flavivirus Host Range Using Composition and Codon-Usage Signatures"** (Znabu & Sexton, 2026).

This repository contains the curated datasets, extracted feature matrices, per-analysis
result tables, figure source data, raw expanded genome sequences, a self-contained
reproduction script for the expanded-dataset analysis, and the figure-generation scripts.

> **Scope note (read first).** This deposit provides the **data, result tables, figure
> source data, and the expanded-dataset reproduction script**. It does **not** include the
> original end-to-end pipeline for (a) retrieving/QC-filtering genomes from NCBI or
> (b) extracting the 97 features from raw sequences, nor a single script that regenerates
> the *primary* ablation/CV/LOSO/genus result tables from `X_features_v1.csv`. Those steps
> are described in the manuscript Methods and in `expansion_README` (prose). The deposited
> feature matrices let a reader reproduce the modelling steps from features onward; the
> `code/retrain_v3_reproduce.py` script reproduces the expanded held-out and LOSO results.

## Layout

```
data/        Curated tables + extracted features + all result CSVs (primary and expanded)
genomes/     Raw FASTA for the newly added expanded genomes
code/        retrain_v3_reproduce.py  (expanded held-out + LOSO reproduction)
code/figure_scripts/   Figure-generation scripts + figure_data/ (per-panel inputs)
requirements.txt       Pinned software versions (scikit-learn 1.3.0, etc.)
```

## Datasets

| File | Description | Records |
|------|-------------|---------|
| `data/model_table_Flavivirus_2class_v1.csv` | Curated primary dataset + host-range labels (+ `accession`) | 1,285 genomes, 53 species |
| `data/X_features_v1.csv` | Primary 97-feature matrix + labels + train/val/test split | 1,285 rows |
| `data/X_features_expanded_v3.csv` | Expanded 97-feature matrix + labels + split | 1,528 rows |
| `data/Flaviviridae_4genus_index_v1.csv` / `..._features.csv` | Genus-level index (+ `accession`) and features | 3,458 genomes |
| `data/isfv_host_range_curation.csv`, `data/dual_host_curation.csv` | Expanded host-range curation / rationale | — |
| `data/accession_metadata_expanded.csv`, `data/EXPANSION_accessions_to_verify.csv` | Expanded accession metadata and list | — |

One record (`M12294.2`, West Nile virus) has all 97 feature fields empty in both feature
tables; it is excluded from the analysed set (1,284 primary / 1,527 expanded populated). The
reproduction script drops it automatically.

## Result tables

Primary: `ablation_results_v2.csv`, `length_control_results_v2.csv`, `cv_stability_results_v2.csv`,
`cv_statistical_tests_v3.csv`, `loso_cv_isfv_species_v1.csv`, `per_species_predictions_v1.csv`,
`misclassified_feature_comparison_v1.csv`, `feature_statistics_v2.csv`, `statistical_tests_v2.csv`,
`lr_coefficients_full_v2.csv`, `rf_importance_full_v2.csv`, `top15_features_comparison_v2.csv`,
`genus_cv_results.csv`, `genus_per_class_metrics.csv`, `genus_feature_importance.csv`.
Expanded: `baseline_comparison.csv`, `loso_per_species_CI.csv`, `loso_genome_clustered_bootstrap.csv`,
`fitzroy_sensitivity_analysis.csv`, `qc_report.csv`.

## Reproducing the expanded analysis

```bash
pip install -r requirements.txt
cd code
cp ../data/X_features_expanded_v3.csv .
python retrain_v3_reproduce.py
```
Reproduces the expanded held-out test metrics and leave-one-species-out CV (Fig S2). It reads
only the feature matrix — no network access or external downloads are required.

## Figures

`code/figure_scripts/` holds the plotting scripts. **Fig 4, 6, 7** have their per-panel inputs
in `figure_scripts/figure_data/` and run as-is. **Fig 1, 2, 3** scripts are included for
transparency but their per-panel input CSVs are not part of this deposit (see
`DEPOSIT_GAPS_AND_TODO.md`); a Fig 5 script is not included.

## GenBank accessions
All sequences are from NCBI GenBank. Accessions are in `data/model_table_Flavivirus_2class_v1.csv`,
`data/Flaviviridae_4genus_index_v1.csv`, and the expanded accession/metadata files, permitting
exact re-retrieval from the NCBI Nucleotide database.

## Software versions
See `requirements.txt` (Python 3.10; Biopython 1.81; scikit-learn 1.3.0; SciPy 1.11.0;
pandas 2.0.0; NumPy 1.24.0; Matplotlib 3.7.0; seaborn 0.12.0).

## License
Data for academic research use. GenBank sequences are subject to NCBI data-usage policies.
