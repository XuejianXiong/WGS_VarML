# WGS_VarML

Machine learning–based prediction of genomic variant pathogenicity
using whole-genome sequencing (WGS) data.

## Overview

WGS_VarML is an end-to-end pipeline for predicting the pathogenicity of genomic variants.

 - High-confidence clinical labels are sourced from ClinVar.
 - Variants are annotated with functional, population, and conservation information using tools such as VEP, CADD, and allele frequency databases like gnomAD.
 - The pipeline produces ML-ready features and trains supervised models for binary classification (pathogenic vs benign).

The project demonstrates a fully reproducible workflow from raw WGS/VCF data to ML-ready features and model evaluation.

## Data Sources

- **ClinVar**: clinical significance labels (pathogenic / benign)
- **gnomAD**: population allele frequency features
- **VEP**: functional consequence annotations
- **CADD & conservation scores**: deleteriousness and evolutionary constraint

Only publicly available datasets are used.

## Repository Structure

```bash
WGS_VarML/
├── README.md
├── requirement.txt
├── .gitignore
│
├── docker/         – lightweight Docker setup for reproducibility
│   └── Dockerfile
│
├── data/           – raw and processed datasets (not tracked)
│   ├── raw/
│   ├── reference/
│   └── processed/
│
├── src/            – core ML and data processing code
│   ├── 04_feature_extraction.py        - extract ML-ready features from VEP-annotated VCF
│   ├── 05_QC_features.py               - generate feature QC report
│   └── utils/
│
├── scripts/       – command-line entry points
│   ├── 01_download_data.sh             - download ClinVar and other reference datasets
│   ├── 02_preprocess_clinvar.sh        - normalize, filter, and prepare ClinVar VCF
│   └── 03_run_vep_docker.sh            - annotate variants with VEP inside Docker
│
├── config/         – configuration files
├── results/        – output results
├── notebooks/      – exploratory analysis and visualization
├── annotations/    – annotation configs
└── tests/          – test code
```

## Feature Extraction

The 04_feature_extraction.py script performs:

- One-hot encoding of multi-value Consequence and single-value Impact
- Retains numeric scores such as SIFT and PolyPhen
- Maps CLNSIG to numeric ML labels (1=pathogenic, 0=benign, -1=unknown)
- Optionally filters unknown variants with --filter-unknown
- Saves output as CSV or Parquet for downstream ML

```bash
# Default CSV output
python scripts/04_feature_extraction.py data/processed/clinvar.vep.vcf.gz

# Filter unknown labels and save as Parquet
python scripts/04_feature_extraction.py data/processed/clinvar.vep.vcf.gz --filter-unknown --format=parquet

# Custom output filename
python scripts/04_feature_extraction.py data/processed/clinvar.vep.vcf.gz data/features/clinvar_ml_ready.csv --filter-unknown
```

## Machine Learning Tasks

- Binary classification (pathogenic vs benign)
- Class imbalance handling
- Chromosome-aware train/validation/test splits

## Models

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)

## Evaluation Metrics

- ROC-AUC
- Precision–Recall AUC
- Confusion matrix
- Calibration analysis

## Reproducibility (Docker)

A lightweight Docker environment is provided for reproducible development:

```bash
# Build Docker image
docker build -t wgs-varml .

# Run interactively
docker run -it -v "$PWD/data:/opt/data" wgs-varml
```

Large datasets and annotation tools (like VEP caches) are mounted at runtime for flexibility.

## Status

🚧 In progress:

- ClinVar preprocessing and normalization
- VEP annotation pipeline integration
- Feature extraction with ML-ready encoding
- Baseline model implementation and evaluation


