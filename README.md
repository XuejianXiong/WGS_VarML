# WGS_VarML

Machine learning–based prediction of genomic variant pathogenicity
using whole-genome sequencing (WGS) data.

## Overview

This project builds a supervised machine learning pipeline to predict
whether genomic variants are pathogenic or benign. High-confidence
clinical labels are obtained from ClinVar, while variants are enriched
with functional, population, and conservation annotations.

The goal is to demonstrate an end-to-end ML workflow that combines
bioinformatics annotation with robust model training and evaluation.

## Data Sources

- **ClinVar**: clinical significance labels (pathogenic / benign)
- **gnomAD**: population allele frequency features
- **VEP**: functional consequence annotations
- **CADD & conservation scores**: deleteriousness and evolutionary constraint

Only publicly available datasets are used.

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

## Repository Structure

- `data/` – raw and processed datasets (not tracked)
- `src/` – core ML and data processing code
- `notebooks/` – exploratory analysis and visualization
- `scripts/` – command-line entry points
- `docker/` – lightweight Docker setup for reproducibility

## Reproducibility (Docker)

A minimal Docker environment is provided for reproducible development
and training.

```bash
docker build -t wgs-varml .
docker run -it wgs-varml
```

The Docker setup is intentionally lightweight and focused on development.
Large datasets and external annotation tools are not baked into the image.

## Status

🚧 In progress:

- ClinVar preprocessing and filtering
- Feature schema design
- Baseline model implementation


