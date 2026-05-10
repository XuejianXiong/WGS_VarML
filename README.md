# WGS_VarML
### Genomic Variant Pathogenicity Prediction Pipeline

A scalable, reproducible Nextflow workflow for whole-genome sequencing (WGS) variant classification.

## Overview

WGS_VarML is an end-to-end computational pipeline designed to predict the clinical significance of genomic variants using machine learning. It integrates high-confidence labels from ClinVar with functional annotations from VEP, gnomAD, and CADD to train and deploy supervised classifiers.

### Key Features

 - Scalable Orchestration: Powered by Nextflow DSL2 for seamless parallel execution.
 - Robust Feature Engineering: automated parsing of complex VEP strings and one-hot encoding.
 - Model Benchmarking: Side-by-side comparison of XGBoost, LightGBM, Random Forest, and Logistic Regression.
 - Deterministic Environments: Fully locked dependencies via pip-compile and Docker support.

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
├── requirements.txt - pinned dependency lockfile
├── input.json      - input paramenters
├── nextflow.config - tesource management & reporting profiles
├── .gitignore
├── docker/         – lightweight Docker setup for reproducibility
├── data/           – raw and processed datasets (not tracked)
│   ├── raw/
│   ├── reference/
│   ├── splits/
│   └── processed/
├── nextflow/       - nextflow DSL2 orchestration
│   ├── main.nf                  - primary workflow entry point
│   └── ...                      - atomic process definitions (Train, Infer, QC)
├── src/            – core ML and data processing code
│   ├── 04_extract_features.py   - extract ML-ready features from VEP-annotated VCF
│   ├── 05_QC_feature.py         - generate feature QC report
│   ├── 06_split_clinvar.py      - split data into train / test / infer
│   ├── 07_train_model.py        - train ML model
│   ├── 08_model_inference.py    - run inference on unlabeled data
│   └── utils/
│         └── config.py          - load configuration parameters       
├── scripts/       – command-line entry points
│   ├── 01_download_data.sh             - download ClinVar and other reference datasets
│   ├── 02_preprocess_clinvar.sh        - normalize, filter, and prepare ClinVar VCF
│   └── 03_run_vep_docker.sh            - annotate variants with VEP inside Docker
├── config/
│   └── config.yaml - configuration file
├── notebooks/      – exploratory analysis and visualization
├── annotations/    – annotation configs
├── results/        – pipeline outputs (Predictions, Reports, Artifacts)
├── tests/          – test code
└── LICENSE
```

## Quick Start (Nextflow)

The entire pipeline is orchestrated via Nextflow. This handles data flow, parallelization, and ensures that the inference step uses the exact artifacts produced during training.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
nextflow run nextflow/main.nf -params-file input.json -resume
```

After completion, detailed performance and resource metrics are available in:
```bash
results/pipeline_info/execution_report.html

results/pipeline_info/pipeline_dag.html
```

## Pipeline Workflow

### 1. Feature Extraction & QC

Transforms raw VEP-annotated VCFs into memory-efficient Snappy-compressed Parquet matrices.

- Handles one-hot encoding for Consequence and Impact.

- Generates an automated HTML QC Report to validate feature distributions and missingness.

### 2. Machine Learning Engine

The pipeline benchmarks four architectures simultaneously:

- Baseline: Logistic Regression

- Ensemble: Random Forest, XGBoost, LightGBM

- Evaluation Strategy:

  - Chromosome-aware stratified splitting to prevent data leakage.

  - Metrics: ROC-AUC, PR-AUC, and Calibration analysis.

### 3. Artifact-Driven Inference

Pre-trained models are bundled with their specific imputer.joblib and feature_order.txt. This ensures zero-skew inference when scoring new, unlabeled variants.


## Reproducibility

- Docker Support

For production environments, use the provided Docker profile to ensure OS-level consistency:

```bash
nextflow run nextflow/main.nf -profile docker
```

- Requirements
  - Python 3.9+
  - Nextflow 23.10+
  - Docker (Optional)

## Status

✅ ClinVar preprocessing & VEP integration

✅ Nextflow DSL2 Orchestration

✅ Parallel Model Benchmarking (XGB/LGBM/RF/LR)

✅ Automated QC & Reporting

🚧 Hyperparameter optimization (Optuna integration)

🚧 SHAP-based feature interpretability modules

