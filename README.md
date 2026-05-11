# WGS_VarML
[![Python](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Nextflow](https://img.shields.io/badge/nextflow-%E2%89%A523.10.0-24bd5e.svg)](https://www.nextflow.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-stable-blue.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-stable-orange.svg)](https://lightgbm.readthedocs.io/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Model_Zoo-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### Genomic Variant Pathogenicity Prediction Pipeline

**WGS_VarML** is an end-to-end, production-grade computational pipeline designed to predict the clinical significance of genomic variants using machine learning. It integrates high-confidence labels from ClinVar with functional annotations from VEP, gnomAD, and CADD to train and deploy supervised classifiers at scale.


## Key Features

- **Scalable Orchestration**: Built with Nextflow DSL2 for seamless parallel execution across local, HPC (Slurm), or Cloud environments.

- **Portable Infrastructure**: Environment parity via Docker ensures "it works on my machine" is "it works everywhere."

- **Model Benchmarking**: Automated side-by-side comparison of XGBoost, LightGBM, Random Forest, and Logistic Regression.

- **Data Integrity**: Chromosome-aware stratified splitting to prevent data leakage between training and evaluation sets.

- **Artifact Management**: Bundle models with specific imputer.joblib and feature_order files for zero-skew inference.

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
├── main.nf                   - primary workflow entry point (DSL2)
├── nextflow.config           - resource management '&' Docker profiles
├── input.json                - pipeline runtime parameters
├── requirements.txt          - pinned python dependency lockfile
├── .gitignore
├── bin/                      – executable ML engines ('python' scripts)
│   ├── 04_extract_features   - extract ML-ready features from VEP-annotated VCF
│   ├── 05_QC_feature         - generate feature QC report
│   ├── 06_split_clinvar      - leakage-aware dataset partitioning
│   ├── 07_train_model        - distributed model training
│   ├── 08_model_inference    - high-throughput variant scoring
│   └── utils/
│         └── config.py       - load configuration parameters       
├── modules/                  - atomic Nextflow process definitions
├── docker/                   – lightweight Docker setup for reproducibility
├── config/                   
│   └── config.yaml           - YAML-based scientific hyper-parameters
├── scripts/                  – bash helpers for data ingestion/VEP annotation
│   ├── 01_download_data.sh             - download ClinVar and other reference datasets
│   ├── 02_preprocess_clinvar.sh        - normalize, filter, and prepare ClinVar VCF
│   └── 03_run_vep_docker.sh            - annotate variants with VEP inside Docker
├── data/                     – raw, reference, processed and splits datasets
├── results/                  – organized outputs: Models, Reports, Predictions
├── tests/                    – test code
└── LICENSE
```

## Quick Start

The entire pipeline is orchestrated via Nextflow. This handles data flow, parallelization, and ensures that the inference step uses the exact artifacts produced during training.

*Note*: This pipeline requires the VEP cache. Download and unzip the cache to a local directory and update config.yaml with the path.

### Prerequisites
- Nextflow (23.10.0+)
- Docker
- Python 3.9+

### Execution

The pipeline is designed to be executed from the project root. The Docker profile handles all dependencies automatically.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build docker image
docker build -t wgs-varml:latest -f docker/Dockerfile .

# 3. Run the full pipeline with Docker
nextflow run main.nf -profile docker -params-file input.json
```

### Outputs & Monitoring

After completion, detailed performance and resource metrics are available in:

- Execution Report: ```results/pipeline_info/execution_report.html```
- Dependency Graph: ```results/pipeline_info/pipeline_dag.html```
- Model Performance: ```results/models/```

## Pipeline Logic

### 1. Feature Extraction & QC

Transforms raw VEP-annotated VCFs into memory-efficient, Snappy-compressed Apache Parquet matrices. It handles one-hot encoding for complex Consequence/Impact strings and generates an automated HTML QC Report to validate feature distributions.

### 2. Machine Learning Engine

The ML engine trains four architectures simultaneously:

- Gradient Boosting: XGBoost & LightGBM

- Ensemble: Random Forest

- Baseline: Logistic Regression (L2 regularized)

### 3. High-Fidelity Inference

The inference module stages pre-trained artifacts (.joblib) alongside unlabeled genomic features. This ensures the inference environment perfectly matches the training environment, preventing feature-order mismatch errors.

## Status & Roadmap

✅ v2.0.0 Ready: DSL2 Orchestration, Docker Support and Parallel Training.

🚧 In Progress:

[ ] Optuna Integration: Automated hyperparameter optimization.

[ ] Explainability: SHAP-based feature importance modules.

[ ] Variant Prioritization: Integration of gene-level constraint scores (pLI/LOEUF).

## License

MIT License – feel free to use, adapt, and share.
