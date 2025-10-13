# WGS_VarML: Machine Learning–Driven Variant Quality Control

**WGS_VarML**: Machine learning–based variant quality control and prioritization module for whole-genome sequencing pipelines.

## 🚀 Features
- Extracts QC and annotation features from VCFs
- Benchmarks variant calls against GIAB truth sets
- Trains ML models (Random Forest, XGBoost, Autoencoder)
- Produces interpretable variant-level QC and ranking
- Generates reports and performance metrics

## 🧱 Repository Structure
WGS_VarML/
├── README.md
├── requirements.txt
│
├── config/
│   ├── paths.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/            # VCFs, truth sets, etc.
│   ├── processed/      # Extracted features
│   └── reference/      # BEDs, reference genomes
│
├── src/
│   ├── __init__.py
│   ├── download_giab.py       # Download and index GIAB truth VCFs
│   ├── feature_extraction.py  # Parse VCFs → tabular features
│   ├── labeling.py            # Label variants as TP/FP via truth set
│   ├── train_model.py         # Train RF/XGBoost/autoencoder models
│   ├── evaluate_model.py      # Evaluate metrics, visualize feature importance
│   ├── filter_variants.py     # Apply trained model to new samples
│   └── utils.py               # Helper functions (VCF parsing, logging)
│
├── notebooks/
│   ├── feature_analysis.ipynb
│   ├── model_comparison.ipynb
│   └── performance_report.ipynb
│
├── models/
│   ├── rf_model.joblib
│   └── xgb_model.joblib
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reports/
│
├── scripts/
│   ├── extract_features.sh
│   ├── train_model.sh
│   └── evaluate_model.sh
│
└── tests/
    ├── test_feature_extraction.py
    ├── test_labeling.py
    └── test_train_model.py

See the `/src`, `/data`, `/models`, and `/results` directories for main components.

## ⚙️ Setup
```bash
git clone https://github.com/<your_username>/VarML.git
cd VarML
pip install -r requirements.txt
```

## ▶️ Example Usage
```bash
bash scripts/extract_features.sh
bash scripts/train_model.sh
bash scripts/evaluate_model.sh
```

## 🧬 Reference Data

Uses Genome in a Bottle (GIAB) high-confidence call sets as truth data.

## 📊 Future Work

Integration with VEP annotations

Deep learning variant classifiers

Integration into Nextflow/WDL for automation