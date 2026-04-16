#!/usr/bin/env python3
"""
Feature Extraction Module for WGS Variant Pathogenicity Prediction.

This module parses VEP-annotated VCF files, extracts functional consequences, 
impact scores (SIFT/PolyPhen), and clinical significance (CLNSIG), and 
transforms them into a structured feature matrix suitable for XGBoost/PyTorch.

Architecture:
    1. Stream-parse VCF using cyvcf2 for high performance.
    2. Atomic decomposition of multi-value consequences (e.g., missense&splice).
    3. Vectorized one-hot encoding of categorical variables.
    4. Serialization to Parquet (recommended) or CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple, Optional, List, Dict, Any

import pandas as pd
import numpy as np
from cyvcf2 import VCF
from logzero import logger, setup_logger

# Assuming utils.config exists in your project structure
try:
    from utils.config import load_config, resolve
except ImportError:
    # Fallback for standalone testing
    def load_config(x): return {}
    def resolve(cli, cfg, default): return cli if cli is not None else (cfg if cfg is not None else default)

setup_logger()

class ExtractionConfig(NamedTuple):
    """Container for validated extraction parameters."""
    input_vcf: Path
    output_path: Path
    output_format: str
    filter_unknown: bool
    config_path: Optional[Path]

def parse_args() -> ExtractionConfig:
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(description="WGS Feature Extractor (Production)")
    parser.add_argument("input_vcf", type=str, help="Path to VEP-annotated VCF (.vcf.gz)")
    parser.add_argument("output_file", nargs="?", help="Output base path")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet", help="Output format")
    parser.add_argument("--filter-unknown", action="store_true", help="Drop variants with clnsig_label -1")
    parser.add_argument("--config", type=str, help="Path to YAML config")

    args = parser.parse_args()
    
    # Logic for resolving output path
    fmt = args.format
    input_path = Path(args.input_vcf)
    if args.output_file:
        out_path = Path(args.output_file).with_suffix(f".{fmt}")
    else:
        out_path = input_path.parent / f"{input_path.stem.split('.')[0]}.features.{fmt}"

    return ExtractionConfig(
        input_vcf=input_path,
        output_path=out_path,
        output_format=fmt,
        filter_unknown=args.filter_unknown,
        config_path=Path(args.config) if args.config else None
    )

def map_clnsig_to_label(clnsig: Optional[str]) -> int:
    """
    Maps ClinVar significance strings to integer labels.
    
    Args:
        clnsig: The raw CLNSIG string from VCF INFO field.
        
    Returns:
        1 for Pathogenic, 0 for Benign, -1 for Uncertain/Unknown.
    """
    if not clnsig:
        return -1
    
    clnsig = clnsig.lower()
    # Using set lookup for O(1) performance
    pathogenic = {"pathogenic", "likely_pathogenic", "pathogenic/likely_pathogenic"}
    benign = {"benign", "likely_benign", "benign/likely_benign"}
    
    if clnsig in pathogenic: return 1
    if clnsig in benign: return 0
    return -1

def extract_vcf_records(vcf_path: str) -> pd.DataFrame:
    """
    Parses VCF and extracts raw features into a Pandas DataFrame.
    
    Optimized to handle large-scale VCFs by minimizing object overhead during 
    the iteration process.
    """
    vcf = VCF(vcf_path)
    
    # Retrieve CSQ headers from VCF metadata
    csq_header = vcf.get_header_type("CSQ")
    if not csq_header:
        logger.error("VCF missing CSQ header. Ensure VEP was run correctly.")
        sys.exit(1)
        
    csq_fields = csq_header['Description'].split("Format: ")[1].strip().split("|")
    
    data: List[Dict[str, Any]] = []
    
    logger.info("Starting VCF stream parsing...")
    for var in vcf:
        # Extract CSQ (taking first transcript only for simplicity in this version)
        csq_raw = var.INFO.get("CSQ")
        csq_data = {}
        if csq_raw:
            first_allele = csq_raw.split(",")[0]
            csq_data = dict(zip(csq_fields, first_allele.split("|")))

        # Map variant info
        clnsig = var.INFO.get("CLNSIG")
        if isinstance(clnsig, (list, tuple)):
            clnsig = clnsig[0]

        record = {
            "chr": var.CHROM,
            "pos": var.POS,
            "ref": var.REF,
            "alt": ",".join(var.ALT) if var.ALT else "",
            "impact": csq_data.get("IMPACT", "NA"),
            "consequence": csq_data.get("Consequence", ""),
            "sift": pd.to_numeric(csq_data.get("SIFT", "").split("(")[0], errors='coerce'),
            "polyphen": pd.to_numeric(csq_data.get("PolyPhen", "").split("(")[0], errors='coerce'),
            "clnsig_label": map_clnsig_to_label(clnsig)
        }
        data.append(record)
        
    return pd.DataFrame(data)

def encode_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs one-hot encoding and data type optimization.
    """
    logger.info(f"Encoding features for {len(df)} variants...")
    
    # 1. Handle Multi-value Consequences (Atomic Splitting)
    # Explode '&' and ',' separated terms
    df_cons = df['consequence'].str.get_dummies(sep='&')
    df_cons = df_cons.add_prefix('cons_')
    
    # 2. One-hot encode IMPACT (Categorical optimization)
    df_impact = pd.get_dummies(df['impact'], prefix='impact')
    
    # 3. Concatenate and Drop raw strings
    df = pd.concat([df, df_cons, df_impact], axis=1)
    df.drop(columns=['consequence', 'impact'], inplace=True)
    
    # 4. Memory optimization: Convert uint8 for dummies
    for col in df.columns:
        if col.startswith(('cons_', 'impact_')):
            df[col] = df[col].astype(np.uint8)
            
    return df

def main():
    config = parse_args()
    
    # Process
    df = extract_vcf_records(str(config.input_vcf))
    
    if config.filter_unknown:
        initial_count = len(df)
        df = df[df['clnsig_label'] != -1].reset_index(drop=True)
        logger.info(f"Filtered {initial_count - len(df)} unknown variants.")
    
    df = encode_and_clean(df)
    
    # Output
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.output_format == 'parquet':
        df.to_parquet(config.output_path, index=False, compression='snappy')
    else:
        df.to_csv(config.output_path, index=False)
        
    logger.info(f"Successfully saved features to {config.output_path}")

if __name__ == "__main__":
    main()