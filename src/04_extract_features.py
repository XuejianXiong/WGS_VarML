#!/usr/bin/env python3
"""
Feature Extraction for WGS Variant Pathogenicity Prediction.

This script transforms raw VEP-annotated VCF files into ML-ready feature matrices.
It specifically addresses the parsing of complex VEP strings (e.g., 'deleterious(0.01)')
and handles large-scale genomic data using memory-efficient Pandas dtypes.

Key Features:
    - Regex-based extraction of SIFT and PolyPhen scores.
    - Vectorized one-hot encoding of VEP Consequences and Impacts.
    - Label mapping for ClinVar significance (Pathogenic=1, Benign=0).
    - Parquet serialization with Snappy compression.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple

import numpy as np
import pandas as pd
from cyvcf2 import VCF
from logzero import logger, setup_logger

# Shared utilities fallback
try:
    from utils.config import load_config, resolve
except ImportError:
    def load_config(x): return {}
    def resolve(cli, cfg, default): return cli if cli is not None else (cfg if cfg is not None else default)

setup_logger()

class ExtractionParams(NamedTuple):
    input_vcf: Path
    output_path: Path
    output_format: str
    filter_unknown: bool

def parse_args() -> ExtractionParams:
    """Parses CLI arguments and resolves output paths."""
    parser = argparse.ArgumentParser(description="Professional WGS Feature Extractor")
    parser.add_argument("input_vcf", type=str, help="Path to VEP VCF (.vcf.gz)")
    parser.add_argument("output_file", nargs="?", help="Base path for output")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--filter-unknown", action="store_true", help="Remove clnsig_label -1")
    
    args = parser.parse_args()
    input_p = Path(args.input_vcf)
    
    if args.output_file:
        out_p = Path(args.output_file).with_suffix(f".{args.format}")
    else:
        out_p = input_p.parent / f"{input_p.stem.split('.')[0]}.features.{args.format}"
        
    return ExtractionParams(input_p, out_p, args.format, args.filter_unknown)

def parse_vep_score(value: str) -> Optional[float]:
    """
    Extracts numeric scores from VEP strings like 'deleterious(0.05)'.
    
    This is critical for resolving NaN issues in genomic feature matrices where
    scores are wrapped in qualitative descriptors.
    """
    if not value or value in ("", ".", "NA"):
        return None
    
    # Matches decimals inside parentheses or standalone numbers
    match = re.search(r"(\d+\.?\d*)", value)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def map_clnsig(clnsig: Any) -> int:
    """Maps ClinVar significance to binary ML labels."""
    if not clnsig:
        return -1
    
    # Handle list return from cyvcf2
    label = clnsig[0].lower() if isinstance(clnsig, (list, tuple)) else str(clnsig).lower()
    
    if any(term in label for term in ["pathogenic", "likely_pathogenic"]):
        return 1
    if any(term in label for term in ["benign", "likely_benign"]):
        return 0
    return -1

def extract_records(vcf_path: Path) -> pd.DataFrame:
    """Streams VCF and parses INFO/CSQ fields."""
    vcf = VCF(str(vcf_path))
    csq_header = vcf.get_header_type("CSQ")
    if not csq_header:
        logger.error("Missing CSQ header in VCF. Re-run VEP annotation.")
        sys.exit(1)
        
    fields = csq_header['Description'].split("Format: ")[1].strip().split("|")
    records = []

    logger.info(f"Processing variants from {vcf_path}...")
    for var in vcf:
        csq_raw = var.INFO.get("CSQ")
        csq_map = {}
        if csq_raw:
            # Use the first transcript annotation (standard for baseline models)
            csq_map = dict(zip(fields, csq_raw.split(",")[0].split("|")))

        records.append({
            "chr": var.CHROM,
            "pos": var.POS,
            "ref": var.REF,
            "alt": ",".join(var.ALT) if var.ALT else "",
            "impact": csq_map.get("IMPACT", "NA"),
            "consequence": csq_map.get("Consequence", ""),
            "sift": parse_vep_score(csq_map.get("SIFT", "")),
            "polyphen": parse_vep_score(csq_map.get("PolyPhen", "")),
            "distance": parse_vep_score(csq_map.get("DISTANCE", "")),
            "clnsig_label": map_clnsig(var.INFO.get("CLNSIG"))
        })
        
    return pd.DataFrame(records)

def optimize_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Performs vectorized one-hot encoding and optimizes memory usage."""
    logger.info("Encoding categorical features...")

    # 1. Consequence Encoding (Handles multiple terms joined by '&')
    cons_dummies = df['consequence'].str.get_dummies(sep='&').add_prefix('cons_')
    
    # 2. Impact Encoding
    impact_dummies = pd.get_dummies(df['impact'], prefix='impact')
    
    # 3. Merge and Clean
    df = pd.concat([df.drop(columns=['consequence', 'impact']), cons_dummies, impact_dummies], axis=1)
    
    # 4. Memory Optimization: Dummies to uint8
    for col in df.columns:
        if col.startswith(('cons_', 'impact_')):
            df[col] = df[col].astype(np.uint8)
            
    return df

def main():
    params = parse_args()
    
    # Load and Filter
    df = extract_records(params.input_vcf)
    
    if params.filter_unknown:
        before = len(df)
        df = df[df['clnsig_label'] != -1].reset_index(drop=True)
        logger.info(f"Filtered {before - len(df)} uncertain variants.")
        
    # Transform
    df = optimize_and_encode(df)
    
    # Save
    params.output_path.parent.mkdir(parents=True, exist_ok=True)
    if params.output_format == 'parquet':
        df.to_parquet(params.output_path, index=False, compression='snappy')
    else:
        df.to_csv(params.output_path, index=False)
        
    logger.info(f"Successfully exported {len(df)} variants to {params.output_path}")

if __name__ == "__main__":
    main()