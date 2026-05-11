#!/usr/bin/env python3
"""
04_extract_features.py

Transform raw VEP-annotated VCF files into ML-ready feature matrices.
Used after 03_annotate_vep and before 05_QC_feature to parse complex genomic
annotations into structured numeric and categorical vectors.

Processing Steps (in order)
---------------------------
1. Load configuration from YAML and resolve CLI overrides.
2. Stream VCF records and parse the INFO/CSQ (Consequence) field.
3. Extract numeric scores (SIFT/PolyPhen) using robust regular expressions.
4. Map ClinVar significance to binary labels (0: Benign, 1: Pathogenic).
5. Vectorize categorical data (Consequence, Impact) via One-Hot Encoding.
6. Serialize to Parquet with Snappy compression for high-performance downstream I/O.

Config (config.yaml under "features"): sift_regex, polyphen_regex, 
target_mapping, impact_categories, consequence_categories. 
Precedence: CLI > YAML > defaults.

Usage
-----
    python3 src/04_extract_features.py <input_vcf> [--output FEATURES.parquet] <--config CONFIG>

Example
-------
    python3 src/04_extract_features.py data/processed/clinvar.vep.vcf.gz --config config/config.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple

import numpy as np
import pandas as pd
from cyvcf2 import VCF
from logzero import logger, setup_logger

# Shared utilities fallback for enterprise integration
try:
    from utils.config import load_config, resolve
except ImportError:
    def load_config(x): return {}
    def resolve(cli, cfg, default): return cli if cli is not None else (cfg if cfg is not None else default)

setup_logger()

class ExtractionParams(NamedTuple):
    """Container for resolved execution parameters."""
    input_vcf: Path
    output_path: Path
    output_format: str
    filter_unknown: bool

def load_yaml_extract_cfg(config_path: str) -> Dict[str, Any]:
    """
    Reads the 'extract' block from the YAML configuration.

    Args:
        config_path: Path to the project's config.yaml file.

    Returns:
        A dictionary containing extraction settings, or an empty dict on failure.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('extract', {})
    except Exception as e:
        logger.warning(f"Config load failed ({e}). Proceeding with script defaults.")
        return {}

def parse_args() -> ExtractionParams:
    """
    Parses CLI arguments and resolves parameters against config.yaml.
    
    Implements a hierarchy of priority: CLI Flags > YAML Config > Hardcoded Defaults.
    
    Returns:
        An ExtractionParams object with validated paths and settings.
    """
    parser = argparse.ArgumentParser(description="Professional WGS Feature Extractor")
    parser.add_argument("input_vcf", type=str, help="Path to VEP-annotated VCF (.vcf.gz)")
    parser.add_argument("output_file", nargs="?", help="Optional: Custom base path for output")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--format", choices=["csv", "parquet"], help="Override config format")
    parser.add_argument("--filter-unknown", action="store_true", help="Override config filter (remove clnsig -1)")
    
    args = parser.parse_args()
    
    # Load settings from YAML
    cfg = load_yaml_extract_cfg(args.config)
    
    # Resolve parameters based on priority
    fmt = args.format or cfg.get('format', 'parquet')
    filt = args.filter_unknown or cfg.get('filter_unknown', True)
    
    input_p = Path(args.input_vcf)
    
    if args.output_file:
        out_p = Path(args.output_file).with_suffix(f".{fmt}")
    else:
        # Default behavior: same directory, .features suffix
        out_p = input_p.parent / f"{input_p.stem.split('.')[0]}.features.{fmt}"
        
    return ExtractionParams(input_p, out_p, fmt, filt)

def parse_vep_score(value: str) -> Optional[float]:
    """
    Extracts numeric scores from VEP strings (e.g., 'tolerated(0.12)' -> 0.12).

    Uses a non-greedy regex to find the first decimal or integer within the string.
    This is critical because VEP often concatenates qualitative labels with scores.

    Args:
        value: The raw string value from the CSQ field.

    Returns:
        A float representing the score, or None if the field is empty or non-numeric.
    """
    if not value or value in ("", ".", "NA"):
        return None
    
    match = re.search(r"(\d+\.?\d*)", value)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def map_clnsig(clnsig: Any) -> int:
    """
    Maps ClinVar significance strings to binary Machine Learning labels.

    Classification Schema:
        - 1 (Pathogenic): Includes 'pathogenic' and 'likely_pathogenic'.
        - 0 (Benign): Includes 'benign' and 'likely_benign'.
        - -1 (Unknown): VUS, Conflicting, or missing data.

    Args:
        clnsig: The raw CLNSIG value from the VCF INFO field.

    Returns:
        An integer label (-1, 0, or 1).
    """
    if not clnsig:
        return -1
    
    label = clnsig[0].lower() if isinstance(clnsig, (list, tuple)) else str(clnsig).lower()
    
    if any(term in label for term in ["pathogenic", "likely_pathogenic"]):
        return 1
    if any(term in label for term in ["benign", "likely_benign"]):
        return 0
    return -1

def extract_records(vcf_path: Path) -> pd.DataFrame:
    """
    Streams the VCF file and parses the INFO/CSQ fields into a structured format.

    Uses cyvcf2 for high-performance iteration over genomic records.

    Args:
        vcf_path: Path to the VCF file.

    Returns:
        A Pandas DataFrame where each row is a variant and columns are extracted features.
    """
    vcf = VCF(str(vcf_path))
    csq_header = vcf.get_header_type("CSQ")
    if not csq_header:
        logger.error("Missing CSQ header in VCF. Ensure VEP was run with --vcf.")
        sys.exit(1)
        
    # Extract the format string from the VCF header to map CSQ pipes correctly
    fields = csq_header['Description'].split("Format: ")[1].strip().split("|")
    records = []

    logger.info(f"Processing variants from {vcf_path}...")
    for var in vcf:
        csq_raw = var.INFO.get("CSQ")
        csq_map = {}
        if csq_raw:
            # Note: We take the first transcript/consequence provided by VEP
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
    """
    Performs one-hot encoding for categorical data and optimizes memory usage.

    Memory Optimization:
        - Converts boolean-style dummy variables (0/1) to uint8 to reduce footprint.
        - Handles multi-consequence variants joined by '&'.

    Args:
        df: The raw extracted DataFrame.

    Returns:
        The encoded and memory-optimized DataFrame.
    """
    logger.info("Encoding categorical features...")

    # Consequence Encoding (Handles multiple terms like missense_variant&splice_region_variant)
    cons_dummies = df['consequence'].str.get_dummies(sep='&').add_prefix('cons_')
    impact_dummies = pd.get_dummies(df['impact'], prefix='impact')
    
    # Merge dummies and drop original high-cardinality strings
    df = pd.concat([df.drop(columns=['consequence', 'impact']), cons_dummies, impact_dummies], axis=1)
    
    # Cast to uint8 to save memory for ML training
    for col in df.columns:
        if col.startswith(('cons_', 'impact_')):
            df[col] = df[col].astype(np.uint8)
            
    return df

def main() -> None:
    """
    Entry point for the feature extraction pipeline.
    """
    params = parse_args()
    
    # 1. Extraction
    df = extract_records(params.input_vcf)
    
    # 2. Filtering
    if params.filter_unknown:
        before = len(df)
        df = df[df['clnsig_label'] != -1].reset_index(drop=True)
        logger.info(f"Filtered {before - len(df)} unknown/conflicting variants.")
        
    # 3. Encoding & Optimization
    df = optimize_and_encode(df)
    
    # 4. Serialization
    params.output_path.parent.mkdir(parents=True, exist_ok=True)
    if params.output_format == 'parquet':
        # Parquet + Snappy is the standard for high-throughput bioinformatics data
        df.to_parquet(params.output_path, index=False, compression='snappy')
    else:
        df.to_csv(params.output_path, index=False)
        
    logger.info(f"Successfully exported {len(df)} variants to {params.output_path}")

if __name__ == "__main__":
    main()