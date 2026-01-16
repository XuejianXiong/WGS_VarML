#!/usr/bin/env python
"""
04_extract_features.py

Extract ML-ready features from a VEP-annotated VCF file.

Features:
- One-hot encode multi-value Consequence
- One-hot encode Impact
- Numeric columns: SIFT, PolyPhen
- CLNSIG mapped to ML label (pathogenic=1, benign=0)
- Optionally filter variants with unknown CLNSIG (--filter-unknown)
- Explicit output format with --format csv|parquet
- Professional logging with logzero

Usage:
    python scripts/04_extract_features.py <input_vcf.gz> [output_file] [--filter-unknown] [--format csv|parquet]

Examples:
    python scripts/04_extract_features.py data/processed/clinvar.vep.vcf.gz
    python scripts/04_extract_features.py data/processed/clinvar.vep.vcf.gz data/features/clinvar_features.parquet --filter-unknown
    python scripts/04_extract_features.py data/processed/clinvar.vep.vcf.gz --format csv
"""

import sys
import os
from typing import Tuple
import pandas as pd
from cyvcf2 import VCF
from logzero import logger, setup_logger

# Configure logger
setup_logger()

# ----------------------------
# Argument parsing
# ----------------------------
def parse_args() -> Tuple[str, str, str, bool]:
    """
    Parse command-line arguments.

    Returns:
        input_vcf: str - Path to input VEP-annotated VCF (.vcf.gz)
        output_file: str - Path to save output features
        output_ext: str - File extension: "csv" or "parquet"
        filter_unknown: bool - Whether to remove variants with clnsig_label=-1
    """
    if len(sys.argv) < 2:
        logger.error("Missing input VCF argument.")
        print(__doc__)
        sys.exit(1)

    input_vcf = sys.argv[1]
    if not os.path.exists(input_vcf):
        logger.error(f"Input VCF does not exist: {input_vcf}")
        sys.exit(1)

    # Default output file
    output_file = None
    output_ext = "csv"  # default
    filter_unknown = False

    # Parse remaining arguments
    for arg in sys.argv[2:]:
        if arg == "--filter-unknown":
            filter_unknown = True
        elif arg.startswith("--format"):
            if "=" in arg:
                output_ext = arg.split("=")[1].lower()
            else:
                logger.error("Use --format=csv or --format=parquet")
                sys.exit(1)
            if output_ext not in ["csv", "parquet"]:
                logger.error("Invalid format. Use --format=csv or --format=parquet")
                sys.exit(1)
        else:
            # assume it's output file path
            output_file = arg

    if output_file is None:
        output_file = os.path.splitext(input_vcf)[0] + f"_features.{output_ext}"
    else:
        # override extension if format is specified
        output_file = os.path.splitext(output_file)[0] + f".{output_ext}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return input_vcf, output_file, output_ext, filter_unknown

# ----------------------------
# Map CLNSIG to ML label
# ----------------------------
def map_clnsig_to_label(clnsig: str | None) -> int:
    if clnsig is None:
        return -1
    clnsig_lower = clnsig.lower()
    if clnsig_lower in ["pathogenic", "likely_pathogenic"]:
        return 1
    elif clnsig_lower in ["benign", "likely_benign", "uncertain_significance"]:
        return 0
    else:
        return -1

# ----------------------------
# Read VCF and extract base fields
# ----------------------------
def read_vcf(input_vcf: str) -> pd.DataFrame:
    logger.info(f"Reading VCF: {input_vcf}")
    vcf = VCF(input_vcf)
    records = []

    for variant in vcf:
        info = variant.INFO
        consequences = info.get("Consequence")
        consequences_list = consequences.split(",") if consequences else []

        records.append({
            "chr": variant.CHROM,
            "pos": variant.POS,
            "ref": variant.REF,
            "alt": ",".join(variant.ALT),
            "impact": info.get("Impact") or "NA",
            "sift": float(info.get("SIFT")) if info.get("SIFT") else None,
            "polyphen": float(info.get("PolyPhen")) if info.get("PolyPhen") else None,
            "clnsig": info.get("CLNSIG") or "NA",
            "clnsig_label": map_clnsig_to_label(info.get("CLNSIG")),
            "consequence_list": consequences_list
        })

    return pd.DataFrame(records)

# ----------------------------
# One-hot encode Consequence and Impact
# ----------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("One-hot encoding 'Consequence'")
    all_consequences = set(c for row in df["consequence_list"] for c in row)
    for cons in all_consequences:
        df[f"cons_{cons}"] = df["consequence_list"].apply(lambda x: int(cons in x))
    df.drop(columns=["consequence_list"], inplace=True)

    logger.info("One-hot encoding 'Impact'")
    impact_dummies = pd.get_dummies(df["impact"], prefix="impact")
    df = pd.concat([df.drop(columns=["impact"]), impact_dummies], axis=1)

    return df

# ----------------------------
# Save features
# ----------------------------
def save_features(df: pd.DataFrame, output_file: str, output_ext: str) -> None:
    if output_ext == "csv":
        df.to_csv(output_file, index=False)
    else:
        df.to_parquet(output_file, index=False)
    logger.info(f"Features saved to {output_file}")
    logger.info(f"Preview:\n{df.head()}")

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    input_vcf, output_file, output_ext, filter_unknown = parse_args()
    df = read_vcf(input_vcf)
    df = encode_features(df)

    if filter_unknown:
        initial_count = len(df)
        df = df[df["clnsig_label"] != -1].reset_index(drop=True)
        logger.info(f"Filtered unknown CLNSIG variants: {initial_count - len(df)} removed, {len(df)} remaining")

    save_features(df, output_file, output_ext)

if __name__ == "__main__":
    main()
