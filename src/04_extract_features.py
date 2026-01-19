#!/usr/bin/env python3
"""
04_extract_features.py

Extract ML-ready features from a VEP-annotated VCF file.

Features:
- One-hot encode multi-value Consequence
- One-hot encode Impact
- Numeric columns: SIFT, PolyPhen
- CLNSIG mapped to ML label (pathogenic=1, benign=0)
- Optional filtering of unknown CLNSIG
- Output as CSV or Parquet
- Professional logging

Usage:
    python scripts/04_extract_features.py <input_vcf.gz> [output_file]
        [--format csv|parquet]
        [--filter-unknown]

Examples:
    python scripts/04_extract_features.py data/processed/clinvar.vep.vcf.gz
    python scripts/04_extract_features.py data/processed/clinvar.vep.vcf.gz \
        data/features/clinvar_features.parquet --format parquet --filter-unknown
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple, Optional, List

import pandas as pd
from cyvcf2 import VCF
from logzero import logger, setup_logger

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
setup_logger()


# ------------------------------------------------------------------------------
# Argument container
# ------------------------------------------------------------------------------
class Args(NamedTuple):
    input_vcf: str
    output_file: str
    output_format: str
    filter_unknown: bool


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
def parse_args() -> Args:
    """
    Parse command-line arguments.

    Returns
    -------
    Args
        Parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract ML-ready features from a VEP-annotated VCF file."
    )

    parser.add_argument(
        "input_vcf",
        type=str,
        help="Path to input VEP-annotated VCF (.vcf.gz)",
    )

    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Output feature file (csv or parquet)",
    )

    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )

    parser.add_argument(
        "--filter-unknown",
        action="store_true",
        help="Remove variants with unknown CLNSIG labels",
    )

    args = parser.parse_args()

    input_vcf = Path(args.input_vcf)
    if not input_vcf.exists():
        parser.error(f"Input VCF does not exist: {input_vcf}")

    if args.output_file is None:
        output_file = (
            input_vcf
            .with_suffix("")
            .with_suffix(f".features.{args.format}")
        )
    else:
        output_file = Path(args.output_file).with_suffix(f".{args.format}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    parsed = Args(
        input_vcf=str(input_vcf),
        output_file=str(output_file),
        output_format=args.format,
        filter_unknown=args.filter_unknown,
    )

    logger.info(
        "Arguments parsed | "
        f"input_vcf={parsed.input_vcf}, "
        f"output_file={parsed.output_file}, "
        f"format={parsed.output_format}, "
        f"filter_unknown={parsed.filter_unknown}"
    )

    return parsed


# ------------------------------------------------------------------------------
# CLNSIG → ML label mapping
# ------------------------------------------------------------------------------
def map_clnsig_to_label(clnsig: Optional[str]) -> int:
    """
    Map ClinVar CLNSIG to ML label.

    Returns
    -------
    int
        1 = pathogenic
        0 = benign
       -1 = unknown / excluded
    """
    if not clnsig:
        return -1

    clnsig = clnsig.lower()

    if clnsig in {"pathogenic", "likely_pathogenic"}:
        return 1
    if clnsig in {"benign", "likely_benign", "uncertain_significance"}:
        return 0

    return -1


# ------------------------------------------------------------------------------
# Read VCF
# ------------------------------------------------------------------------------
def read_vcf(input_vcf: str) -> pd.DataFrame:
    """
    Read a VEP-annotated VCF and extract features from the CSQ field.

    Parameters
    ----------
    input_vcf : str
        Path to VEP-annotated VCF (.vcf.gz)

    Returns
    -------
    pd.DataFrame
        Raw feature table
    """
    logger.info(f"Reading VCF: {input_vcf}")
    vcf = VCF(input_vcf)

    # Extract CSQ field definition safely (cyvcf2-style)
    csq_info = vcf.get_header_type("CSQ")
    if csq_info is None:
        raise RuntimeError("CSQ INFO field not found in VCF header")

    csq_format = csq_info["Description"].split("Format: ")[1]
    csq_fields = csq_format.split("|")

    records: List[dict] = []

    for var in vcf:
        csq_raw = var.INFO.get("CSQ")

        if csq_raw:
            # Use first transcript annotation (standard ML practice)
            csq_entry = csq_raw.split(",")[0]
            csq_values = csq_entry.split("|")
            csq = dict(zip(csq_fields, csq_values))
        else:
            csq = {}

        consequence_list = (
            csq.get("Consequence", "").split(",")
            if csq.get("Consequence")
            else []
        )

        impact = csq.get("Impact", "NA")
        sift = float(csq["SIFT"]) if csq.get("SIFT") not in (None, "") else None
        polyphen = float(csq["PolyPhen"]) if csq.get("PolyPhen") not in (None, "") else None
        clnsig = csq.get("CLNSIG", "NA")

        records.append({
            "chr": var.CHROM,
            "pos": var.POS,
            "ref": var.REF,
            "alt": ",".join(var.ALT),
            "impact": impact,
            "sift": sift,
            "polyphen": polyphen,
            "clnsig": clnsig,
            "clnsig_label": map_clnsig_to_label(clnsig),
            "consequence_list": consequence_list,
        })

    df = pd.DataFrame.from_records(records)
    logger.info(f"Parsed {len(df)} variants from VCF")

    return df


# ------------------------------------------------------------------------------
# Feature encoding
# ------------------------------------------------------------------------------
def _split_atomic_consequences(cons_list: List[str]) -> List[str]:
    """
    Split compound VEP consequences into atomic Sequence Ontology terms.

    Example
    -------
    ["frameshift_variant&splice_region_variant"]
    -> ["frameshift_variant", "splice_region_variant"]
    """
    if not cons_list:
        return []

    atoms = set()
    for cons in cons_list:
        atoms.update(cons.split("&"))

    return sorted(atoms)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode VEP consequence and impact fields into ML-ready numeric features.

    - Splits compound consequences into atomic SO terms
    - One-hot encodes atomic consequences
    - One-hot encodes IMPACT
    - Ensures numeric dtypes (uint8)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'consequence_list' and 'impact'

    Returns
    -------
    pd.DataFrame
        ML-ready feature table
    """
    logger.info("Encoding atomic VEP consequences")

    # Split compound consequences into atomic terms
    df["atomic_consequences"] = df["consequence_list"].apply(
        _split_atomic_consequences
    )

    cons_dummies = (
        df["atomic_consequences"]
        .explode()
        .dropna()
        .pipe(pd.get_dummies)
        .groupby(level=0)
        .max()
        .add_prefix("cons_")
        .astype("uint8")
    )

    df = pd.concat(
        [df.drop(columns=["consequence_list", "atomic_consequences"]), cons_dummies],
        axis=1,
    )


    # ----------------------------
    # FIXED Impact encoding
    # ----------------------------
    logger.info("Encoding IMPACT")

    df["impact"] = df["impact"].fillna("NA").astype(str)

    impact_dummies = (
        pd.get_dummies(df["impact"], prefix="impact")
        .astype("uint8")
    )

    df = pd.concat(
        [df.drop(columns=["impact"]), impact_dummies],
        axis=1,
    )


    return df


# ------------------------------------------------------------------------------
# Save features
# ------------------------------------------------------------------------------
def save_features(df: pd.DataFrame, output_file: str, fmt: str) -> None:
    """
    Save features to disk.

    Parameters
    ----------
    df : pd.DataFrame
    output_file : str
    fmt : str
        csv or parquet
    """
    if fmt == "csv":
        df.to_csv(output_file, index=False)
    else:
        df.to_parquet(output_file, index=False)

    logger.info(f"Features written to {output_file}")
    logger.info(f"Feature matrix shape: {df.shape}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    df = read_vcf(args.input_vcf)
    df = encode_features(df)

    if args.filter_unknown:
        before = len(df)
        df = df[df["clnsig_label"] != -1].reset_index(drop=True)
        logger.info(
            f"Filtered unknown CLNSIG variants: "
            f"{before - len(df)} removed, {len(df)} remaining"
        )

    save_features(df, args.output_file, args.output_format)


if __name__ == "__main__":
    main()
