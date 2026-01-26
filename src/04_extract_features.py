#!/usr/bin/env python3
"""
04_extract_features.py

Extract ML-ready features from a VEP-annotated VCF file.

Features:
- One-hot encode multi-value Consequence
- One-hot encode Impact
- Numeric columns: SIFT, PolyPhen, Protein_position, DISTANCE
- CLNSIG mapped to ML label (pathogenic=1, benign=0, unknown=-1)
- Optional filtering of unknown CLNSIG
- Output as CSV or Parquet
- Configurable via YAML for pipeline consistency

Precedence
----------
CLI arguments > YAML config > hard-coded defaults

Usage:
    python scripts/04_extract_features.py <input_vcf.gz> <output_file> <config.yaml>

Examples:
    python3 src/04_extract_features.py data/processed/clinvar.vep.vcf.gz --config config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple, Optional, List

import pandas as pd
from cyvcf2 import VCF
from logzero import logger, setup_logger

# Shared config utilities
from utils.config import load_config, resolve

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
setup_logger()


# ------------------------------------------------------------------------------
# Argument container
# ------------------------------------------------------------------------------
class Args(NamedTuple):
    input_vcf: str
    output_file: Optional[str]
    output_format: Optional[str]
    filter_unknown: Optional[bool]
    config: Optional[str]


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
def parse_args() -> Args:
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
        help="Output feature file (without extension)",
    )

    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=None,
        help="Output format (overrides YAML)",
    )

    parser.add_argument(
        "--filter-unknown",
        action="store_true",
        default=None,
        help="Remove variants with unknown CLNSIG labels (overrides YAML)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pipeline YAML configuration file",
    )

    args = parser.parse_args()

    return Args(
        input_vcf=args.input_vcf,
        output_file=args.output_file,
        output_format=args.format,
        filter_unknown=args.filter_unknown,
        config=args.config,
    )


# ------------------------------------------------------------------------------
# CLNSIG → ML label mapping
# ------------------------------------------------------------------------------
def map_clnsig_to_label(clnsig: Optional[str]) -> int:
    """
    Map ClinVar CLNSIG string to ML label.

    Returns
    -------
    int
        1 = pathogenic
        0 = benign
       -1 = unknown / conflicting
    """
    if not clnsig:
        return -1

    clnsig = clnsig.lower()
    if clnsig in {"pathogenic", "likely_pathogenic"}:
        return 1
    if clnsig in {"benign", "likely_benign"}:
        return 0
    return -1


# ------------------------------------------------------------------------------
# Read VCF's INFO and CSQ
# ------------------------------------------------------------------------------
def read_vcf(input_vcf: str) -> pd.DataFrame:
    logger.info(f"Reading VCF: {input_vcf}")
    vcf = VCF(input_vcf)

    csq_info = vcf.get_header_type("CSQ")
    if csq_info is None:
        raise RuntimeError("CSQ INFO field not found in VCF header")

    csq_fields = csq_info["Description"].split("Format: ")[1].split("|")
    records: list[dict] = []

    for var in vcf:
        csq_raw = var.INFO.get("CSQ")
        if csq_raw:
            csq_entry = csq_raw.split(",")[0]
            csq = dict(zip(csq_fields, csq_entry.split("|")))
        else:
            csq = {}

        # ----------------------------
        # Safe numeric parsing
        # ----------------------------
        protein_position = None
        raw_pp = csq.get("Protein_position")
        if raw_pp:
            try:
                protein_position = int(raw_pp.split("-")[0])
            except ValueError:
                pass

        def _safe_float(x):
            try:
                return float(x) if x not in (None, "") else None
            except ValueError:
                return None

        def _safe_int(x):
            try:
                return int(x) if x not in (None, "") else None
            except ValueError:
                return None

        sift = _safe_float(csq.get("SIFT"))
        polyphen = _safe_float(csq.get("PolyPhen"))
        distance = _safe_int(csq.get("DISTANCE"))

        consequence_list = (
            csq.get("Consequence", "").split(",") if csq.get("Consequence") else []
        )

        clnsig = var.INFO.get("CLNSIG")
        if isinstance(clnsig, list):
            clnsig = clnsig[0]

        records.append({
            "chr": var.CHROM,
            "pos": var.POS,
            "ref": var.REF,
            "alt": ",".join(var.ALT),
            "impact": csq.get("IMPACT", "NA"),
            "sift": sift,
            "polyphen": polyphen,
            "protein_position": protein_position,
            "distance": distance,
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
    atoms = set()
    for cons in cons_list or []:
        atoms.update(cons.split("&"))
    return sorted(atoms)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding VEP consequences")

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

    logger.info("Encoding IMPACT")
    impact_dummies = pd.get_dummies(
        df["impact"].fillna("NA").astype(str),
        prefix="impact",
    ).astype("uint8")

    df = pd.concat([df.drop(columns=["impact"]), impact_dummies], axis=1)

    for col in ["sift", "polyphen", "protein_position", "distance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ------------------------------------------------------------------------------
# Save features
# ------------------------------------------------------------------------------
def save_features(df: pd.DataFrame, output_file: str, fmt: str) -> None:
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
    config = load_config(args.config)

    # ----------------------------
    # Resolve config values
    # ----------------------------
    extract_cfg = config.get("extract", {})

    output_format = resolve(
        args.output_format,
        extract_cfg.get("format"),
        "csv",
    )

    filter_unknown = resolve(
        args.filter_unknown,
        extract_cfg.get("filter_unknown"),
        False,
    )

    input_vcf = Path(args.input_vcf)

    if args.output_file:
        output_path = Path(args.output_file).with_suffix(f".{output_format}")
    else:
        output_path = (
            input_vcf
            .with_suffix("")
            .with_suffix(f".features.{output_format}")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Effective settings | "
        f"format={output_format}, filter_unknown={filter_unknown}"
    )

    # ----------------------------
    # Run extraction
    # ----------------------------
    df = read_vcf(str(input_vcf))
    df = encode_features(df)

    if filter_unknown:
        before = len(df)
        df = df[df["clnsig_label"] != -1].reset_index(drop=True)
        logger.info(
            f"Filtered unknown CLNSIG: {before - len(df)} removed"
        )

    save_features(df, str(output_path), output_format)


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
