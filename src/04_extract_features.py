#!/usr/bin/env python3
"""
04_extract_features.py

Extract ML-ready features from a VEP-annotated VCF file for downstream ML (06_split_clinvar, 07_train_model).

Features
--------
- One-hot encode multi-value Consequence (atomic terms, e.g. missense_variant)
- One-hot encode Impact (MODERATE, HIGH, etc.)
- Numeric columns: SIFT, PolyPhen, Protein_position, DISTANCE
- CLNSIG mapped to ML label (pathogenic=1, benign=0, unknown=-1)
- Optional filtering of unknown CLNSIG (--filter-unknown or config)
- Output as CSV or Parquet

Precedence: CLI arguments > YAML config > hard-coded defaults

Usage
-----
    python3 src/04_extract_features.py <input_vcf.gz> [output_file] [--format csv|parquet] [--filter-unknown] [--config CONFIG]

Example
-------
    python3 src/04_extract_features.py data/processed/clinvar.vep.vcf.gz --config config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple, Optional

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
    """
    Parsed CLI arguments: input VCF, optional output path/format/filter, config path.
    """

    input_vcf: str
    output_file: Optional[str]
    output_format: Optional[str]
    filter_unknown: Optional[bool]
    config: Optional[str]


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
def parse_args() -> Args:
    """
    Parse CLI: input VCF (required), optional output path, format, filter, config.
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
        help="Output path (without extension). If omitted, derived from input as <input>.features.<format>",
    )

    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=None,
        help="Output format (overrides config extract.format)",
    )

    parser.add_argument(
        "--filter-unknown",
        action="store_true",
        default=None,
        help="Drop variants with unknown/conflicting CLNSIG (overrides config extract.filter_unknown)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (extract.format, extract.filter_unknown)",
    )

    args = parser.parse_args()

    # Return parsed arguments
    parsed = Args(
        input_vcf=args.input_vcf,
        output_file=args.output_file,
        output_format=args.format,
        filter_unknown=args.filter_unknown,
        config=args.config,
    )   
    return parsed


# ------------------------------------------------------------------------------
# Safe numeric parsing (VEP can leave fields empty)
# ------------------------------------------------------------------------------
def _safe_float(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x not in (None, "") else None
    except (ValueError, TypeError):
        return None


def _safe_int(x: Optional[str]) -> Optional[int]:
    try:
        return int(x) if x not in (None, "") else None
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------------------
# CLNSIG → ML label mapping
# ------------------------------------------------------------------------------
def map_clnsig_to_label(clnsig: Optional[str]) -> int:
    """
    Map ClinVar CLNSIG string to numeric ML label.

    Returns
    -------
    int
        1 = pathogenic / likely_pathogenic
        0 = benign / likely_benign
       -1 = unknown / conflicting / other
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
# Read VCF and parse INFO/CSQ
# ------------------------------------------------------------------------------
def read_vcf(input_vcf: str) -> pd.DataFrame:
    """
    Load VEP-annotated VCF and parse CSQ (Consequence) and INFO (CLNSIG, etc.) into a flat table.
    Uses first CSQ subfield per variant when multiple transcripts are present.
    """
    logger.info(f"Reading VCF: {input_vcf}")
    vcf = VCF(input_vcf)

    # CSQ format is described in header (e.g. "Format: Allele|Consequence|IMPACT|...")
    csq_info = vcf.get_header_type("CSQ")
    if csq_info is None:
        raise RuntimeError("CSQ INFO field not found in VCF header")

    desc = csq_info.get("Description", "")
    if "Format: " not in desc:
        raise RuntimeError("CSQ Description does not contain 'Format: '; cannot parse fields")
    csq_fields = desc.split("Format: ")[1].strip().split("|")
    records: list[dict] = []

    for var in vcf:
        # First transcript/annotation only
        csq_raw = var.INFO.get("CSQ")
        if csq_raw:
            csq_entry = csq_raw.split(",")[0]
            csq = dict(zip(csq_fields, csq_entry.split("|")))
        else:
            csq = {}

        protein_position = None
        raw_pp = csq.get("Protein_position")
        if raw_pp:
            try:
                protein_position = int(str(raw_pp).split("-")[0])
            except ValueError:
                pass

        sift = _safe_float(csq.get("SIFT"))
        polyphen = _safe_float(csq.get("PolyPhen"))
        distance = _safe_int(csq.get("DISTANCE"))

        # Consequence can be comma-separated (e.g. "missense_variant,splice_region_variant")
        consequence_list = (
            csq.get("Consequence", "").split(",") if csq.get("Consequence") else []
        )

        clnsig = var.INFO.get("CLNSIG")
        if isinstance(clnsig, list):
            clnsig = clnsig[0]

        # ALT can be None in some VCFs; join safely
        alt_str = ",".join(var.ALT) if var.ALT else ""

        records.append({
            "chr": var.CHROM,
            "pos": var.POS,
            "ref": var.REF,
            "alt": alt_str,
            "impact": csq.get("IMPACT", "NA"),
            "sift": sift,
            "polyphen": polyphen,
            "protein_position": protein_position,
            "distance": distance,
            "clnsig": clnsig,
            "clnsig_label": map_clnsig_to_label(clnsig),
            "consequence_list": consequence_list,
        })

    if not records:
        # Empty VCF: return DataFrame with expected schema so encode_features/save_features work
        empty_columns = [
            "chr", "pos", "ref", "alt", "impact", "sift", "polyphen",
            "protein_position", "distance", "clnsig", "clnsig_label", "consequence_list",
        ]
        df = pd.DataFrame(columns=empty_columns)
        logger.warning("VCF contains no variants; output will be empty")
    else:
        df = pd.DataFrame.from_records(records)

    logger.info(f"Parsed {len(df)} variants from VCF")
    return df


# ------------------------------------------------------------------------------
# Feature encoding
# ------------------------------------------------------------------------------
def _split_atomic_consequences(cons_list: list[str]) -> list[str]:
    """
    Split VEP Consequence strings into atomic terms (e.g. 'X&Y' -> ['X', 'Y']), sorted.
    """
    atoms = set()
    for cons in cons_list or []:
        atoms.update(cons.split("&"))
    return sorted(atoms)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode Consequence and IMPACT; ensure numeric columns are numeric.
    Drops consequence_list / atomic_consequences / impact; adds cons_* and impact_* columns.
    Handles empty DataFrame (no variants) without running one-hot encoding.
    """
    if len(df) == 0:
        logger.info("No variants to encode; returning empty feature matrix")
        # Drop columns that would be encoded so schema is consistent (no consequence_list/impact)
        cols_to_drop = [c for c in ["consequence_list", "impact"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        for col in ["sift", "polyphen", "protein_position", "distance"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    logger.info("Encoding VEP consequences")

    # Atomic consequences (e.g. missense_variant, splice_region_variant) -> one-hot cons_*
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

    # Ensure SIFT, PolyPhen, protein_position, distance are numeric (for imputation downstream)
    for col in ["sift", "polyphen", "protein_position", "distance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ------------------------------------------------------------------------------
# Save features
# ------------------------------------------------------------------------------
def save_features(df: pd.DataFrame, output_file: str, fmt: str) -> None:
    """
    Write feature matrix to disk as CSV or Parquet (no row index).
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
    config = load_config(args.config) if args.config else {}

    # --- Resolve output format and filter (CLI > config > defaults) ---
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
    if not input_vcf.exists():
        raise FileNotFoundError(f"Input VCF not found: {input_vcf}")

    # Output path: explicit path (with extension) or <input>.features.<format>
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

    # --- Read VCF, encode features, optionally drop unknown CLNSIG, save ---
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
