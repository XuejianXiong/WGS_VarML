#!/usr/bin/env python3

"""
06_split_clinvar.py

Deterministically split ClinVar feature data into:
  - train
  - test
  - infer (labels removed)
  - infer_with_labels (audit only)

Notes:
------
  - Hash-based, stable across re-runs
  - Variant-identity–based split (chr:pos:ref:alt)
  - Supports CSV and Parquet inputs

Usage:
------
    python3 src/06_split_clinvar.py --input <features_csv_or_parquet> [--outdir DIR] [--config CONFIG]

Example:
------
    python3 src/06_split_clinvar.py --input data/processed/clinvar.vep.features.csv --outdir data/splits --config config/config.yaml
"""


from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from logzero import logger, setup_logger

from utils.config import load_config, resolve


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse CLI: input feature file, output directory, config path for split fractions.
    """
    parser = argparse.ArgumentParser(
        description="Deterministic ClinVar split for training and inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="ClinVar VEP feature matrix: clinvar.vep.features.csv or .parquet",
    )

    parser.add_argument(
        "--outdir",
        default="data/splits",
        help="Output directory for train/test/infer files",
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config with split.train_frac, split.test_frac, split.infer_frac",
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


def stable_hash(value: str) -> float:
    """
    Map a string (e.g. variant key) to a deterministic float in [0, 1).
    Same key always yields same value across runs; used for reproducible splits.
    """
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_split(variant_key: str, train_frac: float, test_frac: float) -> str:
    """
    Assign variant to 'train', 'test', or 'infer' based on hash and fractions.
    """
    r = stable_hash(variant_key)
    if r < train_frac:
        return "train"
    elif r < train_frac + test_frac:
        return "test"
    else:
        return "infer"


def read_features(path: Path) -> pd.DataFrame:
    """
    Load feature matrix from CSV or Parquet; format inferred from file extension.
    """
    logger.info(f"Reading features from {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def write_df(df: pd.DataFrame, out_dir: Path, base_name: str, split: str, suffix: str):
    """
    Write one split to disk as base_name.{split}{suffix} (e.g. clinvar.vep.features.train.parquet).
    """
    out_path = out_dir / f"{base_name}.{split}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    logger.info(f"Wrote {len(df):,} rows to {out_path}")


# ------------------------------------------------------------------------------
# Configuration helpers
# ------------------------------------------------------------------------------


def load_and_resolve_split_config(config_path: str) -> tuple[float, float, float]:
    """
    Load train/test/infer fractions from config (under 'split' section).
    Returns (train_frac, test_frac, infer_frac); fractions must sum to 1.0.
    """
    config = load_config(config_path)
    # Config may nest under "split"; try both top-level and nested
    split_cfg = config.get("split", config)

    train_frac = resolve(None, split_cfg.get("train_frac"), 0.70)
    test_frac = resolve(None, split_cfg.get("test_frac"), 0.15)
    infer_frac = resolve(None, split_cfg.get("infer_frac"), 0.15)

    if abs(train_frac + test_frac + infer_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + test_frac + infer_frac must sum to 1.0")

    logger.info(
        f"Using split fractions: "
        f"train={train_frac}, test={test_frac}, infer={infer_frac}"
    )

    return train_frac, test_frac, infer_frac


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------


def validate_input_schema(df: pd.DataFrame):
    """
    Ensure required columns exist for variant key (chr, pos, ref, alt) and label (clnsig_label).
    Raises ValueError if any are missing.
    """
    required_cols = {"chr", "pos", "ref", "alt", "clnsig_label"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ------------------------------------------------------------------------------
# Split logic
# ------------------------------------------------------------------------------


def assign_variant_splits(
    df: pd.DataFrame,
    train_frac: float,
    test_frac: float,
) -> pd.Series:
    """
    Assign each row to 'train', 'test', or 'infer' using a hash of chr:pos:ref:alt.
    Same variant always gets the same split across runs (reproducible).
    Returns a Series of split labels aligned to df.
    """
    # Build stable variant key for hashing (used by 07_train_model / 08_model_inference for identity)
    variant_keys = (
        df["chr"].astype(str)
        + ":"
        + df["pos"].astype(str)
        + ":"
        + df["ref"].astype(str)
        + ":"
        + df["alt"].astype(str)
    )

    splits = variant_keys.map(lambda key: assign_split(key, train_frac, test_frac))

    return splits


# ------------------------------------------------------------------------------
# Output materialization
# ------------------------------------------------------------------------------


def materialize_splits(
    df: pd.DataFrame,
    output_dir: Path,
    base_name: str,
    suffix: str,
):
    """
    Write four files: train, test, infer (no labels), infer_with_labels (audit).
    Expects df to have a 'split' column with values 'train' / 'test' / 'infer'.
    """
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    # Infer set: one version without labels (for 08_model_inference), one with (for auditing)
    infer_with_labels = df[df["split"] == "infer"].drop(columns=["split"])
    infer_df = infer_with_labels.drop(columns=["clnsig_label"])

    write_df(train_df, output_dir, base_name, "train", suffix)
    write_df(test_df, output_dir, base_name, "test", suffix)
    write_df(infer_df, output_dir, base_name, "infer", suffix)
    write_df(infer_with_labels, output_dir, base_name, "infer_with_labels", suffix)

    logger.info(
        f"Split sizes — "
        f"train: {len(train_df):,}, "
        f"test: {len(test_df):,}, "
        f"infer: {len(infer_df):,}"
    )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main() -> None:
    setup_logger()
    args = parse_args()

    # --- Load split fractions from config (must sum to 1.0) ---
    train_frac, test_frac, infer_frac = load_and_resolve_split_config(args.config)

    input_path = Path(args.input)
    output_dir = Path(args.outdir)

    # Base name for output files (e.g. clinvar.vep.features -> .train.parquet, .test.parquet, ...)
    base_name = input_path.stem
    if input_path.suffix == ".gz":
        base_name = Path(base_name).stem

    # --- Load feature matrix and validate required columns ---
    df = read_features(input_path)
    logger.info(f"Total variants: {len(df):,}")
    validate_input_schema(df)

    # --- Assign each variant to train / test / infer via hash of chr:pos:ref:alt ---
    logger.info("Assigning deterministic variant-level splits")
    df["split"] = assign_variant_splits(df, train_frac, test_frac)

    for split, frac in df["split"].value_counts(normalize=True).sort_index().items():
        logger.info(f"Split {split:>5}: {frac:.2%}")

    # --- Write train, test, infer, and infer_with_labels to disk ---
    materialize_splits(
        df=df,
        output_dir=output_dir,
        base_name=base_name,
        suffix=input_path.suffix,
    )

    logger.info("ClinVar split completed successfully")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
