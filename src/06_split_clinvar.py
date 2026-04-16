#!/usr/bin/env python3
"""
Deterministic Data Splitter for Genomic Feature Matrices.

This module partitions variant data into training, testing, and inference sets.
Unlike standard random splits, this script uses deterministic hashing to ensure:
1. Stability: Re-running the script on the same data yields the same splits.
2. Leakage Prevention: Variants are grouped so that all mutations for a specific
   entity (e.g., a Gene or Chromosome) stay within the same partition.

Outputs:
    - {base}.train.parquet: Used for model fitting.
    - {base}.test.parquet: Used for unbiased performance evaluation.
    - {base}.infer.parquet: Features only, used for simulated production runs.
    - {base}.infer_with_labels.parquet: Audit file for inference validation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np
from logzero import logger, setup_logger

# Fallback for config logic
try:
    from utils.config import load_config, resolve
except ImportError:
    def load_config(x): return {}
    def resolve(cli, cfg, default): return cli if cli is not None else (cfg if cfg is not None else default)

setup_logger()

class SplitConfig:
    """Production-grade container for split parameters."""
    def __init__(self, train: float, test: float, infer: float, group_by: str = "variant"):
        if not np.isclose(train + test + infer, 1.0):
            raise ValueError(f"Split fractions must sum to 1.0 (Current: {train+test+infer})")
        self.train = train
        self.test = test
        self.infer = infer
        self.group_by = group_by

def get_stable_hash_fraction(identifier: str) -> float:
    """
    Maps any string to a deterministic float in [0, 1) using SHA-256.
    """
    hash_hex = hashlib.sha256(identifier.encode()).hexdigest()
    return int(hash_hex[:8], 16) / 0xFFFFFFFF

def assign_split_logic(val: float, config: SplitConfig) -> str:
    """Maps a hash fraction to a split name."""
    if val < config.train:
        return "train"
    if val < (config.train + config.test):
        return "test"
    return "infer"

def main():
    parser = argparse.ArgumentParser(description="Deterministic Genomic Splitter")
    parser.add_argument("--input", required=True, help="Path to features (Parquet/CSV)")
    parser.add_argument("--outdir", default="data/splits", help="Output directory")
    parser.add_argument("--group-by", choices=["variant", "gene", "chr"], default="variant", 
                        help="Level of data grouping to prevent leakage")
    parser.add_argument("--config", help="Path to YAML config")
    args = parser.parse_args()

    # 1. Setup Config
    # Defaulting to 70/15/15 split
    cfg = SplitConfig(train=0.70, test=0.15, infer=0.15, group_by=args.group_by)
    input_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    logger.info(f"Loading feature matrix from {input_path}...")
    df = pd.read_parquet(input_path) if input_path.suffix == '.parquet' else pd.read_csv(input_path)
    
    # 3. Define Grouping Key
    # If grouping by variant: hash 'chr:pos:ref:alt'
    # If grouping by gene: hash 'gene_symbol'
    # If grouping by chr: hash 'chr'
    if args.group_by == "variant":
        df["split_key"] = (df["chr"].astype(str) + ":" + df["pos"].astype(str) + 
                           ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str))
    elif args.group_by == "gene":
        # Note: Ensure you extracted 'SYMBOL' or 'Gene' in 04_extract_features
        if "gene_symbol" not in df.columns:
            logger.error("Group-by 'gene' requested but 'gene_symbol' column is missing.")
            return
        df["split_key"] = df["gene_symbol"].astype(str)
    else:
        df["split_key"] = df["chr"].astype(str)

    # 4. Deterministic Assignment
    logger.info(f"Assigning splits grouped by: {args.group_by}")
    
    # Vectorized hashing for unique keys to save memory/time
    unique_keys = df["split_key"].unique()
    key_map = {k: assign_split_logic(get_stable_hash_fraction(k), cfg) for k in unique_keys}
    
    df["split"] = df["split_key"].map(key_map)
    df.drop(columns=["split_key"], inplace=True)

    # 5. Materialize Splits
    base = input_path.stem.replace(".features", "")
    for split_name in ["train", "test", "infer"]:
        subset = df[df["split"] == split_name].drop(columns=["split"])
        
        if split_name == "infer":
            # Save audit version
            subset.to_parquet(out_dir / f"{base}.infer_with_labels.parquet", index=False)
            # Save production version (no labels)
            subset.drop(columns=["clnsig_label"]).to_parquet(out_dir / f"{base}.infer.parquet", index=False)
        else:
            subset.to_parquet(out_dir / f"{base}.{split_name}.parquet", index=False)
        
        logger.info(f"Exported {split_name}: {len(subset):,} variants ({len(subset)/len(df):.1%})")

if __name__ == "__main__":
    main()