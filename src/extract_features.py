#!/usr/bin/env python3
"""
extract_features.py
--------------------
Extract variant-level features from a VCF file for downstream ML/AI analysis.

Input:
    data/merged_variants.vcf

Output:
    features/variant_features.csv
"""

import os
import pandas as pd
import vcf  # PyVCF (install via pip install pyvcf)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
VCF_PATH = "data/merged_variants.vcf"
OUTPUT_DIR = "features"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "variant_features.csv")

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def extract_variant_features(record):
    """Extract useful fields from a VCF record."""
    try:
        return {
            "CHROM": record.CHROM,
            "POS": record.POS,
            "ID": record.ID,
            "REF": record.REF,
            "ALT": str(record.ALT[0]) if record.ALT else None,
            "QUAL": record.QUAL,
            "FILTER": ";".join(record.FILTER) if record.FILTER else "PASS",
            "DP": record.INFO.get("DP"),
            "MQ": record.INFO.get("MQ"),
            "QD": record.INFO.get("QD"),
            "FS": record.INFO.get("FS"),
            "SOR": record.INFO.get("SOR"),
            "ReadPosRankSum": record.INFO.get("ReadPosRankSum"),
            "MQRankSum": record.INFO.get("MQRankSum"),
        }
    except Exception as e:
        print(f"Error processing record at {record.CHROM}:{record.POS} — {e}")
        return None

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    if not os.path.exists(VCF_PATH):
        raise FileNotFoundError(f"VCF not found: {VCF_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vcf_reader = vcf.Reader(open(VCF_PATH, "r"))
    records = []
    for rec in vcf_reader:
        data = extract_variant_features(rec)
        if data:
            records.append(data)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Extracted features for {len(df)} variants → {OUTPUT_FILE}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
