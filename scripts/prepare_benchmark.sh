#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# prepare_benchmark.sh
# --------------------
# 1) Download GIAB VCF + BED
# 2) Extract chr22 from GIAB files
# 3) Normalize merged_variants.vcf and GIAB chr22 VCF using GATK
# -------------------------------------------------------------------

# -------------------
# Setup and config
# -------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[INFO] Running from project root: $PROJECT_ROOT"

GIAB_DIR="data/giab"
CHR="chr22"
REF="data/reference/chr22.fa"
MERGED_VCF="data/merged_variants.vcf"

GIAB_VCF_URL="https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/HG002_NA24385_son/latest/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
GIAB_BED_URL="https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/HG002_NA24385_son/latest/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"

mkdir -p "$GIAB_DIR"

GIAB_VCF_FULL="$GIAB_DIR/HG002_GRCh38_full.vcf.gz"
GIAB_BED_FULL="$GIAB_DIR/HG002_GRCh38_full.bed"
GIAB_VCF_CHR="$GIAB_DIR/HG002_GRCh38_${CHR}.vcf.gz"
GIAB_BED_CHR="$GIAB_DIR/HG002_GRCh38_${CHR}.bed"
GIAB_VCF_NORM="$GIAB_DIR/HG002_GRCh38_${CHR}_norm.vcf"
MERGED_VCF_NORM="data/merged_variants_norm.vcf"

# -------------------
# 1) Download GIAB benchmark VCF + BED
# -------------------
if [[ -f "$GIAB_VCF_FULL" && -f "$GIAB_BED_FULL" ]]; then
    echo "[INFO] GIAB VCF and BED already exist — skipping download."
else
    echo "[INFO] Downloading GIAB benchmark VCF and BED..."
    wget -O "$GIAB_VCF_FULL" "$GIAB_VCF_URL"
    wget -O "$GIAB_BED_FULL" "$GIAB_BED_URL"
fi

# -------------------
# 2) Extract chr22 only
# -------------------
if [[ -f "$GIAB_VCF_CHR" && -f "$GIAB_BED_CHR" ]]; then
    echo "[INFO] chr22 VCF and BED already extracted — skipping."
else
    echo "[INFO] Extracting $CHR from GIAB files..."
    if [[ ! -f "${GIAB_VCF_FULL}.tbi" ]]; then
        echo "[INFO] Indexing GIAB full VCF..."
        tabix -p vcf "$GIAB_VCF_FULL"
    fi
    bcftools view -r "$CHR" -Oz -o "$GIAB_VCF_CHR" "$GIAB_VCF_FULL"
    tabix -p vcf "$GIAB_VCF_CHR"

    awk -v chr="$CHR" '$1==chr {print $0}' "$GIAB_BED_FULL" > "$GIAB_BED_CHR"
fi

# -------------------
# 3) Ensure reference index & dict exist
# -------------------
if [[ ! -f "${REF}.fai" ]]; then
    echo "[INFO] Indexing reference..."
    samtools faidx "$REF"
else
    echo "[INFO] Reference .fai index already exists."
fi

if [[ ! -f "${REF%.fa}.dict" ]]; then
    echo "[INFO] Creating GATK sequence dictionary..."
    gatk CreateSequenceDictionary -R "$REF" -O "${REF%.fa}.dict"
else
    echo "[INFO] Reference .dict already exists."
fi

# -------------------
# 4) Normalize both GIAB chr22 and merged_variants.vcf using GATK
# -------------------
normalize_vcf() {
    local input_vcf=$1
    local output_vcf=$2

    echo "[INFO] Normalizing $input_vcf with GATK..."
    gatk LeftAlignAndTrimVariants \
        -R "$REF" \
        -V "$input_vcf" \
        -O "$output_vcf" \
        --split-multi-allelics true \
        --dont-trim-alleles false

    # Compress with bgzip if not already compressed
    if [[ "$output_vcf" != *.gz ]]; then
        bgzip -f "$output_vcf"
        output_vcf="${output_vcf}.gz"
    fi

    # Create tabix index
    tabix -f -p vcf "$output_vcf"
}

# Normalize GIAB chr22
normalize_vcf "$GIAB_VCF_CHR" "$GIAB_VCF_NORM"

# Normalize merged_variants.vcf
normalize_vcf "$MERGED_VCF" "$MERGED_VCF_NORM"

echo "[INFO] ✅ DONE: Normalized files ready for ML feature extraction."
echo "      GIAB normalized VCF : ${GIAB_VCF_NORM}.gz"
echo "      Merged normalized VCF: ${MERGED_VCF_NORM}.gz"
