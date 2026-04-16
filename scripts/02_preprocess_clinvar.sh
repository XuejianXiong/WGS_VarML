#!/usr/bin/env bash

# ==============================================================================
# Script:       02_preprocess_clinvar.sh
# Project:      WGS_VarML
# Description:  Normalizes and filters ClinVar VCF data for ML readiness.
# Features:     - Canonical chromosome filtering (1-22, X, Y, MT)
#               - Multiallelic decomposition (essential for ML feature vectors)
#               - Left-alignment against GRCh38 reference assembly
#               - Automated Samtools/Tabix indexing
#               - Performance logging and variant counting
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ANSI Colors for professional UI
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# ==========================================
# Logging & Error Handling
# ==========================================
log()   { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: ${NC}$*"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${NC}$*"; exit 1; }

# ==========================================
# Environment Pre-flight Checks
# ==========================================
check_tool() {
    command -v "$1" >/dev/null 2>&1 || error "Dependency missing: $1. Please install it to continue."
}

usage() {
    cat << EOF
Usage: $0 <clinvar.vcf.gz> <reference.fa.gz> <output_dir>
Example: $0 data/raw/clinvar.vcf.gz data/reference/GRCh38.fa.gz data/processed
EOF
    exit 1
}

# 1. Validate Arguments
[[ $# -ne 3 ]] && usage

CLINVAR_VCF="$1"
REFERENCE_GZ="$2"
OUTPUT_DIR="$3"

# 2. Check Dependencies
for tool in bcftools tabix samtools gzip; do check_tool "$tool"; done

# 3. Create Output structure
mkdir -p "$OUTPUT_DIR"
PRIMARY_VCF="${OUTPUT_DIR}/clinvar.primary.vcf.gz"
OUTPUT_VCF="${OUTPUT_DIR}/clinvar.norm.vcf.gz"
STATS_FILE="${OUTPUT_DIR}/clinvar.norm_stats.txt"

# ==========================================
# Main Processing Pipeline
# ==========================================

# Step A: Reference Genome Preparation
# Note: bcftools norm -f requires an uncompressed Fasta + FAI index for speed
FASTA_UNZIPPED="${REFERENCE_GZ%.gz}"

if [[ "$REFERENCE_GZ" == *.gz && ! -f "$FASTA_UNZIPPED" ]]; then
    log "Decompressing reference assembly for normalization..."
    gzip -dc "$REFERENCE_GZ" > "$FASTA_UNZIPPED"
fi

if [[ ! -f "${FASTA_UNZIPPED}.fai" ]]; then
    log "Indexing reference genome with samtools..."
    samtools faidx "$FASTA_UNZIPPED"
fi

# Step B: Filter to Primary Chromosomes
# This removes decoys and unplaced scaffolds to focus on high-confidence regions
log "Step 1/3: Filtering to primary chromosomes (1-22, X, Y, MT)..."
bcftools view \
  -r 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y,MT \
  "$CLINVAR_VCF" \
  -Oz -o "$PRIMARY_VCF"

tabix -f -p vcf "$PRIMARY_VCF"

# Step C: Decomposition and Left-Alignment
# -m - : Splits multiallelic sites into separate records
# -f   : Realigns alleles to the reference genome (essential for consistency)
log "Step 2/3: Normalizing alleles and decomposing multiallelics..."
bcftools norm \
  -m - \
  -f "$FASTA_UNZIPPED" \
  -Oz -o "$OUTPUT_VCF" \
  "$PRIMARY_VCF"

tabix -f -p vcf "$OUTPUT_VCF"

# Step D: QC & Summary
log "Step 3/3: Generating post-normalization statistics..."
bcftools stats "$OUTPUT_VCF" > "$STATS_FILE"

# Clean up intermediate primary file to save space
rm -f "$PRIMARY_VCF" "${PRIMARY_VCF}.tbi"

# ==========================================
# Final Summary
# ==========================================
VAR_COUNT=$(bcftools index -n "$OUTPUT_VCF")
log "${BOLD}Preprocessing Complete!${NC}"
log "Final Variant Count: ${VAR_COUNT}"
log "Output Location:     ${OUTPUT_VCF}"
log "QC Stats Location:   ${STATS_FILE}"