#!/usr/bin/env bash
###############################################################################
# Preprocess ClinVar VCF (Primary assembly only)
#
# - Keeps only primary chromosomes: 1-22, X, Y, MT
# - Splits multiallelic variants into multiple rows
# - Left-aligns and normalizes against reference genome
# - Handles gzipped reference FASTA automatically
# - Compresses and indexes output
# - Generates basic QC statistics
#
# Requirements:
#   bcftools >= 1.15
#   tabix
#   samtools
#
# Usage:
#   ./preprocess_clinvar.sh \
#     data/raw/clinvar.vcf.gz \
#     data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
#     data/processed
#
###############################################################################

set -euo pipefail

##############################
# Logging utilities
##############################
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
error_exit() { echo "[ERROR] $*" >&2; exit 1; }

##############################
# Input arguments
##############################
if [[ $# -ne 3 ]]; then
    error_exit "Usage: $0 <clinvar.vcf.gz> <reference.fa(.gz)> <output_dir>"
fi

CLINVAR_VCF="$1"
REFERENCE_FASTA="$2"
OUTPUT_DIR="$3"

##############################
# Validate inputs
##############################
[[ -f "$CLINVAR_VCF" ]] || error_exit "ClinVar VCF not found: $CLINVAR_VCF"
[[ -f "$REFERENCE_FASTA" ]] || error_exit "Reference FASTA not found: $REFERENCE_FASTA"

command -v bcftools >/dev/null 2>&1 || error_exit "bcftools not found"
command -v tabix    >/dev/null 2>&1 || error_exit "tabix not found"
command -v samtools >/dev/null 2>&1 || error_exit "samtools not found"

##############################
# Prepare output paths
##############################
mkdir -p "$OUTPUT_DIR"

OUTPUT_VCF="${OUTPUT_DIR}/clinvar.norm.vcf.gz"
STATS_FILE="${OUTPUT_DIR}/clinvar.norm.stats.txt"
PRIMARY_VCF="${OUTPUT_DIR}/clinvar.primary.vcf.gz"

##############################
# Handle reference FASTA
##############################
FASTA_BASENAME=$(basename "$REFERENCE_FASTA" .gz)
FASTA_PATH="data/reference/${FASTA_BASENAME}"

if [[ "$REFERENCE_FASTA" == *.gz && ! -f "$FASTA_PATH" ]]; then
    log "Decompressing reference genome: $REFERENCE_FASTA"
    gzip -dc "$REFERENCE_FASTA" > "$FASTA_PATH"
fi

if [[ ! -f "${FASTA_PATH}.fai" ]]; then
    log "Indexing reference genome: $FASTA_PATH"
    samtools faidx "$FASTA_PATH"
fi

##############################
# Keep only primary chromosomes
##############################
log "Filtering to primary chromosomes (1-22, X, Y, MT)"
bcftools view \
  -r 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y,MT \
  "$CLINVAR_VCF" \
  -Oz -o "$PRIMARY_VCF"

tabix -p vcf "$PRIMARY_VCF"

##############################
# Normalize & split multiallelics
##############################
log "Normalizing VCF (split multiallelics, left-align, normalize alleles)"
bcftools norm \
  -m - \
  -f "$FASTA_PATH" \
  -Oz \
  -o "$OUTPUT_VCF" \
  "$PRIMARY_VCF"

##############################
# Index output
##############################
log "Indexing normalized VCF"
tabix -p vcf "$OUTPUT_VCF"

##############################
# QC statistics
##############################
log "Generating VCF statistics"
bcftools stats "$OUTPUT_VCF" > "$STATS_FILE"

##############################
# Final summary
##############################
VARIANT_COUNT=$(bcftools view -H "$OUTPUT_VCF" | wc -l | tr -d ' ')
log "Preprocessing complete"
log "Total normalized variants: $VARIANT_COUNT"
log "Stats written to: $STATS_FILE"
log "Done"
