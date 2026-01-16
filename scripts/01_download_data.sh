#!/usr/bin/env bash
# ==============================================================================
# Script: download_all_data.sh
# Project: WGS_VarML
# Purpose: Download public variant dataset ClinVar and Reference genome for ML
#
# Features:
#   - Correct, canonical data sources
#   - Skip download if file already exists
#   - Resumable downloads
#   - Optional checksum verification
#   - Production-level logging and error handling
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ==========================
# Logging
# ==========================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ==========================
# Download helper
# ==========================
download_if_missing() {
    local url="$1"
    local dest="$2"
    local checksum="${3:-}"

    if [[ -f "$dest" ]]; then
        log "File exists, skipping: $dest"
        return 0
    fi

    log "Downloading: $url"
    wget -c -O "$dest" "$url"

    if [[ -n "$checksum" ]]; then
        log "Verifying checksum for $dest"
        echo "$checksum  $dest" | sha256sum -c -
    else
        log "No checksum provided for $dest, skipping verification."
    fi
}

# ==========================
# Create directories
# ==========================
log "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/reference

# ==========================
# ClinVar (GRCh38)
# ==========================
log "Downloading ClinVar GRCh38..."

CLINVAR_BASE="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38"
CLINVAR_VCF="clinvar.vcf.gz"
CLINVAR_TBI="clinvar.vcf.gz.tbi"

download_if_missing \
  "${CLINVAR_BASE}/${CLINVAR_VCF}" \
  "data/raw/${CLINVAR_VCF}"

download_if_missing \
  "${CLINVAR_BASE}/${CLINVAR_TBI}" \
  "data/raw/${CLINVAR_TBI}"

# ==========================
# Reference genome (GRCh38, Ensembl)
# ==========================
log "Downloading GRCh38 reference genome (Ensembl)..."

ENSEMBL_BASE="https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/dna"
REF_FASTA="Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

download_if_missing \
  "${ENSEMBL_BASE}/${REF_FASTA}" \
  "data/reference/${REF_FASTA}"

log "All required reference data are present."
