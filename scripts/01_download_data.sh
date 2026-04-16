#!/usr/bin/env bash

# ==============================================================================
# Script:       01_download_data.sh
# Project:      WGS_VarML
# Description:  Automated retrieval of clinical variants (ClinVar) and 
#               reference assemblies (GRCh38) for ML feature engineering.
# Documentation:
#   - Data is stored in data/raw and data/reference.
#   - Uses wget with resume capability (-c).
#   - Validates integrity via MD5 checksums for ClinVar.
# ==============================================================================

# Strict error handling
set -euo pipefail
IFS=$'\n\t'

# ANSI Color Codes for professional logging
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================
# Functions
# ==========================================

log()   { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: ${NC}$*"; }
warn()  { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: ${NC}$*"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${NC}$*"; exit 1; }

check_tool() {
    if ! command -v "$1" &> /dev/null; then
        error "Dependency missing: $1. Please install it to continue."
    fi
}

download_file() {
    local url="$1"
    local dest="$2"

    if [[ -f "$dest" ]]; then
        log "File already exists, skipping download: $(basename "$dest")"
    else
        log "Downloading: $url"
        # --retry-connrefused handles transient network blips common in large downloads
        wget --quiet --show-progress --retry-connrefused --waitretry=5 -c -O "$dest" "$url" || \
            error "Failed to download $url"
    fi
}

# ==========================================
# Main Execution
# ==========================================

# 1. Environment Pre-flight Checks
check_tool wget
check_tool md5sum

# Ensure running from project root
if [[ ! -d "src" ]]; then
    error "Script must be executed from the project root."
fi

# 2. Setup Directory Structure
log "Initializing local data filesystem..."
mkdir -p data/raw data/reference

# 3. ClinVar Retrieval (GRCh38)
# We fetch the VCF, the Tabix index, and the MD5 for verification
log "Syncing ClinVar clinical significance labels..."
CLINVAR_BASE="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38"

for ext in "vcf.gz" "vcf.gz.tbi" "vcf.gz.md5"; do
    download_file "${CLINVAR_BASE}/clinvar.${ext}" "data/raw/clinvar.${ext}"
done

# 4. Data Integrity Verification
#log "Verifying ClinVar data integrity..."
#(cd data/raw && md5sum -c clinvar.vcf.gz.md5) || error "MD5 Checksum mismatch for ClinVar VCF!"

# 4. Data Integrity Verification
log "Verifying ClinVar data integrity..."

# We use awk to grab just the MD5 hash (column 1) and pair it with the local filename.
# This bypasses the absolute NCBI paths that caused the "No such file" error.
if [[ -f "data/raw/clinvar.vcf.gz.md5" ]]; then
    MD5_HASH=$(awk '{print $1}' data/raw/clinvar.vcf.gz.md5)
    
    # Use 'echo' to pass the cleaned string to md5sum
    if ! echo "$MD5_HASH  data/raw/clinvar.vcf.gz" | md5sum -c -; then
        error "MD5 Checksum mismatch for ClinVar VCF! The file may be corrupted."
    fi
    log "ClinVar integrity verified."
else
    warn "MD5 file not found. Skipping integrity check."
fi

# 5. Reference Assembly (GRCh38)
log "Retrieving Ensembl GRCh38 Primary Assembly..."
ENSEMBL_BASE="https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/dna"
REF_FASTA="Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

download_file "${ENSEMBL_BASE}/${REF_FASTA}" "data/reference/${REF_FASTA}"

log "${BOLD}Data acquisition complete. Pipeline ready for Pre-processing.${NC}"
