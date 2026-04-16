#!/usr/bin/env bash

# ==============================================================================
# Script:       03_run_vep_docker.sh
# Project:      WGS_VarML
# Description:  Orchestrates Ensembl Variant Effect Predictor (VEP) via Docker
#               to generate ML-ready functional annotations for NGS data.
#
# Technical Highlights for ML:
#   - Deterministic Transcript Selection: Uses --pick and --pick_order to ensure
#     a 1:1 relationship between variants and features.
#   - Resource Optimization: Configurable --forking and --buffer_size for HPC.
#   - Environment Isolation: Encapsulates complex Perl/Bio::DB dependencies.
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ANSI Colors
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ==========================================
# Logging & Error Handling
# ==========================================
log()   { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: ${NC}$*"; }
warn()  { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: ${NC}$*"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${NC}$*"; exit 1; }

# Handle interrupts (Ctrl+C) gracefully
trap 'error "Script interrupted by user."' INT TERM

# ==========================================
# Configuration & Defaults
# ==========================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VEP_CACHE_DIR="$HOME/.vep"
VEP_IMAGE="ensemblorg/ensembl-vep:latest"

# Default feature set for ML modeling
DEFAULT_FIELDS="Uploaded_variation,Location,Allele,Gene,SYMBOL,Feature,Feature_type,BIOTYPE,CANONICAL,MANE_SELECT,Consequence,IMPACT,EXON,INTRON,Protein_position,Amino_acids,Codons,SIFT,PolyPhen,DISTANCE,STRAND"

# ----------------------------
# Usage / Help
# ----------------------------
usage() {
    cat << EOF
${BOLD}Usage:${NC} $0 <input_vcf> <output_vcf> [OPTIONS]

${BOLD}Arguments:${NC}
  input_vcf        Path relative to project root (e.g., data/processed/clinvar.norm.vcf.gz)
  output_vcf       Path relative to project root (e.g., data/processed/clinvar.vep.vcf.gz)

${BOLD}Options:${NC}
  --threads N      CPU cores (Default: 4)
  --assembly STR   Genome assembly (Default: GRCh38)
  --buffer N       VEP buffer size (Default: 5000)
  --fields STR     Custom comma-separated fields
EOF
    exit 1
}

[[ $# -lt 2 ]] && usage

INPUT_VCF="$1"
OUTPUT_VCF="$2"
shift 2

# Parsing optional overrides
THREADS=4
ASSEMBLY="GRCh38"
BUFFER_SIZE=5000
FIELDS="$DEFAULT_FIELDS"

while [[ $# -gt 0 ]]; do
    case $1 in
        --threads)  THREADS="$2"; shift 2 ;;
        --assembly) ASSEMBLY="$2"; shift 2 ;;
        --buffer)   BUFFER_SIZE="$2"; shift 2 ;;
        --fields)   FIELDS="$2"; shift 2 ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ==========================================
# Path Resolution & Validation
# ==========================================
ABS_INPUT="$PROJECT_ROOT/$INPUT_VCF"
ABS_OUTPUT="$PROJECT_ROOT/$OUTPUT_VCF"

[[ -f "$ABS_INPUT" ]] || error "Input VCF not found at $ABS_INPUT"
[[ -d "$VEP_CACHE_DIR" ]] || error "VEP cache directory not found at $VEP_CACHE_DIR"

log "Project Root: $PROJECT_ROOT"
log "Mounting VEP Cache: $VEP_CACHE_DIR"

# ==========================================
# VEP Execution (Docker)
# ==========================================
log "Starting VEP annotation for ${BOLD}$ASSEMBLY${NC}..."

# We use --user to ensure output files are owned by the host user, not root.
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PROJECT_ROOT/data:/opt/data" \
    -v "$VEP_CACHE_DIR:/opt/vep/.vep" \
    "$VEP_IMAGE" \
    vep \
        -i "/opt/data/${INPUT_VCF#data/}" \
        -o "/opt/data/${OUTPUT_VCF#data/}" \
        --vcf \
        --compress_output bgzip \
        --force_overwrite \
        --offline \
        --cache \
        --dir_cache /opt/vep/.vep \
        --assembly "$ASSEMBLY" \
        --fork "$THREADS" \
        --buffer_size "$BUFFER_SIZE" \
        --fields "$FIELDS" \
        --pick \
        --pick_order mane_select,canonical,appris,tsl,biotype,rank \
        --no_stats \
        --everything

# ==========================================
# Post-Processing
# ==========================================
if [[ -f "$ABS_OUTPUT" ]]; then
    log "Indexing annotated VCF..."
    # Run tabix via docker to avoid local dependency issues
    docker run --rm -v "$PROJECT_ROOT/data:/opt/data" "$VEP_IMAGE" tabix -p vcf "/opt/data/${OUTPUT_VCF#data/}"
    log "${BOLD}VEP annotation completed successfully.${NC}"
    log "Output file: $ABS_OUTPUT"
else
    error "VEP failed to generate output file."
fi