#!/usr/bin/env bash
# ========================================================================================
# SCRIPT: 03_run_vep_docker.sh
# ========================================================================================
# @Description: Wrapper for Ensembl VEP Docker execution.
# @Usage: bash 03_run_vep_docker.sh <input_vcf> <output_name> [options]
#
# @Arguments:
#   1. INPUT_VCF (Required): Absolute path to the normalized VCF.
#   2. OUTPUT_VCF (Required): Target name for the annotated VCF.
#
# @Options:
#   --threads  : Number of CPU cores for parallel annotation (Default: 4).
#   --assembly : Genome assembly version (Default: GRCh38).
#   --buffer   : VEP buffer size for memory management (Default: 5000).
#   --fields   : Custom CSQ fields to extract (Default: $DEFAULT_FIELDS).
# ========================================================================================

set -euo pipefail
IFS=$'\n\t'

# --- ANSI Colors for Terminal Output ---
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Logging Functions ---
log()   { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: ${NC}$*"; }
warn()  { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: ${NC}$*"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ${NC}$*"; exit 1; }

trap 'error "Script interrupted by user."' INT TERM

# --- Configuration & Defaults ---
VEP_CACHE_DIR="$HOME/.vep"
VEP_IMAGE="ensemblorg/ensembl-vep:latest"
DEFAULT_FIELDS="Uploaded_variation,Location,Allele,Gene,SYMBOL,Feature,Feature_type,BIOTYPE,CANONICAL,MANE_SELECT,Consequence,IMPACT,EXON,INTRON,Protein_position,Amino_acids,Codons,SIFT,PolyPhen,DISTANCE,STRAND"

usage() {
    cat << EOF
${BOLD}Usage:${NC} $0 <input_vcf> <output_vcf> [OPTIONS]

${BOLD}Arguments:${NC}
  input_vcf        Path to input VCF (Absolute or Relative)
  output_vcf       Path for annotated output VCF

${BOLD}Options:${NC}
  --threads N      CPU cores (Default: 4)
  --assembly STR   Genome assembly (Default: GRCh38)
  --buffer N       VEP buffer size (Default: 5000)
  --fields STR     Custom comma-separated fields
EOF
    exit 1
}

[[ $# -lt 2 ]] && usage

# --- Argument Parsing ---
INPUT_VCF="$1"
OUTPUT_VCF="$2"
shift 2

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
# Path Resolution (The "Nextflow Bridge")
# ==========================================
# Resolve absolute paths to ensure Docker volume mapping works correctly
[[ "$INPUT_VCF" == /* ]] && ABS_INPUT="$INPUT_VCF" || ABS_INPUT="$(pwd)/$INPUT_VCF"
[[ "$OUTPUT_VCF" == /* ]] && ABS_OUTPUT="$OUTPUT_VCF" || ABS_OUTPUT="$(pwd)/$OUTPUT_VCF"

[[ -f "$ABS_INPUT" ]] || error "Input VCF not found at $ABS_INPUT"
[[ -d "$VEP_CACHE_DIR" ]] || error "VEP cache directory not found at $VEP_CACHE_DIR"

# Extract directories and filenames for Docker mounting
INPUT_DIR="$(dirname "$ABS_INPUT")"
OUTPUT_DIR="$(dirname "$ABS_OUTPUT")"
IN_FILE="$(basename "$ABS_INPUT")"
OUT_FILE="$(basename "$ABS_OUTPUT")"

log "Staging environment confirmed."
log "Input:  $ABS_INPUT"
log "Output: $ABS_OUTPUT"

# ==========================================
# VEP Execution (Docker)
# ==========================================
log "Launching VEP Container (${BOLD}$ASSEMBLY${NC})..."

# Note: We mount the parent directories of the input and output separately.
# This allows the script to work even if the files are in different work/ folders.
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$INPUT_DIR:/opt/input:ro" \
    -v "$OUTPUT_DIR:/opt/output:rw" \
    -v "$VEP_CACHE_DIR:/opt/vep/.vep:ro" \
    "$VEP_IMAGE" \
    vep \
        -i "/opt/input/$IN_FILE" \
        -o "/opt/output/$OUT_FILE" \
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
# Post-Processing: Tabix Indexing
# ==========================================
if [[ -f "$ABS_OUTPUT" ]]; then
    log "Indexing output VCF..."
    docker run --rm \
        -v "$OUTPUT_DIR:/opt/output:rw" \
        "$VEP_IMAGE" \
        tabix -p vcf "/opt/output/$OUT_FILE"
    
    log "${BOLD}Annotation Complete.${NC}"
else
    error "Annotation failed: Output file not detected."
fi