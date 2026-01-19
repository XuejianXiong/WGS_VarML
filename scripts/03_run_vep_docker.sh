#!/usr/bin/env bash
# =============================================================================
# Run Ensembl VEP (Docker) on a VCF file (production-ready, flexible)
#
# Usage:
#   ./scripts/03_run_vep_docker.sh <input_vcf> <output_vcf> [options]
#
# Options (all optional):
#   --threads N         Number of CPU threads (default: 4)
#   --assembly STR      Genome assembly (default: GRCh38)
#   --fields STR        Comma-separated VEP fields (default: Uploaded_variation,Location,Allele,Gene,SYMBOL,Feature,Feature_type,BIOTYPE,CANONICAL,MANE_SELECT,Consequence,IMPACT,EXON,INTRON,Protein_position,Amino_acids,Codons,SIFT,PolyPhen,DISTANCE,STRAND)
#   --chr STR           Comma-separated chromosomes (default: 1-22,X,Y,MT)
#   --fasta PATH        FASTA reference file (default: based on cache + assembly)
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

log() { echo "[`date '+%Y-%m-%d %H:%M:%S'`] $*"; }

# ----------------------------
# Required positional arguments
# ----------------------------
if [[ $# -lt 2 ]]; then
    log "ERROR: At least 2 arguments required."
    log "Usage: $0 <input_vcf> <output_vcf> [--threads N] [--assembly STR] [--fields STR] [--chr STR] [--fasta PATH]"
    exit 1
fi

INPUT_VCF="$1"
OUTPUT_VCF="$2"
shift 2

# ----------------------------
# Default optional parameters
# ----------------------------
THREADS=4
ASSEMBLY="GRCh38"
VEP_FIELDS="Uploaded_variation,Location,Allele,Gene,SYMBOL,Feature,Feature_type,BIOTYPE,CANONICAL,MANE_SELECT,Consequence,IMPACT,EXON,INTRON,Protein_position,Amino_acids,Codons,SIFT,PolyPhen,DISTANCE,STRAND"
CHR_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y,MT"
FASTA_PATH=""  # will set below if not provided

VEP_CACHE_DIR="$HOME/.vep"
VEP_VERSION="115"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ----------------------------
# Parse optional arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --threads) THREADS="$2"; shift 2 ;;
        --assembly) ASSEMBLY="$2"; shift 2 ;;
        --fields) VEP_FIELDS="$2"; shift 2 ;;
        --chr) CHR_LIST="$2"; shift 2 ;;
        --fasta) FASTA_PATH="$2"; shift 2 ;;
        *) log "ERROR: Unknown option $1"; exit 1 ;;
    esac
done

# Set default FASTA if not provided
if [[ -z "$FASTA_PATH" ]]; then
    FASTA_PATH="/opt/vep/.vep/homo_sapiens/${VEP_VERSION}_${ASSEMBLY}/Homo_sapiens.${ASSEMBLY}.dna.toplevel.fa.gz"
fi

ABS_INPUT="$PROJECT_ROOT/$INPUT_VCF"
ABS_OUTPUT="$PROJECT_ROOT/$OUTPUT_VCF"

# ----------------------------
# Sanity checks
# ----------------------------
[[ "$THREADS" =~ ^[0-9]+$ ]] || { log "ERROR: --threads must be an integer"; exit 1; }
[[ -f "$ABS_INPUT" ]] || { log "ERROR: Input VCF not found: $ABS_INPUT"; exit 1; }
[[ -d "$VEP_CACHE_DIR" ]] || { log "ERROR: VEP cache not found: $VEP_CACHE_DIR"; exit 1; }

if [[ -n "$FASTA_PATH" && "$FASTA_PATH" != /opt/vep/* ]]; then
    [[ -f "$FASTA_PATH" ]] || { log "ERROR: FASTA reference not found: $FASTA_PATH"; exit 1; }
fi
log "Resolved FASTA: $(basename "$FASTA_PATH")"

if [[ -z "${VEP_FIELDS:-}" ]]; then
    log "INFO: Using default VEP field set"
else
    log "INFO: Using custom --fields overrides VEP defaults"
fi

# ----------------------------
# Docker & VEP settings
# ----------------------------
VEP_IMAGE="ensemblorg/ensembl-vep" 

# ----------------------------
# Run VEP via Docker
# ----------------------------
log "Starting VEP Docker annotation"
log "Input       : $ABS_INPUT"
log "Output      : $ABS_OUTPUT"
log "Assembly    : $ASSEMBLY"
log "Threads     : $THREADS"
log "VEP fields  : $VEP_FIELDS"
log "Chromosomes : $CHR_LIST"
log "FASTA       : $FASTA_PATH"
log "Cache       : $VEP_CACHE_DIR"

log "NOTE: MANE_SELECT requires a recent Ensembl cache; verify if missing in output."

log "Running VEP annotation..."

docker run --rm \
    -v "$PROJECT_ROOT/data:/opt/data" \
    -v "$VEP_CACHE_DIR:/opt/vep/.vep" \
    "$VEP_IMAGE" \
    vep \
        -i "/opt/data/${INPUT_VCF#data/}" \
        -o "/opt/data/${OUTPUT_VCF#data/}" \
        --vcf \
        --offline \
        --cache \
        --dir_cache /opt/vep/.vep \
        --assembly "$ASSEMBLY" \
        --fasta "$FASTA_PATH" \
        --fields "$VEP_FIELDS" \
        --fork "$THREADS" \
        --buffer_size 5000 \
        --compress_output bgzip \
        --force_overwrite \
        --chr "$CHR_LIST" \
        --pick \
        --pick_order mane_select,canonical,appris,tsl,biotype,rank \
        --no_stats


log "VEP annotation completed successfully."
log "Output file: $ABS_OUTPUT"
