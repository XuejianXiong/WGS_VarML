#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run Ensembl VEP (Docker) on a VCF file
#
# Usage:
#   scripts/03_run_vep_docker.sh <input_vcf> <output_vcf>
#
# Example:
#   scripts/03_run_vep_docker.sh \
#     data/processed/clinvar.norm.vcf.gz \
#     data/processed/clinvar.vep.vcf.gz
###############################################################################

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input_vcf> <output_vcf>"
  exit 1
fi

INPUT_VCF="$1"
OUTPUT_VCF="$2"

# Resolve project root (script can be run from anywhere)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Docker + VEP settings
VEP_IMAGE="ensemblorg/ensembl-vep"
VEP_CACHE_DIR="$HOME/.vep"
VEP_VERSION="115"
ASSEMBLY="GRCh38"
THREADS=4
BUFFER_SIZE=5000
CHR_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y,MT"

FASTA_PATH="/opt/vep/.vep/homo_sapiens/${VEP_VERSION}_${ASSEMBLY}/Homo_sapiens.${ASSEMBLY}.dna.toplevel.fa.gz"

VEP_FIELDS="Location,Allele,Consequence,Gene,Impact,SIFT,PolyPhen,CLNSIG"

# Sanity checks
[[ -f "$PROJECT_ROOT/$INPUT_VCF" ]] || { echo "Input VCF not found: $INPUT_VCF"; exit 1; }
[[ -d "$VEP_CACHE_DIR" ]] || { echo "VEP cache not found: $VEP_CACHE_DIR"; exit 1; }

echo "Running VEP Docker annotation"
echo "Input : $INPUT_VCF"
echo "Output: $OUTPUT_VCF"

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
    --buffer_size "$BUFFER_SIZE" \
    --compress_output bgzip \
    --force_overwrite \
    --chr "$CHR_LIST"

echo "VEP annotation completed successfully."
