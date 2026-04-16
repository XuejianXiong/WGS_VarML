/*
 * Process: PREPROCESS_VCF
 * Purpose: Normalizes variants and filters for primary chromosomes.
 */
process PREPROCESS_VCF {
    tag "Preprocess: ${vcf.baseName}"
    label 'process_medium'

    input:
    path vcf
    path ref

    output:
    path "processed/clinvar.norm.vcf.gz", emit: norm_vcf
    path "reference/GRCh38.fa",           emit: ref_fasta

    script:
    """
    # Ensure local directory structure for the script's expectations
    mkdir -p processed reference
    
    bash ${projectDir}/scripts/02_preprocess_clinvar.sh \\
        $vcf \\
        $ref \\
        ./processed

    # Standardize path for downstream tool access
    mv data/reference/GRCh38.fa reference/
    """
}