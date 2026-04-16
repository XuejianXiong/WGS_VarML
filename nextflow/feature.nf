/*
========================================================================================
    Module: FEATURE_EXTRACTION
========================================================================================
    Description: 
        Parses VEP-annotated VCF files into ML-ready feature matrices (Parquet/CSV).
        This process bridges the gap between raw genomic data and model training.
    
    Inputs:
        - vcf: The annotated VCF file from the VEP process.
        - config: The project-level config.yaml containing extraction logic.
    
    Outputs:
        - feature_matrix: The processed feature table for downstream ML steps.
========================================================================================
*/

process FEATURE_EXTRACTION {
    tag "${vcf.baseName}"
    label 'process_medium' // Resource allocation defined in nextflow.config

    // Organize results based on outdir defined in input.json
    publishDir "${params.datadir}", mode: params.publish_dir_mode

    input:
    path vcf
    path config

    output:
    path "*.features.*", emit: feature_matrix
    path ".command.log", emit: log

    script:
    // Resolve dynamic flags from params (with fallbacks to config.yaml via the Python script)
    def fmt    = params.extract_format ?: 'parquet'
    def filter = params.filter_unknown ? '--filter-unknown' : ''

    """
    python3 ${baseDir}/../src/04_extract_features.py \\
        ${vcf} \\
        --config ${config} \\
        --format ${fmt} \\
        ${filter}
    """
}