/*
========================================================================================
    MODULE: FEATURE_EXTRACTION
========================================================================================
    @Domain      : Genomic Feature Engineering & Vectorization
    @Description : 
        Parses VEP-annotated VCF files to extract and encode biological features 
        into a structured format (Parquet/CSV). This module handles the critical 
        mapping of categorical consequence strings into numerical representations 
        suitable for Machine Learning model consumption.

    Operational Design & Vectorization:
        1. CSQ Parsing: Decodes the 'CSQ' INFO field into individual feature columns 
           (e.g., SIFT, PolyPhen, BLOSUM62).
        2. Format Optimization: Defaults to 'Parquet' to maintain strict data typing 
           (float32/int8) and provide high-performance columnar I/O for large scales.
        3. Dynamic Filtering: Supports optional exclusion of variants with unknown 
           clinical significance to reduce noise in the training labels.

    Resources:
        - Profiles: 'process_medium' (Recommended 8GB+ RAM for ClinVar; scales with VCF size)
        - Storage : High-efficiency Parquet compression reduces disk footprint by >80%.

    Inputs:
        - vcf    : [path] VEP-annotated VCF from RUN_VEP.
        - config : [path] YAML configuration defining feature weights and categories.

    Outputs:
        - feature_matrix : [path] The vectorized dataset (*.features.parquet or *.csv).
        - log            : [path] Standard execution logs for data lineage auditing.

    Compliance & Traceability:
        - Maintains data lineage by tagging outputs with the source VCF basename.
        - Config-driven extraction ensures reproducible feature sets across experiments.
========================================================================================
*/

process FEATURE_EXTRACTION {
    tag "Extract: ${vcf.baseName}"
    label 'process_medium' 

    // Persistence layer: Structured as the primary input for Step 5 (Splitting)
    publishDir "${params.datadir}/processed", mode: params.publish_dir_mode

    input:
    path vcf
    path config

    output:
    path "*.features.*", emit: feature_matrix
    path ".command.log", emit: log

    script:
    /*
    Execution Logic:
    1. Parametrization: fmt and filter allow CLI-level control over the output matrix.
    2. Scalability: Uses ${baseDir} to locate the Python source relative to the project root,
       ensuring portability across different execution environments (Local, Cloud, HPC).
    */
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