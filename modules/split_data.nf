/*
========================================================================================
    MODULE: SPLIT_DATA
========================================================================================
    @Domain      : Experimental Design & Leakage Prevention
    @Description : 
        Partitions the feature matrix into Training, Testing, and Inference subsets 
        using deterministic SHA-256 hashing. This ensures that the data split is 
        stable across different runs and environments while strictly preventing 
        information leakage between datasets.

    Operational Design & Integrity:
        1. Deterministic Partitioning: Replaces random sampling with hash-based 
           assignment, ensuring that re-running the pipeline on the same data 
           yields identical splits—a prerequisite for ML reproducibility.
        2. Leakage Mitigation: Supports 'group-by' strategies (e.g., Chromosome or 
           Gene). By grouping by 'chr', the module ensures the model is evaluated 
           on genomic contexts it has never encountered during training.
        3. Inference Isolation: Generates a 'clean' inference set (labels removed) 
           to simulate a real-world production environment, alongside an 'audit' 
           version for ground-truth validation.

    Resources:
        - Profiles: 'process_medium' (Memory requirement scales with DataFrame size)
        - Latency : ~5-10 minutes for large-scale Parquet matrices.

    Inputs:
        - feature_matrix : [path] The QC'd Parquet dataset from FEATURE_EXTRACTION.
        - group_by       : [val] The grouping level ('variant', 'gene', or 'chr').

    Outputs:
        - train_set   : [path] 70% subset for model optimization.
        - test_set    : [path] 15% subset for unbiased performance metrics.
        - infer_set   : [path] 15% subset (features only) for production simulation.
        - infer_audit : [path] Reference file containing features + labels for scoring.

    Compliance & Traceability:
        - Splits are persisted to '${params.datadir}/splits' for model training handoff.
        - The use of 'chr' grouping satisfies the "novel genomic context" requirement 
          common in high-impact bioinformatics publications.
========================================================================================
*/

process SPLIT_DATA {
    tag "Split: ${feature_matrix.baseName} [by ${group_by}]"
    label 'process_medium'
    container 'wgs-varml:latest'
    
    // Persistence: Centralized storage for model training stage
    publishDir "${params.datadir}/splits", mode: params.publish_dir_mode

    input:
    path feature_matrix
    val  group_by

    output:
    path "*.train.parquet",              emit: train_set
    path "*.test.parquet",               emit: test_set
    path "*.infer.parquet",              emit: infer_set
    path "*.infer_with_labels.parquet",  emit: infer_audit

    script:
    /*
    Execution Logic:
    1. Deterministic Engine: Invokes src/06_split_clinvar.py.
    2. Zero-Leakage: The --group-by flag is passed directly to the Python hashing engine.
    3. Sandbox Safety: Outputs are generated in the current work directory (.) to 
       be managed by Nextflow's publishing and caching system.
    */
    """
    06_split_clinvar \\
        --input ${feature_matrix} \\
        --group-by ${group_by} \\
        --outdir .
    """
}