#!/usr/bin/env nextflow

/*
========================================================================================
    WGS_VarML Pipeline
========================================================================================
    Description: 
        A production-grade Nextflow DSL2 pipeline for WGS variant preprocessing, 
        VEP annotation, and ML feature extraction.

    Execution:
        nextflow run nextflow/main.nf -params-file input.json -resume

    Dependencies:
        - Nextflow (DSL2)
        - Python 3.x (with pandas, cyvcf2, pyyaml)
        - VEP (Ensembl Variant Effect Predictor)
========================================================================================
*/

nextflow.enable.dsl=2

// --- Import Modules ---
// Modules are resolved relative to this file's directory
include { PREPROCESS_VCF    } from './preprocess.nf'
include { RUN_VEP           } from './annotation.nf' 
include { FEATURE_EXTRACTION } from './feature.nf'
include { QC_FEATURES        } from './qc_feature.nf'

workflow {
    
    // Professional logging utilizing parameters from input.json
    log.info """
    ========================================================
    P R O J E C T : ${params.project_name}
    ========================================================
    Input VCF        : ${params.input_vcf}
    Reference Fasta  : ${params.ref_fasta}
    Annotated Input  : ${params.vep_vcf_input}
    Config YAML      : ${params.config_yaml}
    Results Dir      : ${params.outdir}
    Publish Mode     : ${params.publish_dir_mode}
    --------------------------------------------------------
    """

    // 1. Channel Creation
    // Sourced directly from keys defined in input.json
    ch_input_vcf = Channel.fromPath(params.input_vcf, checkIfExists: true)
    ch_ref_fasta = Channel.fromPath(params.ref_fasta, checkIfExists: true)
    
    // Resolved from 'config_yaml' key in JSON
    ch_config    = Channel.fromPath(params.config_yaml, checkIfExists: true)
    
    // Resolved from 'vep_vcf_input' key for isolated feature extraction
    ch_vep_vcf   = Channel.fromPath(params.vep_vcf_input, checkIfExists: true)

    // 2. Execution Logic
    
    // Step 1: Normalize alleles and filter chromosomes
    // (Uncomment when running the full end-to-end pipeline)
    // PREPROCESS_VCF(ch_input_vcf, ch_ref_fasta)

    // Step 2: Annotate with VEP using the unzipped reference
    // RUN_VEP(PREPROCESS_VCF.out.norm_vcf, PREPROCESS_VCF.out.ref_fasta)

    // Step 3: Feature Extraction (The Application Layer)
    // Currently configured to run in isolation using 'ch_vep_vcf' from input.json
    // FEATURE_EXTRACTION(RUN_VEP.out.vep_vcf, ch_config)
    FEATURE_EXTRACTION(ch_vep_vcf, ch_config)

    // Step 4: Feature QC
    // Reference the exact emit name 'feature_matrix' from your feature.nf
    QC_FEATURES(FEATURE_EXTRACTION.out.feature_matrix, ch_config)

}

// --- Completion Notification ---
workflow.onComplete {
    log.info """
    Pipeline execution summary
    ---------------------------
    Completed at : ${workflow.complete}
    Duration     : ${workflow.duration}
    Success      : ${workflow.success}
    WorkDir      : ${workflow.workDir}
    Exit status  : ${workflow.exitStatus}
    """
}