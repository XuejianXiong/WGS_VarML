#!/usr/bin/env nextflow

/*
========================================================================================
    WGS_VarML Pipeline
========================================================================================
    Author: Xuejian
    Description: End-to-end variant preprocessing and ML annotation.
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl=2

// --- Import Modules ---
include { PREPROCESS_VCF } from './preprocess.nf'
include { RUN_VEP        } from './annotate.nf'

// --- Pipeline Workflow ---
workflow {
    
    // 1. Initial Checks & Channel Creation
    log.info """
    ===========================================
    W G S - V a r M L   P I P E L I N E
    ===========================================
    Input VCF    : ${params.input_vcf}
    Reference    : ${params.ref_fasta}
    Results Dir  : ${params.outdir}
    Profile      : ${workflow.profile}
    -------------------------------------------
    """

    ch_input_vcf = Channel.fromPath(params.input_vcf, checkIfExists: true)
    ch_ref_fasta = Channel.fromPath(params.ref_fasta, checkIfExists: true)

    // 2. Execution Logic
    // Step 1: Normalize alleles and filter chromosomes
    PREPROCESS_VCF(ch_input_vcf, ch_ref_fasta)

    // Step 2: Annotate with VEP using the unzipped reference from Step 1
    RUN_VEP(PREPROCESS_VCF.out.norm_vcf, PREPROCESS_VCF.out.ref_fasta)

    // Future: Step 3 (Feature Extraction)
    // FEATURE_EXTRACTION(RUN_VEP.out.vep_vcf)
}

// --- Completion Notification ---
workflow.onComplete {
    log.info "Pipeline completed at: ${workflow.complete}"
    log.info "Execution status: ${workflow.success ? 'OK' : 'failed'}"
}