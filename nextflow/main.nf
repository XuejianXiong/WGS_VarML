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
include { PREPROCESS_VCF     } from './preprocess.nf'
include { RUN_VEP            } from './annotation.nf' 
include { FEATURE_EXTRACTION } from './feature.nf'
include { QC_FEATURES        } from './qc_feature.nf'
include { SPLIT_DATA         } from './split_data.nf'
include { TRAIN_MODEL        } from './train.nf'
include { INFER_VARIANTS     } from './inference.nf'

workflow {
    
    // Professional logging utilizing parameters from input.json
    log.info """
    ========================================================
    WGS_VarML Pipeline : ${params.project_name}
    ========================================================
    Input VCF        : ${params.input_vcf}
    Reference Fasta  : ${params.ref_fasta}
    Config YAML      : ${params.config_yaml}
    --------------------------------------------------------
    """

    // 1. Channel Creation
    // Sourced directly from keys defined in input.json
    ch_input_vcf = Channel.fromPath(params.input_vcf, checkIfExists: true)
    ch_input_tbi = Channel.fromPath("${params.input_vcf}.tbi", checkIfExists: true)
    ch_ref_fasta = Channel.fromPath(params.ref_fasta, checkIfExists: true)
    
    // Resolved from 'config_yaml' key in JSON
    ch_config    = Channel.fromPath(params.config_yaml, checkIfExists: true)
    
    // For isolated feature extraction
    ch_vep_vcf   = Channel.fromPath(params.vep_vcf_input, checkIfExists: true)
    
    // 2. Execution Logic
    
    // Step 1: Normalize alleles and filter chromosomes
    PREPROCESS_VCF(ch_input_vcf, ch_input_tbi, ch_ref_fasta)

    // Step 2: Annotate with VEP using the unzipped reference
    RUN_VEP(
        PREPROCESS_VCF.out.norm_vcf, 
        PREPROCESS_VCF.out.ref_fasta, 
        params.vep_output_name
    )

    // Step 3: Feature Extraction (The Application Layer)
    // Currently configured to run in isolation using 'ch_vep_vcf' from input.json
    FEATURE_EXTRACTION(RUN_VEP.out.vep_vcf, ch_config)
    //FEATURE_EXTRACTION(ch_vep_vcf, ch_config)

    // Step 4: Feature QC
    // Reference the exact emit name 'feature_matrix' from your feature.nf
    QC_FEATURES(FEATURE_EXTRACTION.out.feature_matrix, ch_config)

    // Step 5: Deterministic Splitting
    // Pass 'variant', 'gene', or 'chr' from params.split_group_by
    SPLIT_DATA(
        FEATURE_EXTRACTION.out.feature_matrix, 
        params.split_group_by ?: 'variant' 
    )

    // Step 6: Parallel Model Training
    // 6.1. BROADCAST: Files that all models need (Value Channels)
    ch_train_file = SPLIT_DATA.out.train_set.first()
    ch_test_file  = SPLIT_DATA.out.test_set.first()
    //ch_train_file = Channel.fromPath(params.ch_train_file, checkIfExists: true).first()
    //ch_test_file = Channel.fromPath(params.ch_test_file, checkIfExists: true).first()
    
    ch_config = Channel.fromPath(params.config_yaml, checkIfExists: true).first()
    
    // 6.2. Define the model architectures stream 
    // Use Channel.from() -> it handles JSON arrays perfectly
    ch_model_types = Channel.from(params.model_list ?: ['rf']).flatten()

    // DEBUG LINE: This will confirm if you have 1 or 4 items
    ch_model_types.view { "Model to be trained: $it" }

    // 6.3. Train models
    TRAIN_MODEL(ch_train_file, ch_test_file, ch_config, ch_model_types)   

    // Step 7: Parallel Model Inference
    // 7.1. Join the tagged outputs into one channel: [type, model, imputer, order]
    ch_model_bundles = TRAIN_MODEL.out.model_file
        .join(TRAIN_MODEL.out.imputer_file)
        .join(TRAIN_MODEL.out.order_file)

    // 7.2. Inference Features (Value Channel)
    ch_infer_file = Channel.fromPath(params.ch_infer_file).first()

    // 7.3. Run Inference
    INFER_VARIANTS(
        ch_model_bundles,
        ch_infer_file
    ) 
}

// --- Completion Notification ---
workflow.onComplete {
    log.info ( workflow.success ? "\nPipeline Done!" : "\nPipeline Failed" )
}