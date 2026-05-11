/*
========================================================================================
    MODULE: INFER_VARIANTS
========================================================================================
    @Domain      : Machine Learning Inference & Deployment
    @Description : 
        A scalable inference engine that applies pre-trained genomic pathogenicity 
        models to unobserved variant features.

    Operational Design & Scalability:
        1. Artifact-Driven Scoring: Reconstructs the exact training environment by 
           loading serialized model binaries, imputers, and feature schemas.
        2. Massively Parallel Inference: Processes different model architectures 
           (RF, XGB, etc.) in isolated execution environments to prevent data leakage.
        3. Traceable Outputs: Auto-renames predictions to preserve model-to-result 
           lineage, ensuring all CSV outputs are identifiable by their source architecture.

    Resources:
        - Profiles: 'process_medium'
        - Frameworks: Leverages optimized Scikit-Learn, XGBoost, and LightGBM 
          inference engines for high-throughput variant scoring.

    Inputs:
        - model_bundle   : [tuple] A composite of (model_type, model, imputer, f_order)
                           representing the full model environment.
        - infer_features : [path] Parquet file containing variant features to be scored.

    Outputs:
        - csv_results    : [path] Pathogenicity scores named by model architecture
                           (e.g., xgb_predictions.csv).
========================================================================================
*/

process INFER_VARIANTS {
    tag "Predict: ${model_type}"
    label 'process_medium'
    container 'wgs-varml:latest'
    
    // Results are published to a dedicated predictions sub-folder for better organization
    publishDir "${params.outdir}", mode: params.publish_dir_mode

    input:
    tuple val(model_type), path(model), path(imputer), path(f_order)
    path infer_features

    output:
    path "${model_type}_predictions.csv",        emit: infer_results
    path "${model_type}.prediction_report.html", emit: infer_report

    script:
    """
    # Execution Note:
    # The --model_dir is set to '.' because Nextflow stages all input path 
    # components directly into the task's top-level working directory.
    
    08_model_inference \
        ${infer_features} \
        . \
        --outdir .
    
    # Post-processing:
    # Rename the script's generic output to a model-specific name to prevent
    # collision and ensure traceability in the final results directory.
    
    mv *predictions.csv ${model_type}_predictions.csv
    mv *prediction_report.html ${model_type}.prediction_report.html
    """
}