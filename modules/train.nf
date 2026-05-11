/*
========================================================================================
    MODULE: TRAIN_MODEL
========================================================================================
    @Domain      : Supervised Machine Learning & Model Selection
    @Description : 
        A parallelized training engine that benchmarks multiple classifier 
        architectures (RF, XGBoost, LightGBM, Logistic Regression) against 
        vectorized genomic features.

    Operational Design & Scalability:
        1. Multi-Architecture Support: Dynamically initializes classifiers based 
           on input values, allowing for side-by-side performance comparison.
        2. Production Readiness: Exports Joblib-serialized model artifacts, 
           fitted imputers, and feature schemas to ensure zero-skew inference.
        3. Comprehensive Metrics: Generates both internal validation and 
           independent hold-out test metrics to verify model generalization.

    Resources:
        - Profiles: 'process_high' 
        - Multi-threading: Leverages 'n_jobs -1' and framework-specific parallelization
          (e.g., OpenMP for LightGBM/XGBoost) based on ${task.cpus}.

    Inputs:
        - train_features : [path] Parquet file containing training partitions.
        - test_features  : [path] Parquet file containing held-out test partitions.
        - config         : [path] Centralized YAML experiment configuration.
        - model_type     : [val] Target architecture ('rf', 'xgb', 'lgbm', 'lr').

    Outputs:
        - model          : [path] Serialized model binary for deployment.
        - val_metrics    : [path] Validation set performance report.
        - test_metrics   : [path] Final independent test set performance report.
        - importance     : [path] Ranked feature contributions for model interpretability.
        - artifacts      : [path] Supplementary binaries (imputer, feature_order).
========================================================================================
*/

process TRAIN_MODEL {
    tag "Train: ${model_type} on ${train_features.baseName}"
    label 'process_high'
    container 'wgs-varml:latest'
    
    // Organizes results into model-specific subdirectories for easy comparison
    publishDir "${params.outdir}/models/${model_type}", 
        mode: params.publish_dir_mode,
        saveAs: { filename -> filename.equals(".command.log") ? "train_${model_type}.log" : filename }

    input:
    path train_features
    path test_features
    path config
    val  model_type

    output:
    tuple val(model_type), path("model.joblib"), emit: model_file
    tuple val(model_type), path("imputer.joblib"), emit: imputer_file
    tuple val(model_type), path("feature_order.txt"), emit: order_file
    path "val_metrics.csv",         emit: val_metrics
    path "test_metrics.csv",        emit: test_metrics
    path "feature_importance.csv",  emit: f_importance
    path ".command.log",            emit: log
    

    script:
    /*
    Execution Logic:
    1. Python Engine: Invokes the 07_train_model.py factory script.
    2. Test Integration: Passes the --test-set argument for true hold-out evaluation.
    3. Sandbox Strategy: All outputs are generated in the current work directory (.) 
       to allow Nextflow to manage the staging and publishing lifecycle.
    */
    """
    07_train_model \\
        ${train_features} \\
        --model-type ${model_type} \\
        --test-set ${test_features} \\
        --config ${config} \\
        --outdir .
    """
}