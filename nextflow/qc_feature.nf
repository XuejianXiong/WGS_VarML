/*
========================================================================================
    MODULE: QC_FEATURES
========================================================================================
    Description: 
        Performs post-extraction Quality Control on ML feature matrices. 
        Generates an interactive HTML report to validate feature distributions, 
        missingness, and class balance prior to model training.

    Operational Design:
        - Decoupled from extraction to allow independent re-runs of visualization.
        - Utilizes Median Absolute Deviation (MAD) principles for signal validation.
        - Supports Parquet/CSV input formats.

    Inputs:
        - path feature_parquet : ML-ready feature table (e.g., .features.parquet).
        - path config          : YAML configuration containing 'qc' logic/thresholds.

    Outputs:
        - path "*_report.html" : Standalone HTML QC report with embedded base64 plots.
========================================================================================
*/

process QC_FEATURES {
    // tag: Displays the current sample/file in the Nextflow console and log files
    tag "${feature_parquet.baseName}"
    
    // label: Assigns resource requirements (CPU/Mem) defined in nextflow.config
    label 'process_low'

    // publishDir: Routes final artifacts to the results/qc directory
    // 'copy' mode ensures data persists after work/ directory cleanup
    publishDir "${params.outdir}", mode: params.publish_dir_mode

    input:
    path feature_parquet
    path config

    output:
    path "*_report.html", emit: qc_report

    script:
    /*
    Execution Logic:
    1. Extracts the base name to ensure output consistency.
    2. Invokes the Python QC engine with the centralized config.yaml.
    3. Python script handles base64 encoding of plots for a self-contained HTML.
    */
    def report_name = "${feature_parquet.baseName}_report.html"

    """
    python3 ${baseDir}/../src/05_QC_feature.py \\
        ${feature_parquet} \\
        --output ${report_name} \\
        --config ${config}
    """
}