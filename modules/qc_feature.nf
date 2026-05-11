/*
========================================================================================
    MODULE: QC_FEATURES
========================================================================================
    @Domain      : Data Quality Assurance & Exploratory Data Analysis (EDA)
    @Description : 
        Generates an automated, self-contained quality control report for the 
        extracted feature matrix. This module identifies data drift, missingness, 
        and feature cardinality issues that could negatively impact ML model 
        convergence or lead to biased predictions.

    Operational Design & Auditing:
        1. Distribution Analysis: Visualizes feature scaling and identifies outliers 
           in continuous variables (e.g., CADD scores, allele frequencies).
        2. Label Integrity: Audits the distribution of 'clnsig_label' to detect 
           class imbalance—a critical factor for genomic pathogenicity models.
        3. Encapsulated Reporting: Produces a standalone HTML artifact using base64 
           plot embedding, ensuring reports remain portable and viewable without 
           external dependencies or web servers.

    Resources:
        - Profiles: 'process_low' (Optimized for single-thread CPU execution)
        - Latency : ~2-5 minutes; designed for rapid feedback loops.

    Inputs:
        - feature_parquet : [path] The vectorized feature matrix (Step 3 output).
        - config          : [path] Centralized YAML containing QC thresholds and 
                            feature group definitions.

    Outputs:
        - qc_report : [path] Comprehensive HTML report for stakeholder review.

    Compliance & Traceability:
        - Reports are published to '${params.outdir}' for permanent archival.
        - Provides a "human-in-the-loop" validation step before data splitting.
========================================================================================
*/

process QC_FEATURES {
    tag "QC: ${feature_parquet.baseName}"
    label 'process_low'
    container 'wgs-varml:latest'

    // Routes diagnostic artifacts to the reporting directory
    publishDir "${params.outdir}", mode: params.publish_dir_mode

    input:
    path feature_parquet
    path config

    output:
    path "*_report.html", emit: qc_report

    script:
    /*
    Execution Logic:
    1. BaseName Extraction: Maintains a consistent naming lineage from VCF to Report.
    2. Decoupled Logic: The Python engine (src/05_QC_feature.py) is kept separate 
       from the workflow logic to allow for independent testing of the QC suite.
    */
    def report_name = "${feature_parquet.baseName}_report.html"

    """
    05_QC_feature \\
        ${feature_parquet} \\
        --output ${report_name} \\
        --config ${config}
    """
}