/*
========================================================================================
    MODULE: RUN_VEP (Variant Effect Predictor)
========================================================================================
    @Domain      : Variant Functional Annotation
    @Description : 
        Enriches genomic variants with functional consequences (e.g., missense, 
        stop_gained) and predictive scores (SIFT, PolyPhen, CADD). This module 
        acts as the primary feature-engineering engine for the ML pipeline.

    Operational Design & Performance:
        1. Symbolic Link Resolution: Utilizes 'realpath' to bridge the gap between 
           Nextflow's symlinked work directory and Docker's volume mounting 
           requirements (crucial for macOS/Unix compatibility).
        2. Zero-Copy I/O: By avoiding 'stageInMode copy', the module preserves 
           disk space and reduces latency when handling high-depth WGS datasets.
        3. Scalability: Assigned to 'process_high' to utilize multi-threading 
           capabilities for high-throughput annotation.

    Resources:
        - Profiles: 'process_high' (Recommended 16+ CPUs, 64GB+ RAM)
        - Cache   : Requires local VEP cache (mapped within the wrapper script).
        - Latency : ~30-60 min for ClinVar; 4-12 hours for whole genomes.

    Inputs:
        - norm_vcf  : [path] Normalized VCF from PREPROCESS_VCF.
        - ref_fasta : [path] Standardized GRCh38 FASTA.
        - out_name  : [val] Target filename for the annotated output.

    Outputs:
        - vep_vcf : [path] Gzipped VCF containing rich CSQ (Consequence) fields.
        - vep_tbi : [path] Tabix index for efficient downstream feature queries.

    Compliance & Traceability:
        - Results are persisted to '${params.datadir}/processed' for auditability.
        - Execution logs capture Docker container IDs and VEP versioning.
========================================================================================
*/

process RUN_VEP {
    tag "Annotate: ${norm_vcf.baseName}"
    label 'process_high'
    
    // Directives: Artifact persistence strategy
    publishDir "${params.datadir}/processed", mode: params.publish_dir_mode

    input:
    path norm_vcf
    path ref_fasta
    val  out_name

    output:
    path "${out_name}",     emit: vep_vcf
    path "${out_name}.tbi", emit: vep_tbi

    script:
    /*
    External Script Dependencies:
        - 03_run_vep_docker.sh: Handles Docker lifecycle and volume mounting.
    
    Technical Note: 
        We resolve ABS_VCF via realpath to ensure the Docker daemon on the host 
        machine can access the physical file, bypassing the Nextflow symlink 
        which is often inaccessible to containerized processes.
    */
    """
    ABS_VCF=\$(realpath ${norm_vcf})

    bash ${projectDir}/../scripts/03_run_vep_docker.sh \\
        "\$ABS_VCF" \\
        ${out_name} \\
        --threads ${task.cpus}
    """
}