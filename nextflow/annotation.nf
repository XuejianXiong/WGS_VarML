/*
 * Process: RUN_VEP
 * Purpose: Functional annotation using Ensembl VEP via Docker.
 */
process RUN_VEP {
    tag "Annotate: ${norm_vcf.baseName}"
    label 'process_high'
    container 'ensemblorg/ensembl-vep:latest'
    
    // Save the final annotated VCF to the results folder
    publishDir "${params.outdir}/annotated", mode: params.publish_dir_mode

    input:
    path norm_vcf
    path ref_fasta

    output:
    path "clinvar.vep.vcf.gz", emit: vep_vcf
    path "clinvar.vep.vcf.gz.tbi", emit: vep_tbi

    script:
    // We use the raw project script to maintain the complex VEP flag logic
    """
    bash ${projectDir}/scripts/03_run_vep_docker.sh \\
        $norm_vcf \\
        $ref_fasta \\
        . \\
        --threads ${task.cpus}

    # Move output to top-level for Nextflow tracking
    mv data/processed/clinvar.vep.vcf.gz .
    mv data/processed/clinvar.vep.vcf.gz.tbi .
    """
}