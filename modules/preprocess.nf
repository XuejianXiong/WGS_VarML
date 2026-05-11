/*
========================================================================================
    MODULE: PREPROCESS_VCF
========================================================================================
    @Domain      : Genomic Variant Stabilization
    @Description : 
        Standardizes raw VCF inputs into a consistent format suitable for Machine 
        Learning feature extraction and VEP annotation. This module mitigates 
        "representation drift" where the same variant might be described differently 
        across different datasets.

    Biological & Technical Rationale:
        1. INDEL Normalization: Left-aligns and parsimonizes indels against the 
           GRCh38 reference to ensure exact matches in ML feature lookups.
        2. Contig Filtering  : Removes non-canonical scaffolds (alt/decoy) to 
           reduce noise and focus on high-confidence primary genomic regions.
        3. Multiallelic Split: Decomposes multiallelic sites into individual 
           records to prevent information loss during downstream vectorization.

    Resources:
        - Profiles: 'process_medium' (Recommended 4-8 CPUs, 16-32GB RAM)
        - Latency : ~10-20 min for ClinVar; 1-2 hours for deep WGS (30x).

    Inputs:
        - vcf : [path] Raw Variant Call Format file (.vcf or .vcf.gz).
        - tbi : [path] Tabix index for the input VCF.
        - ref : [path] Reference genome FASTA (can be compressed or uncompressed).

    Outputs:
        - norm_vcf  : [path] Final processed VCF ready for annotation.
        - ref_fasta : [path] Standardized, decompressed FASTA for tool indexing.

    Compliance & Traceability:
        - Files are published to '${params.datadir}' with cross-session resume capability.
        - Outputs include consistent naming conventions for downstream pipe connectivity.
========================================================================================
*/

process PREPROCESS_VCF {
    tag "Preprocess: ${vcf.baseName}"
    label 'process_medium'
    
    // Directives: Organization of mission-critical data artifacts
    publishDir "${params.datadir}/processed", 
        mode: params.publish_dir_mode, 
        pattern: "*.norm.vcf.gz"
        
    publishDir "${params.datadir}/reference", 
        mode: params.publish_dir_mode, 
        pattern: "*.fa"

    input:
    path vcf
    path tbi
    path ref

    output:
    path "clinvar.norm.vcf.gz", emit: norm_vcf
    path "GRCh38.fa",           emit: ref_fasta

    script:
    /*
    External Script Dependencies:
        - 02_preprocess_clinvar.sh: Encapsulates bcftools and vt logic.
        - Target Directory (.): Outputs directly to the task's work sandbox 
          to ensure proper Nextflow staging and absolute path resolution.
    */
    """
    bash ${projectDir}/scripts/02_preprocess_clinvar.sh \\
        ${vcf} \\
        ${ref} \\
        .
    """
}