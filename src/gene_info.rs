//! Translated from velocyto/gene_info.py

/// Stores basic info on a gene, parsed from the GTF file.
/// Used to build row attributes (Gene, Accession, Chromosome, Strand) in the loom output.
pub struct GeneInfo {
    pub genename: String,
    pub geneid: String,
    pub chrom: String,
    pub strand: char,
    pub start: i64,
    pub end: i64,
}

impl GeneInfo {
    /// Creates a GeneInfo from a chromstrand string (e.g. '1+') and gene name/accession.
    pub fn new(genename: String, geneid: String, chromstrand: &str, start: i64, end: i64) -> Self {
        assert!(!chromstrand.is_empty(), "chromstrand must not be empty");
        let chrom = chromstrand[..chromstrand.len() - 1].to_string();
        let strand = chromstrand.chars().last().unwrap();
        GeneInfo {
            genename,
            geneid,
            chrom,
            strand,
            start,
            end,
        }
    }
}

impl std::fmt::Display for GeneInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GeneInfo({} {} {}{}:{}-{})",
            self.genename, self.geneid, self.chrom, self.strand, self.start, self.end
        )
    }
}
