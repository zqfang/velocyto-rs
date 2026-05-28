//! Translated from velocyto/read.py

/// Container for a read from a SAM/BAM alignment file.
/// Stores the read's segments, strand, and barcode/UMI info.
pub struct Read {
    pub bc: String,
    pub umi: String,
    pub chrom: String,
    pub strand: char,
    pub pos: i64,
    pub segments: Vec<(i64, i64)>,
    pub clip5: Option<i64>,
    pub clip3: Option<i64>,
    pub ref_skipped: bool,
}

impl Read {
    /// Creates a Read with the given segments, strand, reference name, barcode, UMI, and optional mapping quality.
    pub fn new(
        bc: String,
        umi: String,
        chrom: String,
        strand: char,
        pos: i64,
        segments: Vec<(i64, i64)>,
        clip5: Option<i64>,
        clip3: Option<i64>,
        ref_skipped: bool,
    ) -> Self {
        Read {
            bc,
            umi,
            chrom,
            strand,
            pos,
            segments,
            clip5,
            clip3,
            ref_skipped,
        }
    }

    /// Returns true if the read has a reference skip (BAM_CREF_SKIP / CIGAR N), indicating a spliced alignment.
    pub fn is_spliced(&self) -> bool {
        self.ref_skipped
    }

    /// Returns the leftmost position of the read (start of first segment).
    pub fn start(&self) -> i64 {
        self.segments[0].0
    }

    /// Returns the rightmost position of the read (end of last segment).
    pub fn end(&self) -> i64 {
        self.segments[self.segments.len() - 1].1
    }

    /// Returns the total genomic span of the read (end - start + 1).
    pub fn span(&self) -> i64 {
        self.end() - self.start() + 1
    }
}

impl PartialEq for Read {
    fn eq(&self, other: &Self) -> bool {
        self.chrom == other.chrom && self.start() == other.start() && self.end() == other.end()
    }
}

impl PartialOrd for Read {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.chrom != other.chrom {
            return Some(self.chrom.cmp(&other.chrom));
        }
        if self.start() != other.start() {
            return Some(self.start().cmp(&other.start()));
        }
        Some(self.end().cmp(&other.end()))
    }
}

impl std::fmt::Display for Read {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Read(bc={}, umi={}, chrom={}, strand={}, pos={}, ref_skipped={})",
            self.bc, self.umi, self.chrom, self.strand, self.pos, self.ref_skipped
        )
    }
}
