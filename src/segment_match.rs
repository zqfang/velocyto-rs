//! Translated from velocyto/segment_match.py

use crate::constants::SPLIC_INACUR;
use crate::feature::Feature;

/// Represents a match between a read segment and a genomic feature.
/// `is_spliced` corresponds to BAM_CREF_SKIP (N in CIGAR).
pub struct SegmentMatch {
    pub segment: (i64, i64),
    pub feature_idx: usize, // index into FeatureIndex.ivls
    pub is_spliced: bool,
    pub feature: Feature, // clone of the matched Feature (mirrors Python's self.feature ref)
}

impl SegmentMatch {
    /// Creates a SegmentMatch with the given segment coordinates, feature, and splice status.
    pub fn new(
        segment: (i64, i64),
        feature_idx: usize,
        is_spliced: bool,
        feature: Feature,
    ) -> Self {
        SegmentMatch {
            segment,
            feature_idx,
            is_spliced,
            feature,
        }
    }

    /// Returns true if this match is to an intronic feature (kind == b'i').
    pub fn maps_to_intron(&self) -> bool {
        self.feature.kind == b'i'
    }

    /// Returns true if this match is to an exonic feature (kind == b'e').
    pub fn maps_to_exon(&self) -> bool {
        self.feature.kind == b'e'
    }

    /// skip_makes_sense: if not spliced return true;
    /// else check if the segment boundary aligns with the feature boundary within SPLIC_INACUR.
    pub fn skip_makes_sense(&self) -> bool {
        if !self.is_spliced {
            return true;
        }
        (self.feature.start - self.segment.0).abs() <= SPLIC_INACUR
            || (self.feature.end - self.segment.1).abs() <= SPLIC_INACUR
    }
}

impl std::fmt::Display for SegmentMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<SegmentMatch segment={}-{} feature_idx={} spliced={}>",
            self.segment.0, self.segment.1, self.feature_idx, self.is_spliced
        )
    }
}
