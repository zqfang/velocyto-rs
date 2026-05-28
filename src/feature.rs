//! Translated from velocyto/feature.py

use crate::read::Read;
use crate::transcript_model::TranscriptModel;

/// A genomic feature (exon, intron, or masked repeat) from a transcript model.
/// `kind` is b'e' for exon, b'i' for intron, b'm' for masked.
#[derive(Clone)]
pub struct Feature {
    pub start: i64,
    pub end: i64,
    pub kind: u8, // b'e'=101, b'i'=105, b'm'=109
    pub exin_no: i64,
    pub is_validated: bool,
    pub transcript_model_idx: Option<usize>,
}

impl Feature {
    /// Creates a Feature. `kind` should be b'e', b'i', or b'm'.
    /// `exin_no` is the exon/intron number within the transcript model.
    pub fn new(
        start: i64,
        end: i64,
        kind: u8,
        exin_no: i64,
        transcript_model_idx: Option<usize>,
    ) -> Self {
        Feature {
            start,
            end,
            kind,
            exin_no,
            is_validated: false,
            transcript_model_idx,
        }
    }

    /// Python __len__: (self.end - self.start) + 1
    pub fn len(&self) -> i64 {
        (self.end - self.start) + 1
    }

    /// Returns true if this feature is the last (3'-most) feature in its transcript model.
    pub fn is_last_3prime(&self, tm: &TranscriptModel) -> bool {
        if tm.chromstrand.ends_with('+') {
            // last element
            if let Some(last) = tm.list_features.last() {
                std::ptr::eq(self as *const Feature, last as *const Feature)
                    || (self.start == last.start && self.end == last.end)
            } else {
                false
            }
        } else {
            // first element
            if let Some(first) = tm.list_features.first() {
                std::ptr::eq(self as *const Feature, first as *const Feature)
                    || (self.start == first.start && self.end == first.end)
            } else {
                false
            }
        }
    }

    /// Returns the index of the downstream exon in the transcript model's feature list.
    pub fn get_downstream_exon_idx(&self, tm: &TranscriptModel) -> usize {
        if tm.chromstrand.ends_with('+') {
            (self.exin_no * 2) as usize
        } else {
            (tm.list_features.len() as i64 - 2 * self.exin_no + 1) as usize
        }
    }

    /// Returns the index of the upstream exon in the transcript model's feature list.
    pub fn get_upstream_exon_idx(&self, tm: &TranscriptModel) -> usize {
        if tm.chromstrand.ends_with('+') {
            ((self.exin_no * 2) - 2) as usize
        } else {
            (tm.list_features.len() as i64 - 2 * self.exin_no - 1) as usize
        }
    }

    /// Returns true when this feature ends before `read.pos` (the feature is entirely upstream of the read start).
    pub fn ends_upstream_of(&self, read: &Read) -> bool {
        self.end < read.pos
    }

    /// Returns true when this feature does not start after the read end (needed for overlap detection).
    pub fn doesnt_start_after(&self, segment: (i64, i64)) -> bool {
        self.start < segment.1
    }

    /// Returns true when this feature overlaps with the given segment.
    pub fn intersects(&self, segment: (i64, i64), minimum_flanking: i64) -> bool {
        (segment.1 - minimum_flanking > self.start) && (segment.0 + minimum_flanking < self.end)
    }

    /// Returns true when this feature fully contains the given segment.
    pub fn contains(&self, segment: (i64, i64), minimum_flanking: i64) -> bool {
        (segment.0 + minimum_flanking >= self.start)
            && (segment.1 - minimum_flanking <= self.end)
            && (segment.1 - segment.0 > minimum_flanking)
    }

    /// Returns true when the feature start partially overlaps the segment, with at least `minimum_flanking` bases of overlap.
    pub fn start_overlaps_with_part_of(&self, segment: (i64, i64), minimum_flanking: i64) -> bool {
        (segment.0 + minimum_flanking < self.start) && (segment.1 - minimum_flanking > self.start)
    }

    /// Returns true when the feature end partially overlaps the segment, with at least `minimum_flanking` bases of overlap.
    pub fn end_overlaps_with_part_of(&self, segment: (i64, i64), minimum_flanking: i64) -> bool {
        (segment.0 + minimum_flanking < self.end) && (segment.1 - minimum_flanking > self.end)
    }
}

impl PartialEq for Feature {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}
impl Eq for Feature {}

impl PartialOrd for Feature {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Feature {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.start == other.start {
            self.end.cmp(&other.end)
        } else {
            self.start.cmp(&other.start)
        }
    }
}

impl std::fmt::Display for Feature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Feature: {}-{} {}{}",
            self.start, self.end, self.kind as char, self.exin_no
        )
    }
}
