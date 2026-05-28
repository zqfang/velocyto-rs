//! Translated from velocyto/transcript_model.py

use crate::constants::LONGEST_INTRON_ALLOWED;
use crate::feature::Feature;
use crate::read::Read;

/// A transcript model as a list of Feature objects (exons and introns).
/// Built from GTF annotations.
#[derive(Clone)]
pub struct TranscriptModel {
    pub trid: String,
    pub trname: String,
    pub geneid: String,
    pub genename: String,
    pub chromstrand: String,
    pub list_features: Vec<Feature>,
}

impl TranscriptModel {
    /// Creates a new TranscriptModel with the given gene ID, transcript ID, and strand.
    pub fn new(
        trid: String,
        trname: String,
        geneid: String,
        genename: String,
        chromstrand: String,
    ) -> Self {
        TranscriptModel {
            trid,
            trname,
            geneid,
            genename,
            chromstrand,
            list_features: Vec::new(),
        }
    }

    /// Returns the start of the transcript model. NOTE: Only valid after all exons have been appended via append_exon.
    pub fn start(&self) -> i64 {
        self.list_features[0].start
    }

    /// Returns the end of the transcript model. NOTE: Only valid after all exons have been appended via append_exon.
    pub fn end(&self) -> i64 {
        self.list_features[self.list_features.len() - 1].end
    }

    /// Returns true when the transcript model ends before the given position.
    pub fn ends_upstream_of(&self, read: &Read) -> bool {
        self.list_features[self.list_features.len() - 1].end < read.pos
    }

    /// The first feature starts upstream of the segment end
    pub fn starts_upstream_of(&self, segment: (i64, i64)) -> bool {
        self.list_features[0].start < segment.1
    }

    /// Returns true when the transcript model overlaps with the given interval.
    pub fn intersects(&self, segment: (i64, i64), minimum_flanking: i64) -> bool {
        (segment.1 - minimum_flanking > self.start()) && (segment.0 + minimum_flanking < self.end())
    }

    /// Append an exon and create an intron between the previous exon and this one.
    /// Python: exon_feature.transcript_model = self (we track via transcript_model_idx instead)
    pub fn append_exon(&mut self, mut exon_feature: Feature) {
        if self.list_features.is_empty() {
            self.list_features.push(exon_feature);
        } else {
            let intron_number = if self.chromstrand.ends_with('+') {
                self.list_features[self.list_features.len() - 1].exin_no
            } else {
                self.list_features[self.list_features.len() - 1].exin_no - 1
            };
            let intron_start = self.list_features[self.list_features.len() - 1].end + 1;
            let intron_end = exon_feature.start - 1;
            let intron = Feature::new(
                intron_start,
                intron_end,
                b'i',
                intron_number,
                exon_feature.transcript_model_idx,
            );
            self.list_features.push(intron);
            self.list_features.push(exon_feature);
        }
    }

    /// Modify the transcript model by removing features upstream of a very long intron (strand +)
    /// or downstream of it (strand -).
    pub fn chop_if_long_intron(&mut self, maxlen: i64) {
        // Find long introns (len > maxlen and kind == b'i')
        let long_feat_indices: Vec<usize> = self
            .list_features
            .iter()
            .enumerate()
            .filter(|(_, f)| f.len() > maxlen && f.kind == b'i')
            .map(|(i, _)| i)
            .collect();

        if long_feat_indices.is_empty() {
            return;
        }

        if self.chromstrand.ends_with('+') {
            // Use last long intron
            let idx = *long_feat_indices.last().unwrap();
            self.remove_upstream_of(idx);
        } else {
            // Use first long intron
            let idx = long_feat_indices[0];
            self.remove_downstream_of(idx);
        }
        self.trid.push_str("_mod");
        self.trname.push_str("_mod");
    }

    /// Keep only features after (greater than) the feature at longest_feat_idx,
    /// reindexing exin_no from 1.
    fn remove_upstream_of(&mut self, longest_feat_idx: usize) {
        let mut tmp: Vec<Feature> = Vec::new();
        let mut ec: i64 = 1;
        let mut ic: i64 = 1;
        // We need to compare by position: feat > longest_feat means feat.start > longest_feat.start
        // (or equal start with larger end). Use Ord.
        let pivot_start = self.list_features[longest_feat_idx].start;
        let pivot_end = self.list_features[longest_feat_idx].end;
        for feat in self.list_features.drain(..) {
            // feat > longest_feat
            if feat.start > pivot_start || (feat.start == pivot_start && feat.end > pivot_end) {
                let mut f = feat;
                if f.kind == b'e' {
                    f.exin_no = ec;
                    ec += 1;
                    tmp.push(f);
                } else if f.kind == b'i' {
                    f.exin_no = ic;
                    ic += 1;
                    tmp.push(f);
                }
            }
        }
        self.list_features = tmp;
    }

    /// Keep only features before (less than) the feature at longest_feat_idx,
    /// iterating in reverse, reindexing from 1, then reversing back.
    fn remove_downstream_of(&mut self, longest_feat_idx: usize) {
        let mut tmp: Vec<Feature> = Vec::new();
        let mut ec: i64 = 1;
        let mut ic: i64 = 1;
        let pivot_start = self.list_features[longest_feat_idx].start;
        let pivot_end = self.list_features[longest_feat_idx].end;
        // iterate in reverse
        for feat in self.list_features.drain(..).rev() {
            // feat < longest_feat
            if feat.start < pivot_start || (feat.start == pivot_start && feat.end < pivot_end) {
                let mut f = feat;
                if f.kind == b'e' {
                    f.exin_no = ec;
                    ec += 1;
                    tmp.push(f);
                } else if f.kind == b'i' {
                    f.exin_no = ic;
                    ic += 1;
                    tmp.push(f);
                }
            }
        }
        tmp.reverse();
        self.list_features = tmp;
    }
}

impl PartialEq for TranscriptModel {
    fn eq(&self, other: &Self) -> bool {
        !self.list_features.is_empty()
            && !other.list_features.is_empty()
            && self.list_features[0].start == other.list_features[0].start
    }
}

impl PartialOrd for TranscriptModel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.list_features.is_empty() || other.list_features.is_empty() {
            return None;
        }
        Some(
            self.list_features[0]
                .start
                .cmp(&other.list_features[0].start),
        )
    }
}

impl std::fmt::Display for TranscriptModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let list_feats: Vec<String> = self
            .list_features
            .iter()
            .map(|feat| format!("{}{}", feat.kind as char, feat.exin_no))
            .collect();
        write!(f, "<TrMod {}\t{}>", self.trid, list_feats.join("-"))
    }
}
