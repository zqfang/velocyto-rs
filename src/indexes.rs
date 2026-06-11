//! Translated from velocyto/indexes.py

use crate::constants::{MATCH_INSIDE, MATCH_OVER3END, MATCH_OVER5END, MIN_FLANK};
use crate::feature::Feature; // used in ivls Vec<Feature> and mark_overlapping_ivls
use crate::read::Read;
use crate::segment_match::SegmentMatch;
use crate::transcript_model::TranscriptModel;
use std::collections::{HashMap, HashSet};

pub struct TranscriptsIndex {
    pub transcript_models: Vec<TranscriptModel>,
    pub tidx: usize,
    pub maxtidx: usize,
}

impl TranscriptsIndex {
    /// Creates a TranscriptsIndex from a list of TranscriptModel objects.
    /// Initializes `tidx = 0` and `maxtidx = len - 1`.
    pub fn new(transcript_models: Vec<TranscriptModel>) -> Self {
        let maxtidx = if transcript_models.is_empty() {
            0
        } else {
            transcript_models.len() - 1
        };
        TranscriptsIndex {
            transcript_models,
            tidx: 0,
            maxtidx,
        }
    }

    /// Returns true while there are transcript models left to scan (tidx < maxtidx).
    pub fn scan_not_terminated(&self) -> bool {
        self.tidx < self.maxtidx
    }

    /// Returns a set of indices into self.transcript_models that overlap with the read.
    pub fn find_overlapping_transcript_models(&mut self, read: &Read) -> HashSet<usize> {
        let mut matched: HashSet<usize> = HashSet::new();
        if self.transcript_models.is_empty() {
            return matched;
        }

        // Advance tidx while current model ends upstream of read
        while self.scan_not_terminated() && self.transcript_models[self.tidx].ends_upstream_of(read)
        {
            self.tidx += 1;
        }

        // Python carries `tmodel` across segments: only `i` is reset to `self.tidx` at the
        // start of each segment's inner loop, but `tmodel` retains the position it reached
        // at the end of the previous segment's inner loop.
        //
        // Concretely, `tmodel` is used for both the while-condition check and the
        // intersects check; `i` is incremented and used to update `tmodel` at the end of
        // each loop body.  After a segment's inner loop, `tmodel` stays at index `i`
        // (the post-increment position), and the next segment's loop starts with `i`
        // reset to `self.tidx` but `tmodel` still at the carried-over position.
        //
        // `tmodel_i` mirrors exactly where `tmodel` is pointing across segments.
        let mut tmodel_i = self.tidx; // Python: tmodel = self.transcipt_models[self.tidx]
        for segment in &read.segments {
            let mut i = self.tidx; // Python: i = self.tidx (reset per segment)
                                   // while condition uses tmodel (at tmodel_i), body checks tmodel then advances i
                                   // and reassigns tmodel = self.transcipt_models[i]
            while i < self.maxtidx && self.transcript_models[tmodel_i].starts_upstream_of(*segment)
            {
                if self.transcript_models[tmodel_i].intersects(*segment, MIN_FLANK) {
                    matched.insert(tmodel_i);
                }
                i += 1;
                tmodel_i = i; // Python: tmodel = self.transcipt_models[i]
            }
            // tmodel_i carries its value (wherever i reached) into the next segment's loop
        }
        matched
    }
}

pub struct FeatureIndex {
    pub ivls: Vec<Feature>,
    pub iidx: usize,
    pub maxiidx: usize,
}

impl FeatureIndex {
    /// Creates a FeatureIndex from a list of Feature objects. Sorts the intervals on construction.
    pub fn new(mut ivls: Vec<Feature>) -> Self {
        ivls.sort();
        let maxiidx = if ivls.is_empty() { 0 } else { ivls.len() - 1 };
        FeatureIndex {
            ivls,
            iidx: 0,
            maxiidx,
        }
    }

    /// Returns true while there are still intervals to scan (iidx < maxiidx).
    pub fn last_interval_not_reached(&self) -> bool {
        self.iidx < self.maxiidx
    }

    /// Resets the current feature pointer back to the first feature (iidx = 0).
    pub fn reset(&mut self) {
        self.iidx = 0;
    }

    /// Returns true if all segments are MATCH_INSIDE some interval.
    pub fn has_ivls_enclosing(&mut self, read: &Read) -> bool {
        if self.ivls.is_empty() {
            return false;
        }

        // Advance iidx past intervals that end before the read starts
        while self.last_interval_not_reached() && self.ivls[self.iidx].ends_upstream_of(read) {
            self.iidx += 1;
        }

        for segment in &read.segments {
            let mut segment_matchtype: u8 = 0;
            let mut i = self.iidx;
            while i < self.maxiidx && self.ivls[i].doesnt_start_after(*segment) {
                let mut matchtype: u8 = 0;
                if self.ivls[i].contains(*segment, MIN_FLANK) {
                    matchtype = MATCH_INSIDE;
                }
                if self.ivls[i].start_overlaps_with_part_of(*segment, MIN_FLANK) {
                    matchtype |= MATCH_OVER5END;
                }
                if self.ivls[i].end_overlaps_with_part_of(*segment, MIN_FLANK) {
                    matchtype |= MATCH_OVER3END;
                }
                segment_matchtype |= matchtype;
                i += 1;
            }
            // If segment_matchtype ^ MATCH_INSIDE is nonzero, segment is not purely inside
            if segment_matchtype ^ MATCH_INSIDE != 0 {
                return false;
            }
        }
        true
    }

    /// Mark intronic features is_validated = true if a splicing read spans the intron-exon boundary.
    pub fn mark_overlapping_ivls(&mut self, read: &Read, tms: &[TranscriptModel]) {
        if self.ivls.is_empty() {
            return;
        }

        while self.last_interval_not_reached() && self.ivls[self.iidx].ends_upstream_of(read) {
            self.iidx += 1;
        }

        for (_n_seg, segment) in read.segments.iter().enumerate() {
            let mut i = self.iidx;
            while i < self.maxiidx && self.ivls[i].doesnt_start_after(*segment) {
                if self.ivls[i].kind == b'i' {
                    // Check end_overlaps_with_part_of: intron end overlaps segment -> downstream exon
                    let end_overlaps = self.ivls[i].end_overlaps_with_part_of(*segment, MIN_FLANK);
                    let start_overlaps =
                        self.ivls[i].start_overlaps_with_part_of(*segment, MIN_FLANK);

                    if end_overlaps {
                        // get downstream exon from transcript model
                        if let Some(tm_idx) = self.ivls[i].transcript_model_idx {
                            if tm_idx < tms.len() {
                                let tm = &tms[tm_idx];
                                let ds_idx = self.ivls[i].get_downstream_exon_idx(tm);
                                if ds_idx < tm.list_features.len() {
                                    let downstream_exon = &tm.list_features[ds_idx];
                                    if downstream_exon
                                        .start_overlaps_with_part_of(*segment, MIN_FLANK)
                                    {
                                        self.ivls[i].is_validated = true;
                                    }
                                }
                            }
                        }
                    }
                    if start_overlaps {
                        // get upstream exon from transcript model
                        if let Some(tm_idx) = self.ivls[i].transcript_model_idx {
                            if tm_idx < tms.len() {
                                let tm = &tms[tm_idx];
                                let us_idx = self.ivls[i].get_upstream_exon_idx(tm);
                                if us_idx < tm.list_features.len() {
                                    let upstream_exon = &tm.list_features[us_idx];
                                    if upstream_exon.end_overlaps_with_part_of(*segment, MIN_FLANK)
                                    {
                                        self.ivls[i].is_validated = true;
                                    }
                                }
                            }
                        }
                    }
                } else if self.ivls[i].kind != b'e' {
                    panic!(
                        "Unrecognized type of genomic feature '{}'",
                        self.ivls[i].kind as char
                    );
                }
                i += 1;
            }
        }
    }

    /// Find overlapping intervals and return a mapping record keyed by transcript_model_idx.
    /// Post-processes: keep only max-segment-count TMs, then remove any where a SKIP doesn't make sense.
    pub fn find_overlapping_ivls(&mut self, read: &Read) -> HashMap<usize, Vec<SegmentMatch>> {
        let mut mapping_record: HashMap<usize, Vec<SegmentMatch>> = HashMap::new();

        if self.ivls.is_empty() {
            return mapping_record;
        }

        while self.last_interval_not_reached() && self.ivls[self.iidx].ends_upstream_of(read) {
            self.iidx += 1;
        }

        for segment in read.segments.iter() {
            let mut i = self.iidx;
            while i < self.maxiidx && self.ivls[i].doesnt_start_after(*segment) {
                let seg_len = segment.1 - segment.0;
                if self.ivls[i].intersects(*segment, MIN_FLANK) && seg_len > MIN_FLANK {
                    if let Some(tm_idx) = self.ivls[i].transcript_model_idx {
                        mapping_record
                            .entry(tm_idx)
                            .or_default()
                            .push(SegmentMatch::new(
                                *segment,
                                i,
                                read.is_spliced(),
                                self.ivls[i].clone(),
                            ));
                    }
                }
                i += 1;
            }
        }

        // Post-process step 1: remove transcript models with fewer segments than the max
        if !mapping_record.is_empty() {
            let max_n_segments = mapping_record.values().map(|v| v.len()).max().unwrap_or(0);
            mapping_record.retain(|_, v| v.len() >= max_n_segments);
        }

        // Post-process step 2: remove TMs where any SegmentMatch skip doesn't make sense
        if !mapping_record.is_empty() {
            mapping_record
                .retain(|_, segmatch_list| segmatch_list.iter().all(|sm| sm.skip_makes_sense()));
        }

        mapping_record
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::Feature;
    use crate::read::Read;
    use crate::transcript_model::TranscriptModel;

    fn make_read(pos: i64, segments: Vec<(i64, i64)>, spliced: bool) -> Read {
        Read::new(
            "bc1".to_string(),
            "umi1".to_string(),
            String::new(),
            "chr1".to_string(),
            '+',
            pos,
            segments,
            None,
            None,
            spliced,
        )
    }

    fn make_tm(start: i64, end: i64) -> TranscriptModel {
        let mut tm = TranscriptModel::new(
            "TR1".to_string(),
            "Gene1".to_string(),
            "G1".to_string(),
            "Gene1".to_string(),
            "chr1+".to_string(),
        );
        // Add two exons separated by an intron
        let exon1 = Feature::new(start, start + 100, b'e', 1, Some(0));
        let exon2 = Feature::new(end - 100, end, b'e', 2, Some(0));
        tm.append_exon(exon1);
        tm.append_exon(exon2);
        tm
    }

    #[test]
    fn test_transcripts_index_basic() {
        let tm = make_tm(1000, 2000);
        let mut idx = TranscriptsIndex::new(vec![tm]);
        assert_eq!(idx.maxtidx, 0);
        assert!(!idx.scan_not_terminated()); // only 1 model, tidx==maxtidx

        // Read entirely upstream — no overlap
        let read = make_read(500, vec![(500, 600)], false);
        let result = idx.find_overlapping_transcript_models(&read);
        // The transcript model spans 1000-2000, read is 500-600, no overlap expected
        assert!(result.is_empty());
    }

    #[test]
    fn test_feature_index_has_ivls_enclosing() {
        // Two features needed: the inner loop runs while i < maxiidx (faithful Python translation),
        // so the first element is only processed when maxiidx >= 1.
        let f1 = Feature::new(100, 500, b'e', 1, Some(0));
        let f2 = Feature::new(600, 800, b'e', 2, Some(0));
        let mut fidx = FeatureIndex::new(vec![f1, f2]);

        // Segment fully inside f1 (200-300 within 100-500): should return true
        let read_inside = make_read(200, vec![(200, 300)], false);
        assert!(fidx.has_ivls_enclosing(&read_inside));

        fidx.reset();
        // Segment outside both features: should return false
        let read_outside = make_read(900, vec![(900, 1000)], false);
        assert!(!fidx.has_ivls_enclosing(&read_outside));
    }

    #[test]
    fn test_feature_index_find_overlapping_ivls_empty() {
        let mut fidx = FeatureIndex::new(vec![]);
        let read = make_read(100, vec![(100, 200)], false);
        let result = fidx.find_overlapping_ivls(&read);
        assert!(result.is_empty());
    }

    #[test]
    fn test_feature_index_scan_not_terminated() {
        let f1 = Feature::new(100, 200, b'e', 1, Some(0));
        let f2 = Feature::new(300, 400, b'e', 2, Some(0));
        let fidx = FeatureIndex::new(vec![f1, f2]);
        assert!(fidx.last_interval_not_reached()); // iidx=0, maxiidx=1
    }
}
