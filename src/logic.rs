//! Translated from velocyto/logic.py
//! Python ABC → Rust trait. Each Logic subclass becomes a zero-size struct implementing Logic.

use crate::constants::MIN_FLANK;
use crate::molitem::Molitem;
use crate::transcript_model::TranscriptModel;
use ndarray::Array2;
use std::collections::HashMap;

/// Base class from which all the logic variants inherit.
pub trait Logic: Send + Sync {
    /// Returns the name of this logic variant.
    fn name(&self) -> &str;
    /// Returns the names of the counting layers used by this logic.
    fn layers(&self) -> &[&str];
    /// Whether this logic requires strand information.
    fn stranded(&self) -> bool {
        true
    }
    /// Whether intron validation markup should be run before counting.
    fn perform_validation_markup(&self) -> bool {
        true
    }
    /// Whether discordant reads (mapping to both strands) should be accepted.
    fn accept_discordant(&self) -> bool {
        false
    }
    /// Attributes a molecule to one of the counting categories (spliced/unspliced/ambiguous).
    /// Returns a status code for Permissive10X, None for all others.
    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32>;
}

// ---------------------------------------------------------------------------
// Permissive10X
// ---------------------------------------------------------------------------
/// Permissive logic for 10X. All intron-only reads (singleton or not, validated or not) are counted as unspliced.
pub struct Permissive10X;
impl Logic for Permissive10X {
    fn name(&self) -> &str {
        "Permissive10X"
    }
    fn layers(&self) -> &[&str] {
        &["spliced", "unspliced", "ambiguous"]
    }

    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32> {
        let mappings_len = molitem
            .mappings_record
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        if mappings_len == 0 {
            return Some(2);
        }

        // Check single gene
        let n_genes = {
            let m = molitem.mappings_record.as_ref().unwrap();
            m.keys()
                .map(|&idx| tms[idx].geneid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        if n_genes != 1 {
            return Some(3);
        }

        // Inline flag computation (from Python Permissive10X.count loop)
        let mappings = molitem.mappings_record.as_ref().unwrap();
        let mut gene_check: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_onlyintron_model = false;
        let mut has_only_span_exin_model = true; // starts 1 in Python
        let mut has_onlyintron_and_valid_model = false;
        let mut has_valid_mixed_model = false;
        let mut has_invalid_mixed_model = false;
        let mut has_onlyexo_model = false;
        let mut has_mixed_model = false;
        let mut multi_gene = false;
        let mut last_geneid = String::new();

        for (&tm_idx, segments_list) in mappings {
            let tm = &tms[tm_idx];
            gene_check.insert(&tm.geneid);
            if gene_check.len() > 1 {
                multi_gene = true;
            }
            last_geneid = tm.geneid.clone();

            let mut has_introns = false;
            let mut has_exons = false;
            let mut has_validated_intron = false;
            let mut has_exin_intron_span = false;

            for sm in segments_list {
                if sm.maps_to_intron() {
                    has_introns = true;
                    let feat = &sm.feature;
                    if feat.is_validated {
                        has_validated_intron = true;
                        if feat.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let ds_idx = feat.get_downstream_exon_idx(tm);
                            if ds_idx < tm.list_features.len() {
                                let downstream_exon = &tm.list_features[ds_idx];
                                if downstream_exon.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                        if feat.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let us_idx = feat.get_upstream_exon_idx(tm);
                            if us_idx < tm.list_features.len() {
                                let upstream_exon = &tm.list_features[us_idx];
                                if upstream_exon.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                    }
                } else if sm.maps_to_exon() {
                    has_exons = true;
                }
            }

            if has_validated_intron && !has_exons {
                has_onlyintron_and_valid_model = true;
            }
            if has_introns && !has_exons {
                has_onlyintron_model = true;
            }
            if has_exons && !has_introns {
                has_onlyexo_model = true;
            }
            if has_exons && has_introns && !has_validated_intron && !has_exin_intron_span {
                has_invalid_mixed_model = true;
                has_mixed_model = true;
            }
            if has_exons && has_introns && has_validated_intron && !has_exin_intron_span {
                has_valid_mixed_model = true;
                has_mixed_model = true;
            }
            if !has_exin_intron_span {
                has_only_span_exin_model = false;
            }
        }

        if multi_gene {
            return Some(1);
        }

        if mappings_len == 0 {
            return Some(2);
        }

        let gene_ix = match geneid2ix.get(&last_geneid) {
            Some(&ix) => ix,
            None => return Some(4),
        };

        if has_onlyexo_model && !has_onlyintron_model && !has_mixed_model {
            dict_layers_columns.get_mut("spliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_only_span_exin_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_onlyintron_and_valid_model && !has_mixed_model && !has_onlyexo_model {
            // singleton or non-singleton — both count unspliced in Permissive
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_onlyintron_model
            && !has_onlyintron_and_valid_model
            && !has_mixed_model
            && !has_onlyexo_model
        {
            // singleton or non-singleton in non-validated — count unspliced
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_invalid_mixed_model
            && !has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_onlyintron_model && has_onlyexo_model && !has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_onlyintron_model && !has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if !has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        if has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return Some(0);
        }
        Some(4)
    }
}

// ---------------------------------------------------------------------------
// Intermediate10X
// ---------------------------------------------------------------------------
/// Singletons in non-validated introns are discarded; non-singletons in non-validated introns are counted as unspliced.
pub struct Intermediate10X;
impl Logic for Intermediate10X {
    fn name(&self) -> &str {
        "Intermediate10X"
    }
    fn layers(&self) -> &[&str] {
        &["spliced", "unspliced", "ambiguous"]
    }

    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32> {
        let mappings_len = molitem
            .mappings_record
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        if mappings_len == 0 {
            return None;
        }

        let n_genes = {
            let m = molitem.mappings_record.as_ref().unwrap();
            m.keys()
                .map(|&idx| tms[idx].geneid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        if n_genes != 1 {
            return None;
        }

        // Inline flag computation (from Python Intermediate10X.count loop)
        let mappings = molitem.mappings_record.as_ref().unwrap();
        let mut gene_check: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_onlyintron_model = false;
        let mut has_only_span_exin_model = true;
        let mut has_onlyintron_and_valid_model = false;
        let mut has_valid_mixed_model = false;
        let mut has_invalid_mixed_model = false;
        let mut has_onlyexo_model = false;
        let mut has_mixed_model = false;
        let mut multi_gene = false;
        let mut last_geneid = String::new();
        let mut last_segments_len = 0usize;

        for (&tm_idx, segments_list) in mappings {
            let tm = &tms[tm_idx];
            gene_check.insert(&tm.geneid);
            if gene_check.len() > 1 {
                multi_gene = true;
            }
            last_geneid = tm.geneid.clone();
            last_segments_len = segments_list.len();

            let mut has_introns = false;
            let mut has_exons = false;
            let mut has_validated_intron = false;
            let mut has_exin_intron_span = false;

            for sm in segments_list {
                if sm.maps_to_intron() {
                    has_introns = true;
                    let feat = &sm.feature;
                    if feat.is_validated {
                        has_validated_intron = true;
                        if feat.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let ds_idx = feat.get_downstream_exon_idx(tm);
                            if ds_idx < tm.list_features.len() {
                                let downstream_exon = &tm.list_features[ds_idx];
                                if downstream_exon.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                        if feat.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let us_idx = feat.get_upstream_exon_idx(tm);
                            if us_idx < tm.list_features.len() {
                                let upstream_exon = &tm.list_features[us_idx];
                                if upstream_exon.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                    }
                } else if sm.maps_to_exon() {
                    has_exons = true;
                }
            }

            if has_validated_intron && !has_exons {
                has_onlyintron_and_valid_model = true;
            }
            if has_introns && !has_exons {
                has_onlyintron_model = true;
            }
            if has_exons && !has_introns {
                has_onlyexo_model = true;
            }
            if has_exons && has_introns && !has_validated_intron && !has_exin_intron_span {
                has_invalid_mixed_model = true;
                has_mixed_model = true;
            }
            if has_exons && has_introns && has_validated_intron && !has_exin_intron_span {
                has_valid_mixed_model = true;
                has_mixed_model = true;
            }
            if !has_exin_intron_span {
                has_only_span_exin_model = false;
            }
        }

        if multi_gene {
            return None;
        }
        if mappings_len == 0 {
            return None;
        }

        let gene_ix = match geneid2ix.get(&last_geneid) {
            Some(&ix) => ix,
            None => return None,
        };

        if has_onlyexo_model && !has_onlyintron_model && !has_mixed_model {
            dict_layers_columns.get_mut("spliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_only_span_exin_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_and_valid_model && !has_mixed_model && !has_onlyexo_model {
            // singleton or non-singleton in validated — count unspliced
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model
            && !has_onlyintron_and_valid_model
            && !has_mixed_model
            && !has_onlyexo_model
        {
            if last_segments_len == 1 {
                // singleton in non-validated — discard
                return None;
            } else {
                // non-singleton in non-validated — count unspliced
                dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
                return None;
            }
        }
        if has_invalid_mixed_model
            && !has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            return None;
        }
        if has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && !has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && !has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if !has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// ValidatedIntrons10X
// ---------------------------------------------------------------------------
/// Singletons and non-singletons in non-validated introns are discarded; only validated-intron reads are counted as unspliced.
pub struct ValidatedIntrons10X;
impl Logic for ValidatedIntrons10X {
    fn name(&self) -> &str {
        "ValidatedIntrons10X"
    }
    fn layers(&self) -> &[&str] {
        &["spliced", "unspliced", "ambiguous"]
    }

    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32> {
        let mappings_len = molitem
            .mappings_record
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        if mappings_len == 0 {
            return None;
        }

        let n_genes = {
            let m = molitem.mappings_record.as_ref().unwrap();
            m.keys()
                .map(|&idx| tms[idx].geneid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        if n_genes != 1 {
            return None;
        }

        // Inline flag computation (from Python ValidatedIntrons10X.count loop)
        let mappings = molitem.mappings_record.as_ref().unwrap();
        let mut gene_check: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_onlyintron_model = false;
        let mut has_only_span_exin_model = true;
        let mut has_onlyintron_and_valid_model = false;
        let mut has_valid_mixed_model = false;
        let mut has_invalid_mixed_model = false;
        let mut has_onlyexo_model = false;
        let mut has_mixed_model = false;
        let mut multi_gene = false;
        let mut last_geneid = String::new();

        for (&tm_idx, segments_list) in mappings {
            let tm = &tms[tm_idx];
            gene_check.insert(&tm.geneid);
            if gene_check.len() > 1 {
                multi_gene = true;
            }
            last_geneid = tm.geneid.clone();

            let mut has_introns = false;
            let mut has_exons = false;
            let mut has_validated_intron = false;
            let mut has_exin_intron_span = false;

            for sm in segments_list {
                if sm.maps_to_intron() {
                    has_introns = true;
                    let feat = &sm.feature;
                    if feat.is_validated {
                        has_validated_intron = true;
                        if feat.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let ds_idx = feat.get_downstream_exon_idx(tm);
                            if ds_idx < tm.list_features.len() {
                                let downstream_exon = &tm.list_features[ds_idx];
                                if downstream_exon.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                        if feat.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let us_idx = feat.get_upstream_exon_idx(tm);
                            if us_idx < tm.list_features.len() {
                                let upstream_exon = &tm.list_features[us_idx];
                                if upstream_exon.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                    }
                } else if sm.maps_to_exon() {
                    has_exons = true;
                }
            }

            if has_validated_intron && !has_exons {
                has_onlyintron_and_valid_model = true;
            }
            if has_introns && !has_exons {
                has_onlyintron_model = true;
            }
            if has_exons && !has_introns {
                has_onlyexo_model = true;
            }
            if has_exons && has_introns && !has_validated_intron && !has_exin_intron_span {
                has_invalid_mixed_model = true;
                has_mixed_model = true;
            }
            if has_exons && has_introns && has_validated_intron && !has_exin_intron_span {
                has_valid_mixed_model = true;
                has_mixed_model = true;
            }
            if !has_exin_intron_span {
                has_only_span_exin_model = false;
            }
        }

        if multi_gene {
            return None;
        }
        if mappings_len == 0 {
            return None;
        }

        let gene_ix = match geneid2ix.get(&last_geneid) {
            Some(&ix) => ix,
            None => return None,
        };

        if has_onlyexo_model && !has_onlyintron_model && !has_mixed_model {
            dict_layers_columns.get_mut("spliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_only_span_exin_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_and_valid_model && !has_mixed_model && !has_onlyexo_model {
            // singleton or non-singleton in validated — count unspliced
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model
            && !has_onlyintron_and_valid_model
            && !has_mixed_model
            && !has_onlyexo_model
        {
            // singleton or non-singleton in non-validated — discard
            return None;
        }
        if has_invalid_mixed_model
            && !has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            return None;
        }
        if has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && !has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && !has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if !has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Stricter10X
// ---------------------------------------------------------------------------
/// Singletons in validated introns are discarded; only non-singleton validated-intron reads are counted as unspliced.
pub struct Stricter10X;
impl Logic for Stricter10X {
    fn name(&self) -> &str {
        "Stricter10X"
    }
    fn layers(&self) -> &[&str] {
        &["spliced", "unspliced", "ambiguous"]
    }

    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32> {
        let mappings_len = molitem
            .mappings_record
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        if mappings_len == 0 {
            return None;
        }

        let n_genes = {
            let m = molitem.mappings_record.as_ref().unwrap();
            m.keys()
                .map(|&idx| tms[idx].geneid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        if n_genes != 1 {
            return None;
        }

        // Inline flag computation (from Python Stricter10X.count loop)
        let mappings = molitem.mappings_record.as_ref().unwrap();
        let mut gene_check: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_onlyintron_model = false;
        let mut has_only_span_exin_model = true;
        let mut has_onlyintron_and_valid_model = false;
        let mut has_valid_mixed_model = false;
        let mut has_invalid_mixed_model = false;
        let mut has_onlyexo_model = false;
        let mut has_mixed_model = false;
        let mut multi_gene = false;
        let mut last_geneid = String::new();
        let mut last_segments_len = 0usize;

        for (&tm_idx, segments_list) in mappings {
            let tm = &tms[tm_idx];
            gene_check.insert(&tm.geneid);
            if gene_check.len() > 1 {
                multi_gene = true;
            }
            last_geneid = tm.geneid.clone();
            last_segments_len = segments_list.len();

            let mut has_introns = false;
            let mut has_exons = false;
            let mut has_validated_intron = false;
            let mut has_exin_intron_span = false;

            for sm in segments_list {
                if sm.maps_to_intron() {
                    has_introns = true;
                    let feat = &sm.feature;
                    if feat.is_validated {
                        has_validated_intron = true;
                        if feat.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let ds_idx = feat.get_downstream_exon_idx(tm);
                            if ds_idx < tm.list_features.len() {
                                let downstream_exon = &tm.list_features[ds_idx];
                                if downstream_exon.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                        if feat.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let us_idx = feat.get_upstream_exon_idx(tm);
                            if us_idx < tm.list_features.len() {
                                let upstream_exon = &tm.list_features[us_idx];
                                if upstream_exon.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                    }
                } else if sm.maps_to_exon() {
                    has_exons = true;
                }
            }

            if has_validated_intron && !has_exons {
                has_onlyintron_and_valid_model = true;
            }
            if has_introns && !has_exons {
                has_onlyintron_model = true;
            }
            if has_exons && !has_introns {
                has_onlyexo_model = true;
            }
            if has_exons && has_introns && !has_validated_intron && !has_exin_intron_span {
                has_invalid_mixed_model = true;
                has_mixed_model = true;
            }
            if has_exons && has_introns && has_validated_intron && !has_exin_intron_span {
                has_valid_mixed_model = true;
                has_mixed_model = true;
            }
            if !has_exin_intron_span {
                has_only_span_exin_model = false;
            }
        }

        if multi_gene {
            return None;
        }
        if mappings_len == 0 {
            return None;
        }

        let gene_ix = match geneid2ix.get(&last_geneid) {
            Some(&ix) => ix,
            None => return None,
        };

        if has_onlyexo_model && !has_onlyintron_model && !has_mixed_model {
            dict_layers_columns.get_mut("spliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_only_span_exin_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_and_valid_model && !has_mixed_model && !has_onlyexo_model {
            if last_segments_len == 1 {
                // singleton in validated — discard
                return None;
            } else {
                // non-singleton in validated — count unspliced
                dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
                return None;
            }
        }
        if has_onlyintron_model
            && !has_onlyintron_and_valid_model
            && !has_mixed_model
            && !has_onlyexo_model
        {
            // singleton or non-singleton in non-validated — discard
            return None;
        }
        if has_invalid_mixed_model
            && !has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            return None;
        }
        if has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && !has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && !has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if !has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// ObservedSpanning10X
// ---------------------------------------------------------------------------
/// Only observed intron-spanning reads are counted as unspliced; all intron-only reads (singleton/non-singleton, validated/non-validated) are discarded.
pub struct ObservedSpanning10X;
impl Logic for ObservedSpanning10X {
    fn name(&self) -> &str {
        "ObservedSpanning10X"
    }
    fn layers(&self) -> &[&str] {
        &["spliced", "unspliced", "ambiguous"]
    }

    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32> {
        let mappings_len = molitem
            .mappings_record
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        if mappings_len == 0 {
            return None;
        }

        let n_genes = {
            let m = molitem.mappings_record.as_ref().unwrap();
            m.keys()
                .map(|&idx| tms[idx].geneid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        if n_genes != 1 {
            return None;
        }

        // Inline flag computation (from Python ObservedSpanning10X.count loop)
        let mappings = molitem.mappings_record.as_ref().unwrap();
        let mut gene_check: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_onlyintron_model = false;
        let mut has_only_span_exin_model = true;
        let mut has_onlyintron_and_valid_model = false;
        let mut has_valid_mixed_model = false;
        let mut has_invalid_mixed_model = false;
        let mut has_onlyexo_model = false;
        let mut has_mixed_model = false;
        let mut multi_gene = false;
        let mut last_geneid = String::new();

        for (&tm_idx, segments_list) in mappings {
            let tm = &tms[tm_idx];
            gene_check.insert(&tm.geneid);
            if gene_check.len() > 1 {
                multi_gene = true;
            }
            last_geneid = tm.geneid.clone();

            let mut has_introns = false;
            let mut has_exons = false;
            let mut has_validated_intron = false;
            let mut has_exin_intron_span = false;

            for sm in segments_list {
                if sm.maps_to_intron() {
                    has_introns = true;
                    let feat = &sm.feature;
                    if feat.is_validated {
                        has_validated_intron = true;
                        if feat.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let ds_idx = feat.get_downstream_exon_idx(tm);
                            if ds_idx < tm.list_features.len() {
                                let downstream_exon = &tm.list_features[ds_idx];
                                if downstream_exon.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                        if feat.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let us_idx = feat.get_upstream_exon_idx(tm);
                            if us_idx < tm.list_features.len() {
                                let upstream_exon = &tm.list_features[us_idx];
                                if upstream_exon.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                    }
                } else if sm.maps_to_exon() {
                    has_exons = true;
                }
            }

            if has_validated_intron && !has_exons {
                has_onlyintron_and_valid_model = true;
            }
            if has_introns && !has_exons {
                has_onlyintron_model = true;
            }
            if has_exons && !has_introns {
                has_onlyexo_model = true;
            }
            if has_exons && has_introns && !has_validated_intron && !has_exin_intron_span {
                has_invalid_mixed_model = true;
                has_mixed_model = true;
            }
            if has_exons && has_introns && has_validated_intron && !has_exin_intron_span {
                has_valid_mixed_model = true;
                has_mixed_model = true;
            }
            if !has_exin_intron_span {
                has_only_span_exin_model = false;
            }
        }

        if multi_gene {
            return None;
        }
        if mappings_len == 0 {
            return None;
        }

        let gene_ix = match geneid2ix.get(&last_geneid) {
            Some(&ix) => ix,
            None => return None,
        };

        if has_onlyexo_model && !has_onlyintron_model && !has_mixed_model {
            dict_layers_columns.get_mut("spliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_only_span_exin_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_and_valid_model && !has_mixed_model && !has_onlyexo_model {
            // singleton or non-singleton in validated — discard
            return None;
        }
        if has_onlyintron_model
            && !has_onlyintron_and_valid_model
            && !has_mixed_model
            && !has_onlyexo_model
        {
            // non-validated — discard
            return None;
        }
        if has_invalid_mixed_model
            && !has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            return None;
        }
        if has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && !has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && !has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if !has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Discordant10X
// ---------------------------------------------------------------------------
/// Same as Permissive10X but also accepts discordant reads.
pub struct Discordant10X;
impl Logic for Discordant10X {
    fn name(&self) -> &str {
        "Discordant10X"
    }
    fn layers(&self) -> &[&str] {
        &["spliced", "unspliced", "ambiguous"]
    }
    fn accept_discordant(&self) -> bool {
        true
    }

    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32> {
        let mappings_len = molitem
            .mappings_record
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        if mappings_len == 0 {
            return None;
        }

        let n_genes = {
            let m = molitem.mappings_record.as_ref().unwrap();
            m.keys()
                .map(|&idx| tms[idx].geneid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        if n_genes != 1 {
            return None;
        }

        // Inline flag computation (from Python Discordant10X.count loop)
        let mappings = molitem.mappings_record.as_ref().unwrap();
        let mut gene_check: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_onlyintron_model = false;
        let mut has_only_span_exin_model = true;
        let mut has_onlyintron_and_valid_model = false;
        let mut has_valid_mixed_model = false;
        let mut has_invalid_mixed_model = false;
        let mut has_onlyexo_model = false;
        let mut has_mixed_model = false;
        let mut multi_gene = false;
        let mut last_geneid = String::new();

        for (&tm_idx, segments_list) in mappings {
            let tm = &tms[tm_idx];
            gene_check.insert(&tm.geneid);
            if gene_check.len() > 1 {
                multi_gene = true;
            }
            last_geneid = tm.geneid.clone();

            let mut has_introns = false;
            let mut has_exons = false;
            let mut has_validated_intron = false;
            let mut has_exin_intron_span = false;

            for sm in segments_list {
                if sm.maps_to_intron() {
                    has_introns = true;
                    let feat = &sm.feature;
                    if feat.is_validated {
                        has_validated_intron = true;
                        if feat.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let ds_idx = feat.get_downstream_exon_idx(tm);
                            if ds_idx < tm.list_features.len() {
                                let downstream_exon = &tm.list_features[ds_idx];
                                if downstream_exon.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                        if feat.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                            let us_idx = feat.get_upstream_exon_idx(tm);
                            if us_idx < tm.list_features.len() {
                                let upstream_exon = &tm.list_features[us_idx];
                                if upstream_exon.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                    has_exin_intron_span = true;
                                }
                            }
                        }
                    }
                } else if sm.maps_to_exon() {
                    has_exons = true;
                }
            }

            if has_validated_intron && !has_exons {
                has_onlyintron_and_valid_model = true;
            }
            if has_introns && !has_exons {
                has_onlyintron_model = true;
            }
            if has_exons && !has_introns {
                has_onlyexo_model = true;
            }
            if has_exons && has_introns && !has_validated_intron && !has_exin_intron_span {
                has_invalid_mixed_model = true;
                has_mixed_model = true;
            }
            if has_exons && has_introns && has_validated_intron && !has_exin_intron_span {
                has_valid_mixed_model = true;
                has_mixed_model = true;
            }
            if !has_exin_intron_span {
                has_only_span_exin_model = false;
            }
        }

        if multi_gene {
            return None;
        }
        if mappings_len == 0 {
            return None;
        }

        let gene_ix = match geneid2ix.get(&last_geneid) {
            Some(&ix) => ix,
            None => return None,
        };

        if has_onlyexo_model && !has_onlyintron_model && !has_mixed_model {
            dict_layers_columns.get_mut("spliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_only_span_exin_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_and_valid_model && !has_mixed_model && !has_onlyexo_model {
            // singleton or non-singleton in validated — count unspliced
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model
            && !has_onlyintron_and_valid_model
            && !has_mixed_model
            && !has_onlyexo_model
        {
            // singleton or non-singleton in non-validated — count unspliced
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_invalid_mixed_model
            && !has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_valid_mixed_model
            && !has_onlyintron_model
            && !has_onlyexo_model
            && !has_only_span_exin_model
        {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && !has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && !has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if !has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// SmartSeq2
// ---------------------------------------------------------------------------
/// SmartSeq2 logic: unstranded, no validation markup; spanning reads counted in a separate layer.
pub struct SmartSeq2;
impl Logic for SmartSeq2 {
    fn name(&self) -> &str {
        "SmartSeq2"
    }
    fn layers(&self) -> &[&str] {
        &["spliced", "unspliced", "ambiguous", "spanning"]
    }
    fn stranded(&self) -> bool {
        false
    }
    fn perform_validation_markup(&self) -> bool {
        false
    }

    fn count(
        &self,
        molitem: &Molitem,
        cell_bcidx: usize,
        dict_layers_columns: &mut HashMap<String, Array2<u16>>,
        geneid2ix: &HashMap<String, usize>,
        tms: &[TranscriptModel],
    ) -> Option<i32> {
        let mappings_len = molitem
            .mappings_record
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        if mappings_len == 0 {
            return None;
        }

        let n_genes = {
            let m = molitem.mappings_record.as_ref().unwrap();
            m.keys()
                .map(|&idx| tms[idx].geneid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        if n_genes != 1 {
            return None;
        }

        // Inline flag computation (from Python SmartSeq2.count loop)
        // SmartSeq2 differs: no has_validated_intron, no has_non3prime,
        // intron span checking does not require is_validated.
        let mappings = molitem.mappings_record.as_ref().unwrap();
        let mut gene_check: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_onlyintron_model = false;
        let mut has_only_span_exin_model = true;
        let mut has_onlyexo_model = false;
        let mut has_mixed_model = false;
        let mut multi_gene = false;
        let mut last_geneid = String::new();

        for (&tm_idx, segments_list) in mappings {
            let tm = &tms[tm_idx];
            gene_check.insert(&tm.geneid);
            if gene_check.len() > 1 {
                multi_gene = true;
            }
            last_geneid = tm.geneid.clone();

            let mut has_introns = false;
            let mut has_exons = false;
            let mut has_exin_intron_span = false;

            for sm in segments_list {
                if sm.maps_to_intron() {
                    has_introns = true;
                    let feat = &sm.feature;
                    // SmartSeq2 checks span without requiring is_validated
                    if feat.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                        let ds_idx = feat.get_downstream_exon_idx(tm);
                        if ds_idx < tm.list_features.len() {
                            let downstream_exon = &tm.list_features[ds_idx];
                            if downstream_exon.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                has_exin_intron_span = true;
                            }
                        }
                    }
                    if feat.start_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                        let us_idx = feat.get_upstream_exon_idx(tm);
                        if us_idx < tm.list_features.len() {
                            let upstream_exon = &tm.list_features[us_idx];
                            if upstream_exon.end_overlaps_with_part_of(sm.segment, MIN_FLANK) {
                                has_exin_intron_span = true;
                            }
                        }
                    }
                } else if sm.maps_to_exon() {
                    has_exons = true;
                }
            }

            if has_introns && !has_exons {
                has_onlyintron_model = true;
            }
            if has_exons && !has_introns {
                has_onlyexo_model = true;
            }
            if has_exons && has_introns && !has_exin_intron_span {
                has_mixed_model = true;
            }
            if !has_exin_intron_span {
                has_only_span_exin_model = false;
            }
        }

        if multi_gene {
            return None;
        }
        if mappings_len == 0 {
            return None;
        }

        let gene_ix = match geneid2ix.get(&last_geneid) {
            Some(&ix) => ix,
            None => return None,
        };

        if has_onlyexo_model && !has_onlyintron_model && !has_mixed_model {
            dict_layers_columns.get_mut("spliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_only_span_exin_model {
            // count as spanning (not unspliced like 10X)
            dict_layers_columns.get_mut("spanning").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && !has_mixed_model && !has_onlyexo_model {
            dict_layers_columns.get_mut("unspliced").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if has_onlyintron_model && has_onlyexo_model && !has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        if !has_onlyintron_model && has_onlyexo_model && has_mixed_model {
            dict_layers_columns.get_mut("ambiguous").unwrap()[[gene_ix, cell_bcidx]] += 1;
            return None;
        }
        None
    }
}

/// Constructs a boxed Logic implementation by name. 'Default' is an alias for 'Permissive10X'. Panics on unknown names.
pub fn logic_from_name(name: &str) -> Box<dyn Logic> {
    match name {
        "Default" | "Permissive10X" => Box::new(Permissive10X),
        "Intermediate10X" => Box::new(Intermediate10X),
        "ValidatedIntrons10X" => Box::new(ValidatedIntrons10X),
        "Stricter10X" => Box::new(Stricter10X),
        "ObservedSpanning10X" => Box::new(ObservedSpanning10X),
        "Discordant10X" => Box::new(Discordant10X),
        "SmartSeq2" => Box::new(SmartSeq2),
        _ => panic!("Unknown logic: {name}"),
    }
}
