//! Translated from velocyto/molitem.py

use crate::segment_match::SegmentMatch;
use std::collections::HashMap;

/// Represents a molecule (UMI) in the counting pipeline.
/// Holds the set of transcript model mappings that support this molecule.
pub struct Molitem {
    pub mappings_record: Option<HashMap<usize, Vec<SegmentMatch>>>, // keyed by TranscriptModel index
}

impl Molitem {
    /// Creates a new empty Molitem.
    pub fn new() -> Self {
        Molitem {
            mappings_record: None,
        }
    }

    /// Merges or intersects the current mappings with a new set, using union for the inner sets and intersection for the outer keys.
    pub fn add_mappings_record(&mut self, mappings_record: HashMap<usize, Vec<SegmentMatch>>) {
        match self.mappings_record.take() {
            None => {
                self.mappings_record = Some(mappings_record);
            }
            Some(existing) => {
                self.mappings_record = Some(dictionary_intersect(existing, mappings_record));
            }
        }
    }
}

impl Default for Molitem {
    fn default() -> Self {
        Self::new()
    }
}

/// Set union (|) on HashMap<usize, Vec<SegmentMatch>>:
/// keys = union of both key sets; values = concatenated vecs.
pub fn dictionary_union(
    mut d1: HashMap<usize, Vec<SegmentMatch>>,
    d2: HashMap<usize, Vec<SegmentMatch>>,
) -> HashMap<usize, Vec<SegmentMatch>> {
    for (k, v) in d2 {
        d1.entry(k).or_default().extend(v);
    }
    d1
}

/// Set intersection (&) on HashMap<usize, Vec<SegmentMatch>>:
/// keys = intersection of both key sets; values = concatenated vecs.
pub fn dictionary_intersect(
    mut d1: HashMap<usize, Vec<SegmentMatch>>,
    mut d2: HashMap<usize, Vec<SegmentMatch>>,
) -> HashMap<usize, Vec<SegmentMatch>> {
    let mut result = HashMap::new();
    // Iterate over keys present in both
    let keys: Vec<usize> = d1.keys().filter(|k| d2.contains_key(k)).cloned().collect();
    for k in keys {
        let v1 = d1.remove(&k).unwrap();
        let v2 = d2.remove(&k).unwrap();
        let mut combined = v1;
        combined.extend(v2);
        result.insert(k, combined);
    }
    result
}
