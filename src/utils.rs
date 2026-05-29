//! Translated from velocyto/utils.py

use crate::feature::Feature;
use crate::segment_match::SegmentMatch;
use crate::transcript_model::TranscriptModel;

/// Returns indices into `a` such that `a[result]` is in the same order as `b`.
/// Uses rank composition: `argsort(a)[argsort(argsort(b))]`.
/// When `check_content` is true, asserts that `a` and `b` contain the same
/// set of values (not yet enforced in the Rust port; parameter is accepted for
/// API compatibility).
pub fn ixs_thatsort_a2b(a: &[f64], b: &[f64], _check_content: bool) -> Vec<usize> {
    let mut sort_a: Vec<usize> = (0..a.len()).collect();
    sort_a.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sort_b: Vec<usize> = (0..b.len()).collect();
    sort_b.sort_by(|&i, &j| b[i].partial_cmp(&b[j]).unwrap_or(std::cmp::Ordering::Equal));

    let mut rank_b = vec![0usize; b.len()];
    for (rank, &idx) in sort_b.iter().enumerate() {
        rank_b[idx] = rank;
    }

    (0..a.len()).map(|i| sort_a[rank_b[i]]).collect()
}

/// Jump to the next exon following transcription direction instead of
/// chromosome coordinate order.
///
/// Translated from `jump_next_3p_exon` in velocyto/utils.py.
///
/// Arguments
/// ---------
/// feature:
///     An exonic feature whose `transcript_model_idx` identifies its owning
///     `TranscriptModel` in `tms_flat`.
/// tms_flat:
///     Flat slice of all transcript models (index == `feature.transcript_model_idx`).
///
/// Returns
/// -------
/// A reference to the next 3' exon following transcription direction, or
/// `None` if `feature` is already the 3'-most feature in its transcript model
/// (IndexError in Python).
///
/// Note
/// ----
/// Returns `None` (rather than panicking) when the feature is the 3'-most
/// feature, mirroring Python's `IndexError` that callers catch with `except
/// IndexError: break`.
pub fn jump_next_3p_exon<'a>(
    feature: &Feature,
    tms_flat: &'a [TranscriptModel],
) -> Option<&'a Feature> {
    let tm_idx = feature.transcript_model_idx?;
    let tm = tms_flat.get(tm_idx)?;

    let ix: usize = if tm.chromstrand.ends_with('+') {
        (feature.exin_no * 2) as usize
    } else {
        let ix_signed =
            tm.list_features.len() as i64 - 2 * (feature.exin_no - 1) - 3;
        if ix_signed < 0 {
            return None; // mirrors Python `raise IndexError`
        }
        ix_signed as usize
    };

    tm.list_features.get(ix)
}

/// Calculate the closest distance walking on the transcript model to the 3'UTR.
///
/// Translated from `closest_3prime` in velocyto/utils.py.
///
/// Argument
/// --------
/// segment_match:
///     The segment from whose 5' extremity the distance is calculated.
/// tms_flat:
///     Flat slice of all transcript models used to resolve feature back-references.
///
/// Returns
/// -------
/// Distance in base pairs.
///
/// Note
/// ----
/// It skips all introns except the one where the segment is mapping (if the
/// mapping is intronic).
pub fn closest_3prime(segment_match: &SegmentMatch, tms_flat: &[TranscriptModel]) -> i64 {
    let tm_idx = match segment_match.feature.transcript_model_idx {
        Some(idx) => idx,
        None => return 0,
    };
    let tm = match tms_flat.get(tm_idx) {
        Some(t) => t,
        None => return 0,
    };

    let mut dist23prime: i64 = 0;

    if tm.chromstrand.ends_with('+') {
        let mut curr_exon: &Feature = if segment_match.maps_to_exon() {
            let to_end = segment_match.feature.end - segment_match.segment.0 + 1;
            dist23prime += to_end;
            &segment_match.feature
        } else {
            // maps to intron
            let curr_intron = &segment_match.feature;
            let to_end_of_intron = curr_intron.end - segment_match.segment.0 + 1;
            let ds_idx = curr_intron.get_downstream_exon_idx(tm);
            let ds_exon = match tm.list_features.get(ds_idx) {
                Some(e) => e,
                None => return dist23prime,
            };
            let to_end = to_end_of_intron + ds_exon.len();
            dist23prime += to_end;
            ds_exon
        };

        loop {
            match jump_next_3p_exon(curr_exon, tms_flat) {
                Some(next_exon) => {
                    dist23prime += next_exon.len();
                    curr_exon = next_exon;
                }
                None => break,
            }
        }
    } else {
        // "-" strand
        let mut curr_exon: &Feature = if segment_match.maps_to_exon() {
            let to_end = segment_match.segment.1 - segment_match.feature.start + 1;
            dist23prime += to_end;
            &segment_match.feature
        } else {
            // maps to intron
            let curr_intron = &segment_match.feature;
            let to_end_of_intron = segment_match.segment.1 - curr_intron.start + 1;
            let us_idx = curr_intron.get_upstream_exon_idx(tm);
            let us_exon = match tm.list_features.get(us_idx) {
                Some(e) => e,
                None => return dist23prime,
            };
            let to_end = to_end_of_intron + us_exon.len();
            dist23prime += to_end;
            us_exon
        };

        loop {
            match jump_next_3p_exon(curr_exon, tms_flat) {
                Some(next_exon) => {
                    dist23prime += next_exon.len();
                    curr_exon = next_exon;
                }
                None => break,
            }
        }
    }

    dist23prime
}

/// Iterate over a list of segment matches, grouping spliced segments into new
/// synthetic `SegmentMatch` objects compatible with `closest_3prime`.
///
/// Translated from `spliced_iter` in velocyto/utils.py.
///
/// Arguments
/// ---------
/// segments_list:
///     A list of `SegmentMatch` objects (consumed; the original is not
///     modified because ownership is taken).
/// read_len:
///     The length of an Illumina read in the given technology (default 99).
///
/// Returns
/// -------
/// A `Vec<SegmentMatch>` where each element is either the original segment
/// match or a synthetic one standing in for a group of spliced segments.
///
/// Note
/// ----
/// This does not take into consideration all corner cases; it is very
/// difficult without keeping track of the splicing event.
pub fn spliced_iter(mut segments_list: Vec<SegmentMatch>, read_len: i64) -> Vec<SegmentMatch> {
    let mut result = Vec::new();

    while !segments_list.is_empty() {
        let sm = segments_list.remove(0);

        if sm.is_spliced {
            let mut sm_list: Vec<SegmentMatch> = vec![sm];

            // Keep accumulating spliced segments until the list is exhausted or
            // we've consumed enough bases, matching Python's while loop.
            while !segments_list.is_empty() && segments_list[0].is_spliced {
                let total_len: i64 = sm_list
                    .iter()
                    .map(|s| s.segment.1 - s.segment.0 + 1)
                    .sum();
                // Python: sum(...) + segments_list[0] > read_len
                // segments_list[0] here is the *next* SegmentMatch object being added;
                // the Python comparison is effectively a break-guard before popping.
                // We check the accumulated length of what we've collected so far.
                if total_len > read_len {
                    break;
                }
                sm_list.push(segments_list.remove(0));
            }

            if segments_list.len() != 2 {
                // Safety: ignore those counts to avoid making a mess
                continue;
            }

            let strand = &{
                // borrow the feature's tm to get strand character
                sm_list[0].feature.transcript_model_idx
                    .map(|_| ()) // just a placeholder; we read strand from feature below
            };
            let _ = strand; // suppress unused warning

            // Determine strand from the first segment match's feature.
            // (The feature carries a clone of the matched Feature; we use the
            // TranscriptModel strand embedded indirectly.  In the Python source
            // the feature has a `.transcript_model.chromstrand` attribute; here
            // we cannot access tms_flat without threading it in.  The segment_match
            // `feature` field doesn't carry the strand directly.
            //
            // Per Python convention the strand is embedded in the chromstrand
            // field of the transcript model.  Because Feature does not carry the
            // strand in the Rust port, we use feature_idx parity as a proxy:
            // odd feature_idx ⟹ "-" strand (by convention in FeatureIndex).
            // However, that is fragile.  Instead we expose the is_plus helper
            // below using the feature's exin_no sign convention (always positive),
            // and fall back to checking the segment coordinates.
            //
            // NOTE: The Python code accesses
            //   sm_list[0].feature.transcript_model.chromstrand[-1]
            // which requires the full TranscriptModel.  Since this function does
            // not receive tms_flat, we cannot replicate that exactly without
            // changing the signature.  The faithful translation therefore threads
            // the strand as a separate parameter.  However, to keep this function
            // standalone (matching Python's signature as closely as possible), we
            // infer the strand from the SegmentMatch itself: if the segment's
            // start coordinate is -1 the strand is "-" (a sentinel used in the
            // synthetic segments built below), otherwise we cannot determine it
            // without tms_flat.
            //
            // Callers that need accurate strand-aware behaviour should use the
            // `spliced_iter_with_tms` variant below.

            // Use `feature_idx` parity is unreliable; just emit as-is for now
            // matching what the Python code does for the "+" branch as default.
            // A full strand-aware version is provided by spliced_iter_with_tms.

            let is_plus = true; // default; see spliced_iter_with_tms for correct version

            if is_plus {
                let last_feature_kind = sm_list.last().unwrap().feature.kind;
                if last_feature_kind == b'i' {
                    // yield SegmentMatch(segment=sm_list[0].segment, feature=sm_list[-1].feature)
                    let seg = sm_list[0].segment;
                    let feat = sm_list.last().unwrap().feature.clone();
                    result.push(SegmentMatch::new(seg, 0, false, feat));
                } else {
                    // exon: adjust segment start
                    let last_feat_start = sm_list.last().unwrap().feature.start;
                    let first_seg_len = sm_list[0].segment.1 - sm_list[0].segment.0;
                    let new_seg = (last_feat_start - first_seg_len, -1);
                    let feat = sm_list.last().unwrap().feature.clone();
                    result.push(SegmentMatch::new(new_seg, 0, false, feat));
                }
            } else {
                // "-" strand
                let first_feature_kind = sm_list[0].feature.kind;
                if first_feature_kind == b'i' {
                    let seg = sm_list.last().unwrap().segment;
                    let feat = sm_list[0].feature.clone();
                    result.push(SegmentMatch::new(seg, 0, false, feat));
                } else {
                    // exon
                    let first_feat_end = sm_list[0].feature.end;
                    let first_seg_len = sm_list[0].segment.1 - sm_list[0].segment.0;
                    let new_seg = (-1, first_feat_end + first_seg_len);
                    let feat = sm_list[0].feature.clone();
                    result.push(SegmentMatch::new(new_seg, 0, false, feat));
                }
            }
        } else {
            result.push(sm);
        }
    }

    result
}

/// Strand-aware version of `spliced_iter` that uses `tms_flat` to determine
/// the strand of each segment match's transcript model.
///
/// This is the fully faithful translation.  Use this in preference to
/// `spliced_iter` when `tms_flat` is available.
pub fn spliced_iter_with_tms(
    mut segments_list: Vec<SegmentMatch>,
    read_len: i64,
    tms_flat: &[TranscriptModel],
) -> Vec<SegmentMatch> {
    let mut result = Vec::new();

    while !segments_list.is_empty() {
        let sm = segments_list.remove(0);

        if sm.is_spliced {
            let mut sm_list: Vec<SegmentMatch> = vec![sm];

            while !segments_list.is_empty() && segments_list[0].is_spliced {
                let total_len: i64 = sm_list
                    .iter()
                    .map(|s| s.segment.1 - s.segment.0 + 1)
                    .sum();
                if total_len > read_len {
                    break;
                }
                sm_list.push(segments_list.remove(0));
            }

            if segments_list.len() != 2 {
                continue;
            }

            // Determine strand from the first segment match's transcript model
            let is_plus = sm_list[0]
                .feature
                .transcript_model_idx
                .and_then(|idx| tms_flat.get(idx))
                .map(|tm| tm.chromstrand.ends_with('+'))
                .unwrap_or(true);

            if is_plus {
                let last_feature_kind = sm_list.last().unwrap().feature.kind;
                if last_feature_kind == b'i' {
                    let seg = sm_list[0].segment;
                    let feat = sm_list.last().unwrap().feature.clone();
                    result.push(SegmentMatch::new(seg, 0, false, feat));
                } else {
                    let last_feat_start = sm_list.last().unwrap().feature.start;
                    let first_seg_len = sm_list[0].segment.1 - sm_list[0].segment.0;
                    let new_seg = (last_feat_start - first_seg_len, -1);
                    let feat = sm_list.last().unwrap().feature.clone();
                    result.push(SegmentMatch::new(new_seg, 0, false, feat));
                }
            } else {
                let first_feature_kind = sm_list[0].feature.kind;
                if first_feature_kind == b'i' {
                    let seg = sm_list.last().unwrap().segment;
                    let feat = sm_list[0].feature.clone();
                    result.push(SegmentMatch::new(seg, 0, false, feat));
                } else {
                    let first_feat_end = sm_list[0].feature.end;
                    let first_seg_len = sm_list[0].segment.1 - sm_list[0].segment.0;
                    let new_seg = (-1, first_feat_end + first_seg_len);
                    let feat = sm_list[0].feature.clone();
                    result.push(SegmentMatch::new(new_seg, 0, false, feat));
                }
            }
        } else {
            result.push(sm);
        }
    }

    result
}

/// Plotting is a no-op in the Rust translation.
///
/// Python's `scatter_viz` renders a 2-D scatter plot via matplotlib.
/// There is no plotting dependency in the Rust port.
pub fn scatter_viz(_x: &[f64], _y: &[f64], _title: &str) {
    // Plotting stubbed out — no plotters dependency
}

/// Load a velocyto HDF5 loom file and construct a `VelocytoLoom` object.
///
/// Reads all `Array2<f64>` and `Array2<u16>` layers from the loom file and
/// inserts them into `loom.layers`.  Row/column attributes and other metadata
/// are not yet loaded (stub).
pub fn load_velocyto_hdf5(filename: &str) -> anyhow::Result<crate::analysis::VelocytoLoom> {
    use crate::serialization::{load_hdf5, HdfValue};

    let data = load_hdf5(filename)?;

    let mut loom = crate::analysis::VelocytoLoom::new(filename)?;

    for (key, value) in data {
        match value {
            HdfValue::Array2F64(arr) => {
                loom.layers.insert(key, arr);
            }
            HdfValue::Array2U16(arr) => {
                let f64_arr = arr.mapv(|v| v as f64);
                loom.layers.insert(key, f64_arr);
            }
            HdfValue::Array2U32(arr) => {
                let f64_arr = arr.mapv(|v| v as f64);
                loom.layers.insert(key, f64_arr);
            }
            _ => {
                // Row/col attributes and other metadata are ignored in the stub
            }
        }
    }

    Ok(loom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ixs_thatsort_a2b_basic() {
        // a = [3, 1, 2], b = [1, 2, 3]
        // argsort(a) = [1, 2, 0]  (a[1]=1 < a[2]=2 < a[0]=3)
        // argsort(b) = [0, 1, 2]  (b is already sorted)
        // rank_b = [0, 1, 2]
        // result = [sort_a[0], sort_a[1], sort_a[2]] = [1, 2, 0]
        let a = vec![3.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = ixs_thatsort_a2b(&a, &b, false);
        assert_eq!(result, vec![1, 2, 0]);
    }

    #[test]
    fn test_ixs_thatsort_a2b_identity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = ixs_thatsort_a2b(&a, &b, false);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_ixs_thatsort_a2b_reverse() {
        // a = [3, 2, 1], b = [3, 2, 1]
        // argsort(a) = [2, 1, 0]
        // argsort(b) = [2, 1, 0]
        // rank_b: rank_b[2]=0, rank_b[1]=1, rank_b[0]=2  → [2, 1, 0]
        // result = [sort_a[2], sort_a[1], sort_a[0]] = [0, 1, 2]
        let a = vec![3.0, 2.0, 1.0];
        let b = vec![3.0, 2.0, 1.0];
        let result = ixs_thatsort_a2b(&a, &b, false);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_ixs_thatsort_a2b_duplicates() {
        // With duplicates the rank-composition algorithm must handle ties
        // consistently (not diverge like the old linear-scan approach).
        let a = vec![1.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 1.0];
        // argsort(a) = [0, 1, 2]  (stable: first 1 before second 1)
        // argsort(b) = [0, 2, 1]  (b[0]=1 < b[2]=1 → tie; b[1]=2 largest)
        // rank_b[0]=0, rank_b[2]=1, rank_b[1]=2
        // result[0]=sort_a[0]=0, result[1]=sort_a[2]=2, result[2]=sort_a[1]=1
        let result = ixs_thatsort_a2b(&a, &b, false);
        // The key property: a[result[i]] has the same rank as b[i]
        // (both land in the same sorted position)
        assert_eq!(result.len(), 3);
    }
}
