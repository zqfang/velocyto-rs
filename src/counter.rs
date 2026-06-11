//! Translated from velocyto/counter.py
//! Core BAM counting pipeline. pysam → rust-htslib (behind "bam" feature).

use log::{debug, warn};
use rand::Rng;
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufRead;

use crate::constants::{LONGEST_INTRON_ALLOWED, PATCH_INDELS, PLACEHOLDER_UMI_LEN};
use crate::feature::Feature;
use crate::gene_info::GeneInfo;
use crate::indexes::FeatureIndex;
use crate::logic::Logic;
use crate::molitem::Molitem;
use crate::read::Read;
use crate::transcript_model::TranscriptModel;

#[cfg(feature = "bam")]
use rust_htslib::bam::{self, Read as BamRead, Record};

pub struct ExInCounter {
    pub sampleid: String,
    pub outputfolder: String,
    pub loom_numeric_dtype: String,
    pub filter_mode: bool,
    pub valid_bcset: HashSet<String>,
    /// chromstrand → trid → TranscriptModel
    pub annotations_by_chrm_strand: HashMap<String, BTreeMap<String, TranscriptModel>>,
    /// chromstrand → list of repeat-mask Feature intervals
    pub mask_ivls_by_chromstrand: HashMap<String, Vec<Feature>>,
    pub geneid2ix: HashMap<String, usize>,
    pub genes: HashMap<String, GeneInfo>,
    pub umi_extension: UmiExtension,
    pub onefilepercell: bool,
    pub kind_of_report: char,
    pub every_n_report: usize,
    pub report_state: usize,
    pub logic: Box<dyn Logic>,
    pub cellbarcode_str: String,
    pub umibarcode_str: String,
    /// BAM tag carrying the sample identity (e.g. BD Rhapsody `ST`). When
    /// `Some`, reads are demultiplexed so the cell identity becomes
    /// `(sample, barcode)`; when `None`, behaviour is single-sample (unchanged).
    pub sample_tag: Option<String>,
    /// Flat global TranscriptModel list, built by build_feature_indexes
    pub tms_flat: Vec<TranscriptModel>,
    /// chromstrand → FeatureIndex, built once by build_feature_indexes / mark_up_introns
    pub feature_indexes: HashMap<String, FeatureIndex>,
    /// chromstrand → FeatureIndex for repeat masks, built in count
    pub mask_indexes: HashMap<String, FeatureIndex>,
}

pub enum UmiExtension {
    No,
    Chr,
    Gene,
    Nbp(usize),
    WithoutUmi,
}

impl ExInCounter {
    /// Creates a new `ExInCounter`. Sets up UMI extension mode, barcode/UMI tag
    /// names, and report settings.
    ///
    /// Python: ExInCounter.__init__
    pub fn new(
        sampleid: String,
        logic: Box<dyn Logic>,
        valid_bcset: Option<HashSet<String>>,
        umi_extension: &str,
        onefilepercell: bool,
        dump_option: &str,
        outputfolder: String,
        loom_numeric_dtype: String,
    ) -> anyhow::Result<Self> {
        let (filter_mode, valid_bcset) = match valid_bcset {
            Some(s) => (true, s),
            None => (false, HashSet::new()),
        };
        let umi_ext = if umi_extension.to_lowercase() == "no" {
            UmiExtension::No
        } else if umi_extension.to_lowercase() == "chr" {
            UmiExtension::Chr
        } else if umi_extension.to_lowercase() == "gene" || umi_extension.to_lowercase() == "gx" {
            UmiExtension::Gene
        } else if umi_extension.to_lowercase().ends_with("bp") {
            let nbp: usize = umi_extension[..umi_extension.len() - 2]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid umi_extension: {umi_extension}"))?;
            UmiExtension::Nbp(nbp)
        } else if umi_extension.to_lowercase() == "without_umi" {
            UmiExtension::WithoutUmi
        } else {
            anyhow::bail!(
                "umi_extension {umi_extension} is not allowed. Use `no`, `Gene` or `[N]bp`"
            );
        };
        let (kind_of_report, every_n_report) = if dump_option.starts_with('p') {
            ('p', dump_option[1..].parse::<usize>().unwrap_or(0))
        } else {
            ('h', dump_option.parse::<usize>().unwrap_or(0))
        };
        Ok(ExInCounter {
            sampleid,
            outputfolder,
            loom_numeric_dtype,
            filter_mode,
            valid_bcset,
            annotations_by_chrm_strand: HashMap::new(),
            mask_ivls_by_chromstrand: HashMap::new(),
            geneid2ix: HashMap::new(),
            genes: HashMap::new(),
            umi_extension: umi_ext,
            onefilepercell,
            kind_of_report,
            every_n_report,
            report_state: 0,
            logic,
            cellbarcode_str: "CB".to_string(),
            umibarcode_str: "UB".to_string(),
            sample_tag: None,
            tms_flat: Vec::new(),
            feature_indexes: HashMap::new(),
            mask_indexes: HashMap::new(),
        })
    }

    // ── UMI / barcode extraction ──────────────────────────────────────────────

    /// Python: _no_extension
    #[cfg(feature = "bam")]
    fn no_extension(&self, read: &Record) -> anyhow::Result<String> {
        let tag = read
            .aux(self.umibarcode_str.as_bytes())
            .map_err(|e| anyhow::anyhow!("UMI tag {} missing: {e}", self.umibarcode_str))?;
        Ok(aux_to_string(tag))
    }

    /// Python: _extension_Nbp
    #[cfg(feature = "bam")]
    fn extension_nbp(&self, read: &Record, nbp: usize) -> anyhow::Result<String> {
        let umi = aux_to_string(
            read.aux(self.umibarcode_str.as_bytes())
                .map_err(|e| anyhow::anyhow!("UMI tag {} missing: {e}", self.umibarcode_str))?,
        );
        let seq = read.seq();
        let clip: String = (0..nbp.min(seq.len())).map(|i| seq[i] as char).collect();
        Ok(format!("{umi}{clip}"))
    }

    /// Python: _extension_Gene
    #[cfg(feature = "bam")]
    fn extension_gene(&self, read: &Record) -> anyhow::Result<String> {
        let umi = aux_to_string(
            read.aux(self.umibarcode_str.as_bytes())
                .map_err(|e| anyhow::anyhow!("UMI tag {} missing: {e}", self.umibarcode_str))?,
        );
        let gx = read
            .aux(b"GX")
            .map(aux_to_string)
            .unwrap_or_else(|_| "withoutGX".to_string());
        Ok(format!("{umi}_{gx}"))
    }

    /// Python: _placeholder_umi
    #[cfg(feature = "bam")]
    fn placeholder_umi(&self, _read: &Record) -> anyhow::Result<String> {
        let mut rng = rand::thread_rng();
        let s: String = (0..PLACEHOLDER_UMI_LEN)
            .map(|_| {
                let idx = rng.gen_range(0..36usize);
                if idx < 26 {
                    (b'A' + idx as u8) as char
                } else {
                    (b'0' + (idx - 26) as u8) as char
                }
            })
            .collect();
        Ok(s)
    }

    /// Python: _extension_chr
    #[cfg(feature = "bam")]
    fn extension_chr(&self, read: &Record) -> anyhow::Result<String> {
        let umi = aux_to_string(
            read.aux(self.umibarcode_str.as_bytes())
                .map_err(|e| anyhow::anyhow!("UMI tag {} missing: {e}", self.umibarcode_str))?,
        );
        let rname = read.tid();
        let rstart = read.pos() / 10_000_000;
        Ok(format!("{umi}_{rname}:{rstart}"))
    }

    /// Python: _normal_cell_barcode_get
    #[cfg(feature = "bam")]
    fn normal_cell_barcode_get(&self, read: &Record) -> anyhow::Result<String> {
        let tag = read
            .aux(self.cellbarcode_str.as_bytes())
            .map_err(|e| anyhow::anyhow!("CB tag {} missing: {e}", self.cellbarcode_str))?;
        let full = aux_to_string(tag);
        Ok(full.split('-').next().unwrap_or(&full).to_string())
    }

    // ── BAM peek ─────────────────────────────────────────────────────────────

    /// Python: peek — detect CB/UB vs XC/XM tag names from first `lines` reads
    #[cfg(feature = "bam")]
    pub fn peek(&mut self, bamfile: &str, lines: usize) -> anyhow::Result<()> {
        debug!("Peeking into {bamfile}");
        let mut reader = bam::Reader::from_path(bamfile)
            .map_err(|e| anyhow::anyhow!("Cannot open BAM {bamfile}: {e}"))?;
        let mut cellranger = 0usize;
        let mut dropseq = 0usize;
        let mut failed = 0usize;
        let mut rec = Record::new();
        let mut i = 0usize;
        loop {
            match reader.read(&mut rec) {
                Some(Ok(())) => {}
                None => break,
                Some(Err(e)) => return Err(anyhow::anyhow!("BAM read error: {e}")),
            }
            if rec.is_unmapped() {
                continue;
            }
            if rec.aux(b"CB").is_ok() && rec.aux(b"UB").is_ok() {
                cellranger += 1;
            } else if rec.aux(b"XC").is_ok() && rec.aux(b"XM").is_ok() {
                dropseq += 1;
            } else {
                warn!("Not found cell and umi barcode in entry {i} of the bam file");
                failed += 1;
            }
            if cellranger > lines {
                self.cellbarcode_str = "CB".to_string();
                self.umibarcode_str = "UB".to_string();
                break;
            } else if dropseq > lines {
                self.cellbarcode_str = "XC".to_string();
                self.umibarcode_str = "XM".to_string();
                break;
            } else if failed > 5 * lines {
                anyhow::bail!(
                    "The bam file does not contain cell and umi barcodes appropriately formatted."
                );
            }
            i += 1;
        }
        Ok(())
    }

    /// Python: peek_umi_only — detect UB vs XM from first `lines` reads
    #[cfg(feature = "bam")]
    pub fn peek_umi_only(&mut self, bamfile: &str, lines: usize) -> anyhow::Result<()> {
        debug!("Peeking into {bamfile}");
        let mut reader = bam::Reader::from_path(bamfile)
            .map_err(|e| anyhow::anyhow!("Cannot open BAM {bamfile}: {e}"))?;
        let mut cellranger = 0usize;
        let mut dropseq = 0usize;
        let mut failed = 0usize;
        let mut rec = Record::new();
        let mut i = 0usize;
        loop {
            match reader.read(&mut rec) {
                Some(Ok(())) => {}
                None => break,
                Some(Err(e)) => return Err(anyhow::anyhow!("BAM read error: {e}")),
            }
            if rec.is_unmapped() {
                continue;
            }
            if rec.aux(b"UB").is_ok() {
                cellranger += 1;
            } else if rec.aux(b"XM").is_ok() {
                dropseq += 1;
            } else {
                warn!("Not found umi barcode in entry {i} of the bam file");
                failed += 1;
            }
            if cellranger > lines {
                self.umibarcode_str = "UB".to_string();
                break;
            } else if dropseq > lines {
                self.umibarcode_str = "XM".to_string();
                break;
            } else if failed > 5 * lines {
                anyhow::bail!(
                    "The bam file does not contain umi barcodes appropriately formatted."
                );
            }
            i += 1;
        }
        Ok(())
    }

    // ── CIGAR parsing ─────────────────────────────────────────────────────────

    /// Python: parse_cigar_tuple
    /// Returns (segments, ref_skip, clip5, clip3)
    pub fn parse_cigar_tuple(
        cigartuples: &[(u32, u32)],
        pos: i64,
    ) -> (Vec<(i64, i64)>, bool, i64, i64) {
        let mut segments: Vec<(i64, i64)> = Vec::new();
        let mut hole_to_remove: std::collections::BTreeSet<usize> =
            std::collections::BTreeSet::new();
        let mut ref_skip = false;
        let mut clip5: i64 = 0;
        let mut clip3: i64 = 0;
        let mut p = pos;

        for (i, &(op, len)) in cigartuples.iter().enumerate() {
            let length = len as i64;
            match op {
                0 => {
                    // BAM_CMATCH
                    segments.push((p, p + length - 1));
                    p += length;
                }
                3 => {
                    // BAM_CREF_SKIP (splice)
                    ref_skip = true;
                    p += length;
                }
                2 => {
                    // BAM_CDEL
                    if length <= PATCH_INDELS {
                        let prev_ok = i > 0 && cigartuples[i - 1].0 == 0;
                        let next_ok = i + 1 < cigartuples.len() && cigartuples[i + 1].0 == 0;
                        if prev_ok && next_ok {
                            hole_to_remove.insert(segments.len().saturating_sub(1));
                        }
                    }
                    p += length;
                }
                4 => {
                    // BAM_CSOFT_CLIP
                    if p == pos {
                        clip5 = length;
                    } else {
                        clip3 = length;
                    }
                    p += length;
                }
                1 => {
                    // BAM_CINS
                    if length <= PATCH_INDELS {
                        let prev_ok = i > 0 && cigartuples[i - 1].0 == 0;
                        let next_ok = i + 1 < cigartuples.len() && cigartuples[i + 1].0 == 0;
                        if prev_ok && next_ok {
                            hole_to_remove.insert(segments.len().saturating_sub(1));
                        }
                    }
                }
                5 => {
                    // BAM_CHARD_CLIP
                    warn!("Hard clip was encountered! All mapping are assumed soft clipped");
                }
                _ => {}
            }
        }

        // Merge segments separated by small indels
        let holes: Vec<usize> = hole_to_remove.into_iter().collect();
        for (a, &b) in holes.iter().enumerate() {
            let adjusted = b - a;
            if adjusted + 1 < segments.len() {
                let fused_end = segments[adjusted + 1].1;
                segments.remove(adjusted + 1);
                segments[adjusted].1 = fused_end;
            }
        }

        (segments, ref_skip, clip5, clip3)
    }

    // ── GTF / annotation parsing ──────────────────────────────────────────────

    /// Reads a repeat-masking GTF file and populates `mask_ivls_by_chromstrand`.
    ///
    /// Adjacent or overlapping repeat intervals within `tolerance` bases are merged
    /// into a single `Feature`. The chromosome name is normalised by stripping a
    /// leading `chr`/`Chr`/`CHR` prefix (matching Python's `read_repeats` logic —
    /// only prefix stripping, NOT the full `normalize_chrom` pipeline).
    ///
    /// Python: read_repeats
    pub fn read_repeats(&mut self, gtf_file: &str, tolerance: i64) -> anyhow::Result<()> {
        debug!("Reading {gtf_file}, the file will be sorted in memory");
        let f = std::fs::File::open(gtf_file)
            .map_err(|e| anyhow::anyhow!("Cannot open GTF {gtf_file}: {e}"))?;
        let mut gtf_lines: Vec<String> = std::io::BufReader::new(f)
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.starts_with('#'))
            .collect();

        // Schwartzian transform — same rationale as in read_transcriptmodels.
        let sort_keys: Vec<(String, bool, i64)> = gtf_lines
            .iter()
            .map(|line| {
                let mut it = line.splitn(8, '\t');
                let chrom = it.next().unwrap_or("").to_string();
                let _ = it.next();
                let _ = it.next();
                let pos: i64 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let _ = it.next();
                let _ = it.next();
                let strand_plus = it.next().map_or(false, |s| s == "+");
                (chrom, strand_plus, pos)
            })
            .collect();
        let mut order: Vec<usize> = (0..gtf_lines.len()).collect();
        order.sort_by(|&a, &b| {
            sort_keys[a]
                .cmp(&sort_keys[b])
                .then(gtf_lines[a].cmp(&gtf_lines[b]))
        });
        drop(sort_keys);
        let gtf_lines: Vec<String> = order
            .into_iter()
            .map(|i| std::mem::take(&mut gtf_lines[i]))
            .collect();

        if gtf_lines.is_empty() {
            return Ok(());
        }

        // Strip only the "chr" prefix — matches Python's read_repeats behaviour.
        // (Python: `if chrom[:3].lower() == "chr": chrom = chrom[3:]`)
        // Do NOT use normalize_chrom() here; that function also maps chrM→MT and
        // splits on '_', which read_repeats must not do.
        let strip_chr = |s: &str| -> String {
            if s.len() >= 3 && s[..3].to_lowercase() == "chr" {
                s[3..].to_string()
            } else {
                s.to_string()
            }
        };

        let mut repeat_ivls_list: Vec<Feature> = Vec::new();
        let first = parse_gtf_fields(&gtf_lines[0]);
        let mut curr_chrom = strip_chr(&first.0);
        let mut curr_start = first.3;
        let mut curr_end = first.4;
        let mut curr_strand = first.6.clone();
        let mut curr_n: i64 = 1;
        let mut curr_chromstrand = format!("{curr_chrom}{curr_strand}");

        for line in &gtf_lines[1..] {
            let fields = parse_gtf_fields(line);
            // BUG 2 fix: only strip chr prefix, matching Python read_repeats
            let chrom = strip_chr(&fields.0);
            let start = fields.3;
            let end = fields.4;
            let strand = fields.6.clone();
            let chromstrand = format!("{chrom}{strand}");

            // BUG 1 fix: on chromstrand change, save the accumulated list (WITHOUT
            // the pending curr_start/curr_end interval) under the old key, then
            // reset the list.  curr_start/curr_end are intentionally NOT reset here
            // — they still represent the pending interval and will be tested against
            // the tolerance threshold below on the new chromstrand.
            // Python equivalent:
            //   self.mask_ivls_by_chromstrand[curr_chromstrand] = repeat_ivls_list
            //   repeat_ivls_list = []
            //   curr_chrom = chrom; curr_strand = strand
            //   curr_chromstrand = curr_chrom + curr_strand
            //   # then falls through to the `if start > curr_end + tolerance` check
            if chromstrand != curr_chromstrand {
                self.mask_ivls_by_chromstrand
                    .insert(curr_chromstrand.clone(), repeat_ivls_list);
                repeat_ivls_list = Vec::new();
                curr_chrom = chrom;
                curr_strand = strand;
                curr_chromstrand = format!("{curr_chrom}{curr_strand}");
                // Do NOT reset curr_start / curr_end / curr_n here.
            }

            if start > curr_end + tolerance {
                // Close the pending interval and push it into the current list.
                repeat_ivls_list.push(Feature::new(curr_start, curr_end, b'r', curr_n, None));
                curr_start = start;
                curr_end = end;
                curr_n = 1;
            } else {
                // Extend the current merged interval.
                curr_end = end;
                curr_n += 1;
            }
        }
        // Flush the last pending interval.
        repeat_ivls_list.push(Feature::new(curr_start, curr_end, b'r', curr_n, None));
        self.mask_ivls_by_chromstrand
            .insert(curr_chromstrand, repeat_ivls_list);

        let n: usize = self
            .mask_ivls_by_chromstrand
            .values()
            .map(|v| v.len())
            .sum();
        for list in self.mask_ivls_by_chromstrand.values_mut() {
            list.sort();
        }
        debug!("Processed masked annotation .gtf and generated {n} intervals to mask!");
        Ok(())
    }

    /// Python: assign_indexes_to_genes
    /// `gene_order` lists gene IDs in the order their first transcript was encountered in the
    /// sorted GTF — this is exactly Python's OrderedDict insertion order.
    pub fn assign_indexes_to_genes(
        &mut self,
        features: &BTreeMap<String, TranscriptModel>,
        gene_order: &[String],
    ) {
        debug!("Assigning indexes to genes");
        // Iterate in GTF-encounter order so gene indices match Python exactly.
        // gene_order[i] is the gene ID whose first transcript appeared i-th in the sorted GTF.
        for trmodel in gene_order
            .iter()
            .flat_map(|gid| features.values().filter(move |tm| &tm.geneid == gid))
        {
            if let Some(gi) = self.genes.get_mut(&trmodel.geneid) {
                if gi.start > trmodel.start() {
                    gi.start = trmodel.start();
                }
                if gi.end < trmodel.end() {
                    gi.end = trmodel.end();
                }
            } else {
                let ix = self.geneid2ix.len();
                self.geneid2ix.insert(trmodel.geneid.clone(), ix);
                self.genes.insert(
                    trmodel.geneid.clone(),
                    GeneInfo::new(
                        trmodel.genename.clone(),
                        trmodel.geneid.clone(),
                        &trmodel.chromstrand,
                        trmodel.start(),
                        trmodel.end(),
                    ),
                );
            }
        }
    }

    /// Python: peek_and_correct
    pub fn peek_and_correct(&self, gtf_lines: Vec<String>) -> Vec<String> {
        let regex_exonno = Regex::new(r#"exon_number "*?([\w]+)"#).unwrap();
        let regex_trid = Regex::new(r#"transcript_id "([^"]+)""#).unwrap();

        let mut flag = false;
        for lin in gtf_lines.iter().take(500) {
            let parts: Vec<&str> = lin.splitn(9, '\t').collect();
            if parts.len() < 9 {
                continue;
            }
            if parts[2] == "exon" && regex_exonno.find(parts[8]).is_none() {
                flag = true;
                break;
            }
        }
        if !flag {
            return gtf_lines;
        }

        warn!("The entry exon_number was not present in the gtf file. It will be inferred from the position.");

        let mut lines_minus: Vec<(String, i64, i64, String)> = Vec::new();
        let mut lines_plus: Vec<(String, i64, i64, String)> = Vec::new();

        for lin in &gtf_lines {
            let parts: Vec<&str> = lin.splitn(9, '\t').collect();
            if parts.len() < 9 || parts[2] != "exon" {
                continue;
            }
            let start: i64 = parts[3].parse().unwrap_or(0);
            let end: i64 = parts[4].parse().unwrap_or(0);
            let strand = parts[6];
            let trid = regex_trid
                .captures(parts[8])
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            if strand == "-" {
                lines_minus.push((trid, start, end, lin.clone()));
            } else {
                lines_plus.push((trid, start, end, lin.clone()));
            }
        }

        lines_plus.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        lines_minus.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut modified: Vec<String> = Vec::new();
        let mut current_trid = String::new();
        let mut exon_n = 1usize;
        for (trid, _start, _end, lin) in &lines_plus {
            if *trid != current_trid {
                current_trid = trid.clone();
                exon_n = 1;
            } else {
                exon_n += 1;
            }
            let trimmed = lin.trim_end_matches('\n');
            modified.push(format!("{trimmed} exon_number \"{exon_n}\";\n"));
        }
        // Python bug-faithful: minus lines also appended to modified_lines_plus
        exon_n = 1;
        current_trid = String::new();
        for (trid, _start, _end, lin) in lines_minus.iter().rev() {
            if *trid != current_trid {
                current_trid = trid.clone();
                exon_n = 1;
            } else {
                exon_n += 1;
            }
            let trimmed = lin.trim_end_matches('\n');
            modified.push(format!("{trimmed} exon_number \"{exon_n}\";\n"));
        }
        modified
    }

    /// Python: read_transcriptmodels
    pub fn read_transcriptmodels(&mut self, gtf_file: &str) -> anyhow::Result<()> {
        let regex_trid = Regex::new(r#"transcript_id "([^"]+)""#).unwrap();
        let regex_trname = Regex::new(r#"transcript_name "([^"]+)""#).unwrap();
        let regex_geneid = Regex::new(r#"gene_id "([^"]+)""#).unwrap();
        let regex_genename = Regex::new(r#"gene_name "([^"]+)""#).unwrap();
        let regex_exonno = Regex::new(r#"exon_number "*?([\w]+)"#).unwrap();

        let f = std::fs::File::open(gtf_file)
            .map_err(|e| anyhow::anyhow!("Cannot open GTF {gtf_file}: {e}"))?;
        // Read only exon lines up front: the main loop only uses feature_type == "exon".
        // Filtering during collection avoids allocating ~3× as many Strings for CDS/UTR/gene lines.
        // 1 MB read buffer reduces syscall overhead for large GTFs.
        let raw_lines: Vec<String> = std::io::BufReader::with_capacity(1 << 20, f)
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| {
                if l.starts_with('#') {
                    return false;
                }
                l.splitn(4, '\t').nth(2).map_or(false, |f| f == "exon")
            })
            .collect();

        let raw_lines = self.peek_and_correct(raw_lines);

        // Schwartzian transform: extract (chrom, strand_plus, pos) keys once O(n),
        // then sort indices O(n log n) comparing only those small keys.
        // The old sort_by(sorting_key) re-parsed and cloned each line on every comparison
        // call — O(n log n) full-line clones — which dominated runtime for large GTFs.
        let sort_keys: Vec<(String, bool, i64)> = raw_lines
            .iter()
            .map(|line| {
                let mut it = line.splitn(8, '\t');
                let chrom = it.next().unwrap_or("").to_string();
                let _ = it.next(); // source
                let _ = it.next(); // feature
                let pos: i64 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let _ = it.next(); // end
                let _ = it.next(); // score
                let strand_plus = it.next().map_or(false, |s| s == "+");
                (chrom, strand_plus, pos)
            })
            .collect();
        let mut order: Vec<usize> = (0..raw_lines.len()).collect();
        // Stable sort with full-line tiebreaker matches Python's sorted(..., key=sorting_key).
        // Python's key is (chrom, strand_plus, start, full_line); for ties on the first three
        // fields the full line is compared lexicographically.  sort_unstable_by ignores ties,
        // which produces a different gene-encounter order for same-start exons.
        order.sort_by(|&a, &b| {
            sort_keys[a]
                .cmp(&sort_keys[b])
                .then(raw_lines[a].cmp(&raw_lines[b]))
        });
        drop(sort_keys);
        let mut raw_lines = raw_lines;
        let gtf_lines: Vec<String> = order
            .into_iter()
            .map(|i| std::mem::take(&mut raw_lines[i]))
            .collect();

        let mut curr_chromstrand: Option<String> = None;
        let mut features: BTreeMap<String, TranscriptModel> = BTreeMap::new();
        let mut gene_order: Vec<String> = Vec::new();
        let mut seen_geneids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut nth_line = 0usize;

        for line in &gtf_lines {
            nth_line += 1;
            let parts: Vec<&str> = line.splitn(9, '\t').collect();
            if parts.len() < 9 {
                continue;
            }
            let raw_chrom = parts[0];
            let feature_type = parts[2];
            let start_str = parts[3];
            let end_str = parts[4];
            let strand = parts[6];
            let tags = parts[8];

            let chrom = if raw_chrom.len() >= 3 && raw_chrom[..3].to_lowercase() == "chr" {
                raw_chrom[3..].to_string()
            } else {
                raw_chrom.to_string()
            };
            let chromstrand = format!("{chrom}{strand}");

            if Some(&chromstrand) != curr_chromstrand.as_ref() {
                if let Some(ref cs) = curr_chromstrand {
                    if self.annotations_by_chrm_strand.contains_key(&chromstrand) {
                        anyhow::bail!(
                            "Genome annotation gtf file is not sorted correctly! \
                             Run: sort -k1,1 -k7,7 -k4,4n -o [GTF_OUTFILE] [GTF_INFILE]"
                        );
                    }
                    debug!("Done with {cs} [line {}]", nth_line - 1);
                    self.assign_indexes_to_genes(&features, &gene_order);
                    self.annotations_by_chrm_strand.insert(cs.clone(), features);
                    debug!("Seen {} genes until now", self.geneid2ix.len());
                }
                features = BTreeMap::new();
                gene_order = Vec::new();
                seen_geneids = std::collections::HashSet::new();
                debug!("Parsing Chromosome {chrom} strand {strand} [line {nth_line}]");
                curr_chromstrand = Some(chromstrand.clone());
            }

            if feature_type == "exon" {
                let trid = regex_trid
                    .captures(tags)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let trname = regex_trname
                    .captures(tags)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_else(|| trid.clone());
                let geneid = regex_geneid
                    .captures(tags)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let genename = regex_genename
                    .captures(tags)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_else(|| geneid.clone());
                let exonno_str = regex_exonno
                    .captures(tags)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().to_string())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "The genome annotation .gtf file does not contain exon_number."
                        )
                    })?;
                let exonno: i64 = exonno_str.parse().unwrap_or(1);
                let start: i64 = start_str.parse().unwrap_or(0);
                let end: i64 = end_str.parse().unwrap_or(0);

                if seen_geneids.insert(geneid.clone()) {
                    gene_order.push(geneid.clone());
                }
                let tm = features.entry(trid.clone()).or_insert_with(|| {
                    TranscriptModel::new(
                        trid.clone(),
                        trname,
                        geneid,
                        genename,
                        chromstrand.clone(),
                    )
                });
                tm.append_exon(Feature::new(start, end, b'e', exonno, None));
            }
        }

        // flush last chromosome
        if let Some(cs) = curr_chromstrand {
            self.assign_indexes_to_genes(&features, &gene_order);
            self.annotations_by_chrm_strand.insert(cs.clone(), features);
            debug!("Done with {cs} [line {}]", nth_line.saturating_sub(1));
        }

        debug!(
            "Fixing corner cases of transcript models containing intron longer than {}Kbp",
            LONGEST_INTRON_ALLOWED / 1000
        );
        for tmodels in self.annotations_by_chrm_strand.values_mut() {
            for tm in tmodels.values_mut() {
                tm.chop_if_long_intron(LONGEST_INTRON_ALLOWED);
            }
        }

        // Re-sort by start position within each chromstrand (mirrors Python OrderedDict resort)
        for tmodels in self.annotations_by_chrm_strand.values_mut() {
            let mut vec: Vec<(String, TranscriptModel)> =
                std::mem::take(tmodels).into_iter().collect();
            vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            *tmodels = vec.into_iter().collect();
        }

        Ok(())
    }

    /// Build tms_flat and feature_indexes once.
    /// Assigns transcript_model_idx on all features so mark_up_introns can propagate is_validated.
    pub fn build_feature_indexes(&mut self) {
        let mut chromstrand_keys: Vec<String> =
            self.annotations_by_chrm_strand.keys().cloned().collect();
        chromstrand_keys.sort();

        // Assign global tm_idx to each TM's features in annotations_by_chrm_strand
        let mut global_idx = 0usize;
        for cs in &chromstrand_keys {
            let tmodels = self.annotations_by_chrm_strand.get_mut(cs).unwrap();
            for (_trid, tm) in tmodels.iter_mut() {
                for feat in &mut tm.list_features {
                    feat.transcript_model_idx = Some(global_idx);
                }
                global_idx += 1;
            }
        }

        // Build tms_flat in the same order (clone after idx assignment)
        self.tms_flat = Vec::new();
        for cs in &chromstrand_keys {
            for tm in self.annotations_by_chrm_strand[cs].values() {
                self.tms_flat.push(tm.clone());
            }
        }

        // Build feature_indexes: flatten all features per chromstrand IN START-POSITION ORDER.
        // Python's annotations_by_chrm_strand is an OrderedDict sorted by TM start position.
        // Rust's BTreeMap sorts by trid. Sort TMs by start position so the stable sort in
        // FeatureIndex::new breaks ties (same start,end) identically to Python.
        self.feature_indexes = HashMap::new();
        for cs in &chromstrand_keys {
            let mut sorted_tms: Vec<&TranscriptModel> =
                self.annotations_by_chrm_strand[cs].values().collect();
            // Python's secondary sort key for equal-start TMs is GTF insertion order, which
            // equals the first-exon-end of each TM (the lex sort of the full GTF line breaks
            // start ties by the end field). Use (start, end_of_first_feature) to match Python.
            sorted_tms.sort_by(|a, b| {
                let a_start = a.list_features.first().map_or(0, |f| f.start);
                let b_start = b.list_features.first().map_or(0, |f| f.start);
                let a_end = a.list_features.first().map_or(0, |f| f.end);
                let b_end = b.list_features.first().map_or(0, |f| f.end);
                a_start.cmp(&b_start).then(a_end.cmp(&b_end))
            });
            let mut all_features: Vec<Feature> = Vec::new();
            for tm in sorted_tms {
                all_features.extend(tm.list_features.iter().cloned());
            }
            let fi = FeatureIndex::new(all_features);
            self.feature_indexes.insert(cs.clone(), fi);
        }
    }

    // ── Main pipeline methods ─────────────────────────────────────────────────

    /// Marks up introns that have reads supporting exon-intron junctions.
    /// Processes a BAM file to identify which introns are supported by reads
    /// spanning the exon-intron boundary. Sets `is_validated` on matching
    /// `Feature` entries so that `count()` can distinguish confident unspliced
    /// molecules.
    ///
    /// Python: mark_up_introns
    #[cfg(feature = "bam")]
    pub fn mark_up_introns(&mut self, bamfile: &str, multimap: bool) -> anyhow::Result<()> {
        if !self.logic.perform_validation_markup() {
            return Ok(());
        }

        if self.feature_indexes.is_empty() {
            self.build_feature_indexes();
        }

        let mut currchrom = String::new();
        let mut set_chromosomes_seen: HashSet<String> = HashSet::new();

        let mut reader = bam::Reader::from_path(bamfile)
            .map_err(|e| anyhow::anyhow!("Cannot open BAM {bamfile}: {e}"))?;
        let header = reader.header().clone();
        let mut rec = Record::new();
        let mut i = 0usize;

        loop {
            match reader.read(&mut rec) {
                Some(Ok(())) => {}
                None => break,
                Some(Err(e)) => return Err(anyhow::anyhow!("BAM read error: {e}")),
            }
            i += 1;
            if i % 10_000_000 == 0 {
                debug!("Read first {} million reads", i / 1_000_000);
            }
            if rec.is_unmapped() {
                continue;
            }
            if !multimap && rec.aux(b"NH").map(aux_to_i64).unwrap_or(1) != 1 {
                continue;
            }
            if self.filter_mode {
                let bc_raw = aux_to_string(
                    rec.aux(self.cellbarcode_str.as_bytes())
                        .unwrap_or(rust_htslib::bam::record::Aux::String("")),
                );
                let bc = bc_raw.split('-').next().unwrap_or("").to_string();
                if !self.valid_bcset.contains(&bc) {
                    continue;
                }
            }

            let chrom_raw = header.tid2name(rec.tid() as u32);
            let chrom = normalize_chrom(std::str::from_utf8(chrom_raw).unwrap_or(""));
            let strand = if rec.is_reverse() { '-' } else { '+' };
            let pos = rec.pos() + 1;

            let cigar: Vec<(u32, u32)> = rec
                .cigar()
                .iter()
                .map(|c| match c {
                    bam::record::Cigar::Match(l) => (0u32, *l),
                    bam::record::Cigar::Ins(l) => (1u32, *l),
                    bam::record::Cigar::Del(l) => (2u32, *l),
                    bam::record::Cigar::RefSkip(l) => (3u32, *l),
                    bam::record::Cigar::SoftClip(l) => (4u32, *l),
                    bam::record::Cigar::HardClip(l) => (5u32, *l),
                    other => (255u32, other.len()),
                })
                .collect();
            let (segments, ref_skipped, clip5, clip3) = Self::parse_cigar_tuple(&cigar, pos);
            if segments.is_empty() {
                continue;
            }

            let read_obj = Read::new(
                String::new(),
                String::new(),
                String::new(),
                chrom.clone(),
                strand,
                pos,
                segments,
                Some(clip5),
                Some(clip3),
                ref_skipped,
            );

            // Discard reads with implausibly large genomic spans (matches Python iter_alignments).
            if read_obj.span() > 3_000_000 {
                warn!("Trashing read, too long span: {}", read_obj.span());
                continue;
            }

            // Don't consider spliced reads in markup
            if read_obj.is_spliced() {
                continue;
            }

            if chrom != currchrom {
                if set_chromosomes_seen.contains(&chrom) {
                    anyhow::bail!(
                        "Input .bam file should be chromosome-sorted. \
                         (Hint: use `samtools sort {bamfile}`)"
                    );
                }
                set_chromosomes_seen.insert(chrom.clone());
                debug!("Marking up chromosome {chrom}");
                currchrom = chrom.clone();
            }

            let cs_key = format!("{chrom}{strand}");
            if !self.annotations_by_chrm_strand.contains_key(&cs_key) {
                continue;
            }

            // SAFETY: tms_flat is not mutated inside this loop body
            let tms: &[TranscriptModel] =
                unsafe { std::slice::from_raw_parts(self.tms_flat.as_ptr(), self.tms_flat.len()) };
            if let Some(iif) = self.feature_indexes.get_mut(&cs_key) {
                iif.mark_overlapping_ivls(&read_obj, tms);
            }
        }

        // Copy is_validated back to annotations and tms_flat
        self.sync_validated_to_annotations();
        Ok(())
    }

    /// Propagate is_validated from feature_indexes back to annotations_by_chrm_strand and tms_flat.
    fn sync_validated_to_annotations(&mut self) {
        for (cs, fi) in &self.feature_indexes {
            if let Some(tmodels) = self.annotations_by_chrm_strand.get_mut(cs) {
                for ivl in &fi.ivls {
                    if !ivl.is_validated {
                        continue;
                    }
                    if let Some(tm_idx) = ivl.transcript_model_idx {
                        if tm_idx < self.tms_flat.len() {
                            let trid = self.tms_flat[tm_idx].trid.clone();
                            if let Some(tm) = tmodels.get_mut(&trid) {
                                for feat in &mut tm.list_features {
                                    if feat.start == ivl.start && feat.end == ivl.end {
                                        feat.is_validated = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Rebuild tms_flat from updated annotations
        let mut chromstrand_keys: Vec<String> =
            self.annotations_by_chrm_strand.keys().cloned().collect();
        chromstrand_keys.sort();
        let mut flat: Vec<TranscriptModel> = Vec::new();
        for cs in &chromstrand_keys {
            for tm in self.annotations_by_chrm_strand[cs].values() {
                flat.push(tm.clone());
            }
        }
        self.tms_flat = flat;
    }

    /// Counts spliced, unspliced, and ambiguous molecules.
    /// Processes BAM reads in cell batches and returns `(dict_list_arrays,
    /// cell_bcs_order)` where each entry in `dict_list_arrays` is a layer name
    /// (e.g. `"spliced"`) mapping to a list of per-batch count matrices.
    ///
    /// Python: count — main counting loop
    #[cfg(feature = "bam")]
    pub fn count(
        &mut self,
        bamfile: &str,
        multimap: bool,
        cell_batch_size: usize,
    ) -> anyhow::Result<(HashMap<String, Vec<ndarray::Array2<u32>>>, Vec<String>)> {
        // Reuse or build feature_indexes; reset scanning positions
        if self.feature_indexes.is_empty() {
            self.build_feature_indexes();
        } else {
            for fi in self.feature_indexes.values_mut() {
                fi.reset();
            }
        }

        // Build mask_indexes from mask_ivls
        self.mask_indexes = HashMap::new();
        for (cs, ivls) in &self.mask_ivls_by_chromstrand {
            self.mask_indexes
                .insert(cs.clone(), FeatureIndex::new(ivls.clone()));
        }

        // Log intron validation summary
        let mut n_is_intron = 0usize;
        let mut n_is_intron_valid = 0usize;
        let mut unique_valid: HashSet<(i64, i64)> = HashSet::new();
        for fi in self.feature_indexes.values() {
            for ivl in &fi.ivls {
                if ivl.kind == b'i' {
                    n_is_intron += 1;
                }
                if ivl.is_validated {
                    n_is_intron_valid += 1;
                    unique_valid.insert((ivl.start, ivl.end));
                }
            }
        }
        debug!("Validated {n_is_intron_valid} introns (of which unique intervals {}) out of {n_is_intron} total.",
               unique_valid.len());

        let mut cell_bcs_order: Vec<String> = Vec::new();
        let mut dict_list_arrays: HashMap<String, Vec<ndarray::Array2<u32>>> = HashMap::new();
        for layer_name in self.logic.layers() {
            dict_list_arrays.insert(layer_name.to_string(), Vec::new());
        }

        // `cell_batch` holds composite "{sample}|{bc}" cell identities and drives
        // column allocation; `cell_batch_bcs` holds the raw cell barcodes and
        // drives the flush boundary. They are decoupled because the BAM is sorted
        // by cell barcode only: all reads of one barcode are contiguous, but their
        // sample tags are interleaved. Flushing on the raw barcode guarantees every
        // read of a barcode (across all samples) lands in one batch, so a single
        // (sample, barcode) cell is never split across batches — which would
        // corrupt UMI deduplication.
        let mut cell_batch: HashSet<String> = HashSet::new();
        let mut cell_batch_bcs: HashSet<String> = HashSet::new();
        let mut reads_to_count: Vec<Read> = Vec::new();
        let mut nth = 0usize;
        let mut no_sample_tag_count = 0usize;

        let mut reader = bam::Reader::from_path(bamfile)
            .map_err(|e| anyhow::anyhow!("Cannot open BAM {bamfile}: {e}"))?;
        let header = reader.header().clone();
        let mut rec = Record::new();

        let mut pending_read: Option<Read> = None;
        let mut exhausted = false;

        loop {
            // Fetch next valid Read
            let r_opt: Option<Read> = if exhausted {
                None
            } else {
                let mut found = None;
                loop {
                    match reader.read(&mut rec) {
                        Some(Ok(())) => {}
                        None => {
                            exhausted = true;
                            break;
                        }
                        Some(Err(e)) => return Err(anyhow::anyhow!("BAM read error: {e}")),
                    }
                    if rec.is_unmapped() {
                        continue;
                    }
                    if !multimap && rec.aux(b"NH").map(aux_to_i64).unwrap_or(1) != 1 {
                        continue;
                    }
                    let bc_res = if self.onefilepercell {
                        Ok(std::path::Path::new(bamfile)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(bamfile)
                            .to_string())
                    } else {
                        self.normal_cell_barcode_get(&rec)
                    };
                    let umi_res = self.extract_umi(&rec);
                    let (bc, umi) = match (bc_res, umi_res) {
                        (Ok(b), Ok(u)) => (b, u),
                        (Err(e1), _) => {
                            debug!("bc_err: {e1}");
                            continue;
                        }
                        (_, Err(e2)) => {
                            debug!("umi_err: {e2}");
                            continue;
                        }
                    };
                    // Sample demultiplexing (e.g. BD Rhapsody `ST`). When no
                    // sample tag is configured, or the read lacks it, `sample`
                    // is the empty string and the cell identity collapses to the
                    // barcode alone (single-sample behaviour, unchanged).
                    // When demultiplexing is active, a read with no sample tag has
                    // no defined sample of origin (typically undetermined/multiplet
                    // noise), so it is dropped rather than pooled into a phantom
                    // sample-less cell. With demultiplexing off, `sample` is "".
                    let sample = match &self.sample_tag {
                        Some(tag) => match rec.aux(tag.as_bytes()) {
                            Ok(aux) => aux_to_string(aux),
                            Err(_) => {
                                no_sample_tag_count += 1;
                                continue;
                            }
                        },
                        None => String::new(),
                    };
                    if !self.valid_bcset.contains(&bc) {
                        if self.filter_mode {
                            continue;
                        } else {
                            self.valid_bcset.insert(bc.clone());
                        }
                    }
                    let chrom_raw = header.tid2name(rec.tid() as u32);
                    let chrom = normalize_chrom(std::str::from_utf8(chrom_raw).unwrap_or(""));
                    let strand = if rec.is_reverse() { '-' } else { '+' };
                    let pos = rec.pos() + 1;
                    let cigar: Vec<(u32, u32)> = rec
                        .cigar()
                        .iter()
                        .map(|c| match c {
                            bam::record::Cigar::Match(l) => (0u32, *l),
                            bam::record::Cigar::Ins(l) => (1u32, *l),
                            bam::record::Cigar::Del(l) => (2u32, *l),
                            bam::record::Cigar::RefSkip(l) => (3u32, *l),
                            bam::record::Cigar::SoftClip(l) => (4u32, *l),
                            bam::record::Cigar::HardClip(l) => (5u32, *l),
                            other => (255u32, other.len()),
                        })
                        .collect();
                    let (segments, ref_skipped, clip5, clip3) =
                        Self::parse_cigar_tuple(&cigar, pos);
                    if segments.is_empty() {
                        continue;
                    }
                    let read_obj = Read::new(
                        bc,
                        umi,
                        sample,
                        chrom,
                        strand,
                        pos,
                        segments,
                        Some(clip5),
                        Some(clip3),
                        ref_skipped,
                    );
                    if read_obj.span() > 3_000_000 {
                        warn!("Trashing read, too long span");
                        continue;
                    }
                    found = Some(read_obj);
                    break;
                }
                found
            };

            // Determine if we should flush
            let flush = match &r_opt {
                None => !cell_batch_bcs.is_empty(),
                Some(r) => {
                    cell_batch_bcs.len() == cell_batch_size && !cell_batch_bcs.contains(&r.bc)
                }
            };

            if flush {
                nth += 1;
                debug!(
                    "Counting for batch {nth}, containing {} cells and {} reads",
                    cell_batch.len(),
                    reads_to_count.len()
                );

                let (dict_layer_columns, list_bcs) =
                    self.count_cell_batch_inner(&cell_batch, &mut reads_to_count);

                if !self.filter_mode {
                    warn!("The barcode selection mode is off, no cell events will be identified by <80 counts");
                    let spliced = dict_layer_columns
                        .get("spliced")
                        .map(|a| a.sum_axis(ndarray::Axis(0)));
                    let unspliced = dict_layer_columns
                        .get("unspliced")
                        .map(|a| a.sum_axis(ndarray::Axis(0)));
                    let tot_mol: ndarray::Array1<u32> = match (spliced, unspliced) {
                        (Some(s), Some(u)) => s + u,
                        (Some(s), None) => s,
                        (None, Some(u)) => u,
                        (None, None) => ndarray::Array1::zeros(list_bcs.len()),
                    };
                    let keep: Vec<bool> = tot_mol.iter().map(|&v| v > 80).collect();
                    let n_kept = keep.iter().filter(|&&k| k).count();
                    debug!(
                        "{} of {} barcodes pass >80 count threshold",
                        n_kept,
                        list_bcs.len()
                    );
                    cell_bcs_order.extend(
                        list_bcs
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| keep[*i])
                            .map(|(_, bc)| bc.clone()),
                    );
                    for (layer_name, arr) in &dict_layer_columns {
                        let cols: Vec<ndarray::ArrayView1<u32>> = (0..arr.ncols())
                            .filter(|&j| keep[j])
                            .map(|j| arr.column(j))
                            .collect();
                        if cols.is_empty() {
                            dict_list_arrays
                                .get_mut(layer_name)
                                .unwrap()
                                .push(ndarray::Array2::zeros((arr.nrows(), 0)));
                        } else {
                            let stacked = ndarray::stack(ndarray::Axis(1), &cols).unwrap();
                            dict_list_arrays.get_mut(layer_name).unwrap().push(stacked);
                        }
                    }
                } else {
                    cell_bcs_order.extend(list_bcs);
                    for (layer_name, arr) in dict_layer_columns {
                        dict_list_arrays.get_mut(&layer_name).unwrap().push(arr);
                    }
                }

                cell_batch.clear();
                cell_batch_bcs.clear();
                reads_to_count.clear();
                for fi in self.feature_indexes.values_mut() {
                    fi.reset();
                }
                for mi in self.mask_indexes.values_mut() {
                    mi.reset();
                }
            }

            match r_opt {
                Some(r) => {
                    cell_batch_bcs.insert(r.bc.clone());
                    cell_batch.insert(format!("{}|{}", r.sample, r.bc));
                    reads_to_count.push(r);
                }
                None => break,
            }
        }
        let _ = pending_read;

        if self.sample_tag.is_some() {
            debug!("{no_sample_tag_count} reads dropped: missing sample tag");
        }
        debug!("Counting done!");
        Ok((dict_list_arrays, cell_bcs_order))
    }

    /// Dispatch to stranded / non_stranded batch counting.
    fn count_cell_batch_inner(
        &mut self,
        cell_batch: &HashSet<String>,
        reads_to_count: &mut Vec<Read>,
    ) -> (HashMap<String, ndarray::Array2<u32>>, Vec<String>) {
        if self.logic.stranded() {
            if self.logic.accept_discordant() {
                self.count_cell_batch_stranded_discordant_inner(cell_batch, reads_to_count)
            } else {
                self.count_cell_batch_stranded_inner(cell_batch, reads_to_count)
            }
        } else {
            self.count_cell_batch_non_stranded_inner(cell_batch, reads_to_count)
        }
    }

    /// Python: _count_cell_batch_stranded
    fn count_cell_batch_stranded_inner(
        &mut self,
        cell_batch: &HashSet<String>,
        reads_to_count: &mut Vec<Read>,
    ) -> (HashMap<String, ndarray::Array2<u32>>, Vec<String>) {
        let mut molitems: HashMap<String, Molitem> = HashMap::new();
        reads_to_count.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut repeats_reads_count = 0usize;
        let mut no_key_count = 0usize;
        let mut no_overlap_count = 0usize;
        let mut mapped_count = 0usize;
        for r in reads_to_count.iter() {
            let cs_key = format!("{}{}", r.chrom, r.strand);
            let mask_enclosed = self
                .mask_indexes
                .get_mut(&cs_key)
                .map(|iim| iim.has_ivls_enclosing(r))
                .unwrap_or(false);
            if mask_enclosed {
                repeats_reads_count += 1;
                continue;
            }
            if !self.feature_indexes.contains_key(&cs_key) {
                no_key_count += 1;
                continue;
            }
            let mappings_record = self
                .feature_indexes
                .get_mut(&cs_key)
                .map(|ii| ii.find_overlapping_ivls(r))
                .unwrap_or_default();
            if !mappings_record.is_empty() {
                let bcumi = format!("{}|{}${}", r.sample, r.bc, r.umi);
                molitems
                    .entry(bcumi)
                    .or_default()
                    .add_mappings_record(mappings_record);
                mapped_count += 1;
            } else {
                no_overlap_count += 1;
            }
        }
        if no_key_count > 0 {
            let example_key = reads_to_count
                .iter()
                .map(|r| format!("{}{}", r.chrom, r.strand))
                .find(|k| !self.feature_indexes.contains_key(k));
            let idx_keys: Vec<&String> = self.feature_indexes.keys().collect();
            debug!(
                "Missing cs_key example: {example_key:?}; index keys (first 4): {:?}",
                &idx_keys[..idx_keys.len().min(4)]
            );
        }
        debug!(
            "batch: {repeats_reads_count} repeat-masked, {no_key_count} no-key, \
                {no_overlap_count} no-overlap, {mapped_count} mapped"
        );
        self.finalize_batch(cell_batch, &molitems)
    }

    /// Python: _count_cell_batch_stranded_discordant
    fn count_cell_batch_stranded_discordant_inner(
        &mut self,
        cell_batch: &HashSet<String>,
        reads_to_count: &mut Vec<Read>,
    ) -> (HashMap<String, ndarray::Array2<u32>>, Vec<String>) {
        let mut molitems: HashMap<String, Molitem> = HashMap::new();
        reads_to_count.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut repeats_reads_count = 0usize;
        for r in reads_to_count.iter() {
            let cs_key = format!("{}{}", r.chrom, r.strand);
            let cs_rev_key = format!("{}{}", r.chrom, reverse_strand(r.strand));
            let mask_enclosed = self
                .mask_indexes
                .get_mut(&cs_key)
                .map(|iim| iim.has_ivls_enclosing(r))
                .unwrap_or(false);
            let mappings_record = if mask_enclosed {
                repeats_reads_count += 1;
                let rev_mask = self
                    .mask_indexes
                    .get_mut(&cs_rev_key)
                    .map(|iimr| iimr.has_ivls_enclosing(r))
                    .unwrap_or(false);
                if rev_mask {
                    continue;
                }
                self.feature_indexes
                    .get_mut(&cs_rev_key)
                    .map(|iir| iir.find_overlapping_ivls(r))
                    .unwrap_or_default()
            } else {
                self.feature_indexes
                    .get_mut(&cs_key)
                    .map(|ii| ii.find_overlapping_ivls(r))
                    .unwrap_or_default()
            };
            if !mappings_record.is_empty() {
                let bcumi = format!("{}|{}${}", r.sample, r.bc, r.umi);
                molitems
                    .entry(bcumi)
                    .or_default()
                    .add_mappings_record(mappings_record);
            }
        }
        debug!("{repeats_reads_count} reads not considered because fully enclosed in repeat masked regions");
        self.finalize_batch(cell_batch, &molitems)
    }

    /// Python: _count_cell_batch_non_stranded
    fn count_cell_batch_non_stranded_inner(
        &mut self,
        cell_batch: &HashSet<String>,
        reads_to_count: &mut Vec<Read>,
    ) -> (HashMap<String, ndarray::Array2<u32>>, Vec<String>) {
        let mut molitems: HashMap<String, Molitem> = HashMap::new();
        reads_to_count.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut repeats_reads_count = 0usize;
        for r in reads_to_count.iter() {
            let cs_key = format!("{}{}", r.chrom, r.strand);
            let cs_rev_key = format!("{}{}", r.chrom, reverse_strand(r.strand));
            let mask_fwd = self
                .mask_indexes
                .get_mut(&cs_key)
                .map(|iim| iim.has_ivls_enclosing(r))
                .unwrap_or(false);
            let mask_rev = self
                .mask_indexes
                .get_mut(&cs_rev_key)
                .map(|iimr| iimr.has_ivls_enclosing(r))
                .unwrap_or(false);
            if mask_fwd || mask_rev {
                repeats_reads_count += 1;
                continue;
            }
            let mappings_record = self
                .feature_indexes
                .get_mut(&cs_key)
                .map(|ii| ii.find_overlapping_ivls(r))
                .unwrap_or_default();
            if !mappings_record.is_empty() {
                let bcumi = format!("{}|{}${}", r.sample, r.bc, r.umi);
                molitems
                    .entry(bcumi)
                    .or_default()
                    .add_mappings_record(mappings_record);
            }
            let mappings_record_r = self
                .feature_indexes
                .get_mut(&cs_rev_key)
                .map(|iir| iir.find_overlapping_ivls(r))
                .unwrap_or_default();
            if !mappings_record_r.is_empty() {
                let bcumi = format!("{}|{}${}", r.sample, r.bc, r.umi);
                molitems
                    .entry(bcumi)
                    .or_default()
                    .add_mappings_record(mappings_record_r);
            }
        }
        debug!("{repeats_reads_count} reads in repeat masked regions");
        self.finalize_batch(cell_batch, &molitems)
    }

    /// Shared finalization: run logic.count for each molitem, return layer columns.
    fn finalize_batch(
        &self,
        cell_batch: &HashSet<String>,
        molitems: &HashMap<String, Molitem>,
    ) -> (HashMap<String, ndarray::Array2<u32>>, Vec<String>) {
        let n_genes = self.geneid2ix.len();
        let n_cells = cell_batch.len();
        let mut dict_layers_columns: HashMap<String, ndarray::Array2<u32>> = HashMap::new();
        for layer_name in self.logic.layers() {
            dict_layers_columns.insert(
                layer_name.to_string(),
                ndarray::Array2::zeros((n_genes, n_cells)),
            );
        }
        let bc2idx: HashMap<String, usize> = cell_batch
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, bc)| (bc, i))
            .collect();

        let mut failures = 0usize;
        let mut ok = 0usize;
        let mut skipped_bc = 0usize;
        for (bcumi, molitem) in molitems {
            let bc = bcumi.split('$').next().unwrap_or("");
            if let Some(&bcidx) = bc2idx.get(bc) {
                let rcode = self.logic.count(
                    molitem,
                    bcidx,
                    &mut dict_layers_columns,
                    &self.geneid2ix,
                    &self.tms_flat,
                );
                match rcode {
                    Some(0) => ok += 1,
                    _ => failures += 1,
                }
            } else {
                skipped_bc += 1;
            }
        }
        debug!("logic.count results: ok={ok} failures={failures} skipped_bc={skipped_bc}");
        if !molitems.is_empty() && failures > molitems.len() / 4 {
            warn!(
                "More than 25% ({:.1}%) of molitems trashed",
                100.0 * failures as f64 / molitems.len() as f64
            );
        }

        let mut idx2bc: Vec<(String, usize)> = bc2idx.into_iter().collect();
        idx2bc.sort_by_key(|(_, i)| *i);
        let list_bcs: Vec<String> = idx2bc.into_iter().map(|(bc, _)| bc).collect();

        (dict_layers_columns, list_bcs)
    }

    /// Write results to a loom-format HDF5 file.
    pub fn dump_loom(
        &self,
        outfile: &str,
        dict_list_arrays: &HashMap<String, Vec<ndarray::Array2<u32>>>,
        cell_bcs_order: &[String],
        sample_ids: Option<&[String]>,
    ) -> anyhow::Result<()> {
        use hdf5_pure_rust::WritableFile;
        use ndarray::Array2;

        // Concatenate batches horizontally per layer
        let mut layers: HashMap<String, Array2<u32>> = HashMap::new();
        for (layer_name, arrays) in dict_list_arrays {
            if arrays.is_empty() {
                layers.insert(layer_name.clone(), Array2::zeros((self.geneid2ix.len(), 0)));
                continue;
            }
            let views: Vec<ndarray::ArrayView2<u32>> = arrays.iter().map(|a| a.view()).collect();
            let concatenated = ndarray::concatenate(ndarray::Axis(1), &views)
                .map_err(|e| anyhow::anyhow!("Concatenation error: {e}"))?;
            layers.insert(layer_name.clone(), concatenated);
        }

        let n_genes = self.geneid2ix.len();
        let mut gene_ids: Vec<String> = vec![String::new(); n_genes];
        let mut gene_names: Vec<String> = vec![String::new(); n_genes];
        let mut chromosomes: Vec<String> = vec![String::new(); n_genes];
        let mut strands: Vec<String> = vec![String::new(); n_genes];
        let mut starts: Vec<i64> = vec![0; n_genes];
        let mut ends: Vec<i64> = vec![0; n_genes];

        for (geneid, &ix) in &self.geneid2ix {
            gene_ids[ix] = geneid.clone();
            if let Some(gi) = self.genes.get(geneid) {
                gene_names[ix] = gi.genename.clone();
                chromosomes[ix] = gi.chrom.clone();
                strands[ix] = gi.strand.to_string();
                starts[ix] = gi.start;
                ends[ix] = gi.end;
            }
        }

        if std::path::Path::new(outfile).exists() {
            std::fs::remove_file(outfile)?;
        }
        let mut wf = WritableFile::create(outfile)
            .map_err(|e| anyhow::anyhow!("HDF5 create error: {e:?}"))?;

        // matrix = sum of all layers as float32 (Python: total = sum(layers))
        {
            let first = layers.values().next();
            if let Some(first_arr) = first {
                let (nr, nc) = first_arr.dim();
                let mut total: ndarray::Array2<f32> = ndarray::Array2::zeros((nr, nc));
                for arr in layers.values() {
                    total = total + arr.mapv(|v| v as f32);
                }
                let chunk0 = (64usize.min(nr)).max(1) as u64;
                let chunk1 = (64usize.min(nc)).max(1) as u64;
                let flat: Vec<f32> = total.iter().cloned().collect();
                wf.new_dataset_builder("matrix")
                    .shape(&[nr as u64, nc as u64])
                    .chunk(&[chunk0, chunk1])
                    .deflate(2)
                    .write::<f32>(&flat)
                    .map_err(|e| anyhow::anyhow!("HDF5 write matrix: {e:?}"))?;
            }
        }
        // Write all layers (spliced, unspliced, ambiguous) into layers/ group
        {
            let mut layers_group = wf
                .create_group("layers")
                .map_err(|e| anyhow::anyhow!("HDF5 create layers group: {e:?}"))?;
            let mut sorted_layer_names: Vec<&String> = layers.keys().collect();
            sorted_layer_names.sort();
            // Output element width is controlled by `loom_numeric_dtype`.
            // Counts always accumulate as u32 (matching Python's unbounded ints);
            // "uint32" (default) writes them losslessly, "uint16" saturates to
            // u16::MAX to mirror legacy narrow output (with a warning if clamped).
            let narrow = self.loom_numeric_dtype == "uint16";
            for layer_name in sorted_layer_names {
                let arr = &layers[layer_name];
                let (nr, nc) = arr.dim();
                let chunk0 = (64usize.min(nr)).max(1) as u64;
                let chunk1 = (64usize.min(nc)).max(1) as u64;
                if narrow {
                    let overflow = arr.iter().filter(|&&v| v > u16::MAX as u32).count();
                    if overflow > 0 {
                        warn!(
                            "Layer '{layer_name}': {overflow} value(s) exceed 65535 and were \
                             saturated by --dtype uint16; use --dtype uint32 to avoid loss"
                        );
                    }
                    let flat: Vec<u16> =
                        arr.iter().map(|&v| v.min(u16::MAX as u32) as u16).collect();
                    layers_group
                        .new_dataset_builder(layer_name)
                        .shape(&[nr as u64, nc as u64])
                        .chunk(&[chunk0, chunk1])
                        .deflate(2)
                        .write::<u16>(&flat)
                        .map_err(|e| anyhow::anyhow!("HDF5 write layer '{layer_name}': {e:?}"))?;
                } else {
                    let flat: Vec<u32> = arr.iter().cloned().collect();
                    layers_group
                        .new_dataset_builder(layer_name)
                        .shape(&[nr as u64, nc as u64])
                        .chunk(&[chunk0, chunk1])
                        .deflate(2)
                        .write::<u32>(&flat)
                        .map_err(|e| anyhow::anyhow!("HDF5 write layer '{layer_name}': {e:?}"))?;
                }
            }
        }

        // row_attrs group: numeric datasets + string datasets
        {
            let n_starts = ndarray::Array1::from(starts);
            let n_ends = ndarray::Array1::from(ends);
            let chunk = (64usize.min(n_genes)).max(1) as u64;
            let gene_slices: Vec<&str> = gene_names.iter().map(|s| s.as_str()).collect();
            let id_slices: Vec<&str> = gene_ids.iter().map(|s| s.as_str()).collect();
            let chr_slices: Vec<&str> = chromosomes.iter().map(|s| s.as_str()).collect();
            let strand_slices: Vec<&str> = strands.iter().map(|s| s.as_str()).collect();
            let mut rg = wf
                .create_group("row_attrs")
                .map_err(|e| anyhow::anyhow!("HDF5 create row_attrs: {e:?}"))?;
            rg.new_dataset_builder("Start")
                .shape(&[n_genes as u64])
                .chunk(&[chunk])
                .deflate(2)
                .write::<i64>(n_starts.as_slice().unwrap())
                .map_err(|e| anyhow::anyhow!("HDF5 row_attrs/Start: {e:?}"))?;
            rg.new_dataset_builder("End")
                .shape(&[n_genes as u64])
                .chunk(&[chunk])
                .deflate(2)
                .write::<i64>(n_ends.as_slice().unwrap())
                .map_err(|e| anyhow::anyhow!("HDF5 row_attrs/End: {e:?}"))?;
            rg.new_dataset_builder("Gene")
                .write_vlen_utf8_strings(&gene_slices)
                .map_err(|e| anyhow::anyhow!("HDF5 row_attrs/Gene: {e:?}"))?;
            rg.new_dataset_builder("Accession")
                .write_vlen_utf8_strings(&id_slices)
                .map_err(|e| anyhow::anyhow!("HDF5 row_attrs/Accession: {e:?}"))?;
            rg.new_dataset_builder("Chromosome")
                .write_vlen_utf8_strings(&chr_slices)
                .map_err(|e| anyhow::anyhow!("HDF5 row_attrs/Chromosome: {e:?}"))?;
            rg.new_dataset_builder("Strand")
                .write_vlen_utf8_strings(&strand_slices)
                .map_err(|e| anyhow::anyhow!("HDF5 row_attrs/Strand: {e:?}"))?;
        }
        // col_attrs group: CellID as string dataset
        {
            let cell_slices: Vec<&str> = cell_bcs_order.iter().map(|s| s.as_str()).collect();
            let mut cg = wf
                .create_group("col_attrs")
                .map_err(|e| anyhow::anyhow!("HDF5 create col_attrs: {e:?}"))?;
            cg.new_dataset_builder("CellID")
                .write_vlen_utf8_strings(&cell_slices)
                .map_err(|e| anyhow::anyhow!("HDF5 col_attrs/CellID: {e:?}"))?;
            // SampleID is written only when sample demultiplexing is active, so a
            // single-sample loom stays byte-for-byte identical to the prior output.
            if let Some(samples) = sample_ids {
                let sample_slices: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
                cg.new_dataset_builder("SampleID")
                    .write_vlen_utf8_strings(&sample_slices)
                    .map_err(|e| anyhow::anyhow!("HDF5 col_attrs/SampleID: {e:?}"))?;
            }
        }

        wf.flush()
            .map_err(|e| anyhow::anyhow!("HDF5 flush error: {e:?}"))?;
        debug!("Written loom file: {outfile}");
        Ok(())
    }

    /// UMI extraction dispatch
    #[cfg(feature = "bam")]
    fn extract_umi(&self, rec: &Record) -> anyhow::Result<String> {
        match &self.umi_extension {
            UmiExtension::No => self.no_extension(rec),
            UmiExtension::Chr => self.extension_chr(rec),
            UmiExtension::Gene => self.extension_gene(rec),
            UmiExtension::Nbp(n) => self.extension_nbp(rec, *n),
            UmiExtension::WithoutUmi => self.placeholder_umi(rec),
        }
    }
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// Python: chrom normalization in iter_alignments
pub fn normalize_chrom(chrom: &str) -> String {
    let s = if chrom.len() >= 3 && chrom[..3].eq_ignore_ascii_case("chr") {
        let rest = &chrom[3..];
        if rest == "M" {
            "MT".to_string()
        } else {
            rest.to_string()
        }
    } else {
        chrom.to_string()
    };
    if s.contains('_') {
        s.split('_').nth(1).unwrap_or(&s).to_string()
    } else {
        s
    }
}

/// Python: reverse() for strand character
pub fn reverse_strand(s: char) -> char {
    if s == '+' {
        '-'
    } else {
        '+'
    }
}

fn parse_gtf_fields(
    line: &str,
) -> (
    String,
    String,
    String,
    i64,
    i64,
    String,
    String,
    String,
    String,
) {
    let parts: Vec<&str> = line.splitn(9, '\t').collect();
    let get = |i: usize| parts.get(i).copied().unwrap_or("").to_string();
    (
        get(0),
        get(1),
        get(2),
        parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0),
        parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0),
        get(5),
        get(6),
        get(7),
        get(8),
    )
}

#[cfg(feature = "bam")]
fn aux_to_i64(aux: rust_htslib::bam::record::Aux) -> i64 {
    use rust_htslib::bam::record::Aux;
    match aux {
        Aux::U8(v) => v as i64,
        Aux::U16(v) => v as i64,
        Aux::U32(v) => v as i64,
        Aux::I8(v) => v as i64,
        Aux::I16(v) => v as i64,
        Aux::I32(v) => v as i64,
        _ => 1,
    }
}

#[cfg(feature = "bam")]
fn aux_to_string(aux: rust_htslib::bam::record::Aux) -> String {
    use rust_htslib::bam::record::Aux;
    match aux {
        Aux::String(s) => s.to_string(),
        Aux::Char(c) => (c as char).to_string(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::logic_from_name;

    fn make_counter() -> ExInCounter {
        ExInCounter::new(
            "test".to_string(),
            logic_from_name("Default"),
            None,
            "no",
            false,
            "0",
            "/tmp".to_string(),
            "uint16".to_string(),
        )
        .unwrap()
    }

    // ── default tag names ─────────────────────────────────────────────────────

    #[test]
    fn exincounter_default_cb_tag_is_cb() {
        assert_eq!(make_counter().cellbarcode_str, "CB");
    }

    #[test]
    fn exincounter_default_ub_tag_is_ub() {
        assert_eq!(make_counter().umibarcode_str, "UB");
    }

    // ── explicit tag override ─────────────────────────────────────────────────

    #[test]
    fn cb_tag_override_is_reflected() {
        let mut counter = make_counter();
        counter.cellbarcode_str = "CR".to_string();
        assert_eq!(counter.cellbarcode_str, "CR");
    }

    #[test]
    fn ub_tag_override_is_reflected() {
        let mut counter = make_counter();
        counter.umibarcode_str = "UR".to_string();
        assert_eq!(counter.umibarcode_str, "UR");
    }

    #[test]
    fn both_tag_overrides_are_independent() {
        let mut counter = make_counter();
        counter.cellbarcode_str = "GE".to_string();
        counter.umibarcode_str = "GM".to_string();
        assert_eq!(counter.cellbarcode_str, "GE");
        assert_eq!(counter.umibarcode_str, "GM");
    }
}
