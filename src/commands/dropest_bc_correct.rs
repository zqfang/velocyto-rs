//! Translated from velocyto/commands/dropest_bc_correct.py
//!
//! The Python implementation uses `rpy2` to load DropEst's `.rds` file
//! directly and reads `rds$merge_targets` (a named character vector mapping
//! original → corrected barcodes). Because Rust cannot call R, this port
//! reads an equivalent two-column TSV that must be exported from R first:
//!
//! ```r
//! rds <- readRDS("file.rds")
//! write.table(
//!     data.frame(names(rds$merge_targets), rds$merge_targets),
//!     sep = "\t", col.names = FALSE, row.names = FALSE,
//!     file = "mapping.tsv"
//! )
//! ```

use anyhow::Context;
use clap::Args;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Arguments for the `dropest_bc_correct` subcommand.
///
/// Takes a DropEst BAM file and a barcode-mapping TSV, corrects the `CB` tag
/// on every read, and writes a new BAM alongside a `barcodes_<prefix>.tsv`
/// whitelist file.
#[derive(Args)]
pub struct DropestBcCorrectArgs {
    /// Enable verbose (debug) logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Input BAM file produced by DropEst
    pub bamfilepath: String,
    /// Two-column TSV: `original_bc<TAB>corrected_bc`.
    /// Export from DropEst RDS with:
    /// `write.table(data.frame(names(rds$merge_targets), rds$merge_targets), sep="\t", ...)`
    pub mapping_file: String,
    /// Output path for the corrected BAM. Defaults to `correct_<input>.bam`
    /// in the same directory as the input.
    #[arg(short = 'o', long)]
    pub corrected_output: Option<String>,
}

/// Corrects DropEst cell barcodes in a BAM file and writes a valid-barcodes list.
///
/// Translated from `dropest_bc_correct.dropest_bc_correct` in Python. The
/// Python version loaded barcode mappings directly from a DropEst `.rds` file
/// via `rpy2`; this Rust port reads a pre-exported two-column TSV instead (see
/// module-level documentation for the R export command).
///
/// Steps performed:
/// 1. Load `original_bc → corrected_bc` mappings from the TSV.
/// 2. Collect all unique corrected barcodes and write them to
///    `barcodes_<prefix>.tsv` next to the input BAM (where `<prefix>` is
///    everything before the first `_` in the BAM filename).
/// 3. Stream through the BAM; for every read whose `CB` tag matches an entry
///    in the mapping, overwrite the tag with the corrected value, then write
///    the record to the output BAM.
pub fn dropest_bc_correct(args: DropestBcCorrectArgs) -> anyhow::Result<()> {
    let mapping = load_mapping_tsv(&args.mapping_file)?;

    let bam_path = Path::new(&args.bamfilepath);
    let parent = bam_path.parent().unwrap_or(Path::new("."));
    let basename = bam_path.file_name().unwrap_or_default().to_string_lossy();
    let first_part = basename.splitn(2, '_').next().unwrap_or(&basename);

    // Python: writes unique corrected barcodes to barcodes_{prefix}.tsv
    let bc_out = parent
        .join(format!("barcodes_{first_part}.tsv"))
        .to_string_lossy()
        .to_string();
    log::info!("Writing barcodes list to {bc_out}");
    let unique_bcs: std::collections::HashSet<&String> = mapping.values().collect();
    let mut bc_file =
        BufWriter::new(fs::File::create(&bc_out).with_context(|| format!("Creating {bc_out}"))?);
    for bc in &unique_bcs {
        writeln!(bc_file, "{}", bc)?;
    }

    correct_bam_barcodes(bam_path, &mapping, args.corrected_output.as_deref())
}

/// Public re-export of [`load_mapping_tsv`] for use in integration tests.
pub fn load_mapping_tsv_pub(path: &str) -> anyhow::Result<HashMap<String, String>> {
    load_mapping_tsv(path)
}

/// Parse a two-column tab-separated file into an `original → corrected` map.
///
/// Blank lines and lines that lack both columns are silently skipped.
/// Leading/trailing whitespace is stripped from each field.
fn load_mapping_tsv(path: &str) -> anyhow::Result<HashMap<String, String>> {
    let f = fs::File::open(path).with_context(|| format!("Opening {path}"))?;
    let mut map = HashMap::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let mut cols = line.splitn(2, '\t');
        let orig = cols.next().unwrap_or("").trim().to_string();
        let corr = cols.next().unwrap_or("").trim().to_string();
        if !orig.is_empty() && !corr.is_empty() {
            map.insert(orig, corr);
        }
    }
    Ok(map)
}

/// Stream through a BAM file, replacing the `CB` tag on mapped reads, and
/// write the corrected records to a new BAM.
///
/// The output path defaults to `correct_<basename>` in the same directory as
/// the input when `corrected_output` is `None`.
#[cfg(feature = "bam")]
fn correct_bam_barcodes(
    bam_path: &Path,
    mapping: &HashMap<String, String>,
    corrected_output: Option<&str>,
) -> anyhow::Result<()> {
    use noodles_bam as bam;
    use noodles_sam::alignment::io::Write as _;
    use noodles_sam::alignment::record::data::field::Tag;
    use noodles_sam::alignment::record_buf::data::field::Value;
    use noodles_sam::alignment::RecordBuf;
    use std::fs::File;

    let parent = bam_path.parent().unwrap_or(Path::new("."));
    let basename = bam_path.file_name().unwrap_or_default().to_string_lossy();
    let out_path = match corrected_output {
        Some(p) => p.to_string(),
        None => parent
            .join(format!("correct_{basename}"))
            .to_string_lossy()
            .to_string(),
    };

    let mut infile = File::open(bam_path)
        .map(bam::io::Reader::new)
        .with_context(|| format!("Opening BAM {}", bam_path.display()))?;
    let header = infile
        .read_header()
        .with_context(|| format!("Reading BAM header {}", bam_path.display()))?;
    let mut outfile = File::create(&out_path)
        .map(bam::io::Writer::new)
        .with_context(|| format!("Creating BAM {out_path}"))?;
    outfile
        .write_header(&header)
        .with_context(|| "Writing BAM header")?;

    // The lazy bam::Record is a near-zero-copy view; read into an owned
    // RecordBuf so the CB tag can be rewritten before re-encoding.
    let cb_tag = Tag::from([b'C', b'B']);
    let mut record = RecordBuf::default();
    while infile
        .read_record_buf(&header, &mut record)
        .with_context(|| "Reading BAM record")?
        != 0
    {
        // Resolve the correction first so the immutable borrow of `record` ends
        // before `data_mut()` takes a mutable borrow.
        let corrected: Option<String> = match record.data().get(&cb_tag) {
            Some(Value::String(cb)) => mapping.get(cb.to_string().as_str()).cloned(),
            _ => None,
        };
        if let Some(corrected) = corrected {
            record
                .data_mut()
                .insert(cb_tag, Value::String(corrected.into()));
        }
        outfile
            .write_alignment_record(&header, &record)
            .with_context(|| "Writing BAM record")?;
    }
    log::info!("Done. Corrected BAM at {out_path}");
    Ok(())
}

/// Stub that errors when the crate is built without the `bam` feature.
#[cfg(not(feature = "bam"))]
fn correct_bam_barcodes(
    _bam_path: &Path,
    _mapping: &HashMap<String, String>,
    _corrected_output: Option<&str>,
) -> anyhow::Result<()> {
    anyhow::bail!("BAM support requires --features bam at compile time")
}
