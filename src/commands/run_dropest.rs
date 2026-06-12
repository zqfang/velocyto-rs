//! Translated from velocyto/commands/run_dropest.py

use clap::Args;
use std::path::Path;

use super::run::run_inner;

/// Arguments for the `run_dropest` subcommand.
///
/// Intended for BAM files produced by the DropEst pipeline, ideally after
/// barcode correction with `dropest_bc_correct`. Fixed parameters relative to
/// `run`: `onefilepercell = false`, `without_umi = false`,
/// `umi_extension = "chr"`, `multimap = false`.
#[derive(Args)]
pub struct RunDropestArgs {
    /// Enable verbose (debug) logging
    #[arg(short, long)]
    pub verbose: bool,

    /// BAM file produced by DropEst (ideally barcode-corrected with `dropest_bc_correct`)
    pub bamfile: String,
    /// Genome annotation GTF file
    pub gtffile: String,
    /// Valid barcodes file (TSV). When omitted, looked up automatically as
    /// `barcodes_<prefix>.tsv` next to the BAM file
    #[arg(short = 'b', long)]
    pub bcfile: Option<String>,
    /// Molecule-filtering logic class name (default: Default)
    #[arg(short = 'l', long, default_value = "Default")]
    pub logic: String,
    /// Output folder (created if absent)
    #[arg(short = 'o', long)]
    pub outputfolder: Option<String>,
    /// Sample name used as the output loom filename stem
    #[arg(short = 'e', long)]
    pub sampleid: Option<String>,
    /// GTF file containing genomic intervals to mask (e.g. repeats)
    #[arg(short = 'm', long)]
    pub repmask: Option<String>,
    /// Number of threads for samtools sort
    #[arg(long, default_value_t = 16)]
    pub samtools_threads: usize,
    /// MB of memory per thread for samtools sort
    #[arg(long, default_value_t = 2048)]
    pub samtools_memory: usize,
    /// dtype for loom layer arrays: "uint32" (default, lossless) or "uint16" (smaller, saturates at 65535)
    #[arg(short = 't', long, default_value = "uint32")]
    pub dtype: String,
    /// Output file format: "h5ad" (default, AnnData), "loom", or "both"
    #[arg(long, default_value = "h5ad")]
    pub output_format: String,
    /// Debug dump: save a molecular mapping report every N cells (0 = disabled)
    #[arg(short = 'd', long, default_value = "0")]
    pub dump: String,
}

/// Runs the velocity analysis on DropEst-preprocessed data.
///
/// Translated from `run_dropest.run_dropest` in Python.
///
/// If `--bcfile` is not supplied the barcode list is located automatically as
/// `barcodes_<prefix>.tsv` in the same directory as the BAM, where `<prefix>`
/// is everything before the first `_` in the BAM filename. This convention
/// matches the file written by `dropest_bc_correct`. An error is logged and
/// the function returns early if the file cannot be found.
///
/// A warning is emitted when the BAM filename does not contain the string
/// `"correct"`, which would indicate the file has not gone through barcode
/// correction.
///
/// Delegates to [`run_inner`] with `umi_extension = "chr"` (DropEst default).
pub fn run_dropest(args: RunDropestArgs) -> anyhow::Result<()> {
    // Python: auto-find bcfile as barcodes_{prefix}.tsv next to the BAM
    let bcfile: String = match args.bcfile {
        Some(f) => f,
        None => {
            let bam_path = Path::new(&args.bamfile);
            let parent = bam_path.parent().unwrap_or(Path::new("."));
            let basename = bam_path.file_name().unwrap_or_default().to_string_lossy();
            let first_part = basename.splitn(2, '_').next().unwrap_or(&basename);
            let candidate = parent
                .join(format!("barcodes_{first_part}.tsv"))
                .to_string_lossy()
                .to_string();
            log::info!("Attempting to find barcode list at {candidate}");
            if Path::new(&candidate).exists() {
                log::info!("{candidate} found");
                candidate
            } else {
                log::error!("In run_dropest --bcfile/-b is required. Use `run` for custom usage.");
                return Ok(());
            }
        }
    };

    if !args.bamfile.contains("correct") {
        log::warn!(
            "Input BAM does not contain 'correct'; may not be output of dropest_bc_correct."
        );
    }

    run_inner(
        &[args.bamfile],
        &args.gtffile,
        Some(&bcfile),
        args.outputfolder.as_deref(),
        args.sampleid.as_deref(),
        args.repmask.as_deref(),
        false,
        &args.logic,
        false,
        "chr",
        false,
        args.samtools_threads,
        args.samtools_memory,
        &args.dump,
        &args.dtype,
        &[],
        None,
        None,
        None,
        &args.output_format,
    )
}
