//! Translated from velocyto/commands/run_smartseq2.py

use clap::Args;

use super::run::run_inner;

/// Arguments for the `run_smartseq2` subcommand.
///
/// Designed for plate-based Smart-seq2 data where each cell has its own
/// position-sorted BAM file. Fixed parameters: `onefilepercell = true`,
/// `without_umi = true`, `logic = "SmartSeq2"`, `multimap = false`,
/// `samtools_threads = 1`, `samtools_memory = 1` (no re-sorting needed since
/// each BAM is already per-cell).
#[derive(Args)]
pub struct RunSmartseq2Args {
    /// Enable verbose (debug) logging
    #[arg(short, long)]
    pub verbose: bool,

    /// One BAM file per cell (use shell glob expansion to pass multiple files)
    #[arg(required = true)]
    pub bamfiles: Vec<String>,
    /// Genome annotation GTF file
    pub gtffile: String,
    /// Output folder (created if absent; defaults to `<bam-dir>/velocyto`)
    #[arg(short = 'o', long)]
    pub outputfolder: Option<String>,
    /// Sample name used as the output loom filename stem
    #[arg(short = 'e', long)]
    pub sampleid: Option<String>,
    /// GTF file containing genomic intervals to mask (e.g. repeats)
    #[arg(short = 'm', long)]
    pub repmask: Option<String>,
    /// dtype for loom layer arrays (default: uint32)
    #[arg(short = 't', long, default_value = "uint32")]
    pub dtype: String,
    /// Debug dump: save a molecular mapping report every N cells (0 = disabled)
    #[arg(short = 'd', long, default_value = "0")]
    pub dump: String,
}

/// Runs the velocity analysis on Smart-seq2 data (one BAM file per cell).
///
/// Translated from `run_smartseq2.run_smartseq2` in Python.
///
/// Each input BAM is treated as a single cell (`onefilepercell = true`). UMIs
/// are not used (`without_umi = true`) because Smart-seq2 is a full-length
/// protocol without UMI barcodes. The `SmartSeq2` logic class is selected
/// automatically. Samtools sorting is skipped (thread and memory limits are
/// both set to 1) because each per-cell BAM is already fully sorted.
///
/// Delegates to [`run_inner`].
pub fn run_smartseq2(args: RunSmartseq2Args) -> anyhow::Result<()> {
    run_inner(
        &args.bamfiles,
        &args.gtffile,
        None,
        args.outputfolder.as_deref(),
        args.sampleid.as_deref(),
        args.repmask.as_deref(),
        true, // onefilepercell
        "SmartSeq2",
        true, // without_umi
        "no",
        false, // multimap
        1,     // samtools_threads
        1,     // samtools_memory
        &args.dump,
        &args.dtype,
        &[],
    )
}
