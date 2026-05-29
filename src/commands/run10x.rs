//! Translated from velocyto/commands/run10x.py

use anyhow::bail;
use clap::Args;
use std::path::Path;

use super::run::run_inner;

/// Arguments for the `run10x` subcommand.
///
/// `samplefolder` is the cellranger sample folder (must contain `outs/`).
/// `gtffile` is the genome annotation file used to build transcript models.
/// All other options mirror the Python CLI options of the same name.
#[derive(Args)]
pub struct Run10xArgs {
    /// Enable verbose (debug) logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Path to the cellranger sample folder
    pub samplefolder: String,
    /// Genome annotation GTF file
    pub gtffile: String,
    /// Table containing metadata of the various samples (CSV; rows = samples, cols = entries)
    #[arg(short = 's', long)]
    pub metadatatable: Option<String>,
    /// GTF file containing intervals to mask (e.g. repeats)
    #[arg(short = 'm', long)]
    pub mask: Option<String>,
    /// The logic to use for molecule filtering (default: Default)
    #[arg(short = 'l', long, default_value = "Default")]
    pub logic: String,
    /// Consider non-unique mappings (not recommended)
    #[arg(short = 'M', long, default_value_t = false)]
    pub multimap: bool,
    /// Number of threads for samtools sort
    #[arg(long, default_value_t = 16)]
    pub samtools_threads: usize,
    /// MB of memory per thread for samtools sort
    #[arg(long, default_value_t = 2048)]
    pub samtools_memory: usize,
    /// dtype for loom layer arrays: "uint32" (default, lossless) or "uint16" (smaller, saturates at 65535)
    #[arg(short = 't', long, default_value = "uint32")]
    pub dtype: String,
    /// Debug dump: save a molecular mapping report every N cells (0 = disabled)
    #[arg(short = 'd', long, default_value = "0")]
    pub dump: String,
    /// BAM tag for cell barcode (overrides auto-detection; e.g. `CB` or `XC`)
    #[arg(long)]
    pub cb_tag: Option<String>,
    /// BAM tag for UMI barcode (overrides auto-detection; e.g. `UB` or `XM`)
    #[arg(long)]
    pub ub_tag: Option<String>,
}

/// Runs the velocity analysis for a 10X Chromium sample.
///
/// Locates the position-sorted BAM (`outs/possorted_genome_bam.bam`) and the
/// filtered barcodes file automatically from the CellRanger output folder, then
/// delegates to [`run_inner`].
///
/// Checks the CellRanger `_log` file for a successful completion marker and
/// aborts if the expected output loom file already exists.
///
/// Note: Python also loads tSNE projection (`_X`, `_Y`) and graph-based cluster
/// labels (`Clusters`) into `additional_ca`. Those float column attributes are
/// currently omitted from this port (hdf5-pure-rs only exposes a string API for
/// attrs).
pub fn run10x(args: Run10xArgs) -> anyhow::Result<()> {
    let sf = Path::new(&args.samplefolder);

    // Check cellranger completion log
    let log_path = sf.join("_log");
    if !log_path.exists() {
        log::error!("Older cellranger version: cannot verify outputs. Proceeding anyway.");
    } else if !std::fs::read_to_string(&log_path)
        .unwrap_or_default()
        .contains("Pipestance completed successfully!")
    {
        log::error!("Cellranger outputs may not be ready.");
    }

    let bamfile = sf
        .join("outs")
        .join("possorted_genome_bam.bam")
        .to_string_lossy()
        .to_string();
    let bcfile = find_barcodes_file(sf)?;
    let outputfolder = sf.join("velocyto").to_string_lossy().to_string();
    let sampleid = sf
        .file_name()
        .map(|n| {
            n.to_string_lossy()
                .trim_end_matches('/')
                .trim_end_matches('\\')
                .to_string()
        })
        .unwrap_or_else(|| "sample".to_string());

    let loom_out = Path::new(&outputfolder).join(format!("{sampleid}.loom"));
    if loom_out.exists() {
        bail!("Output already exists: {}. Aborted!", loom_out.display());
    }

    // Note: Python also loads tsne/clusters files into additional_ca (_X, _Y, Clusters).
    // hdf5-pure-rust does not support float array attrs; those are omitted here.

    run_inner(
        &[bamfile],
        &args.gtffile,
        Some(&bcfile),
        Some(&outputfolder),
        Some(&sampleid),
        args.mask.as_deref(),
        false,
        &args.logic,
        false,
        "no",
        args.multimap,
        args.samtools_threads,
        args.samtools_memory,
        &args.dump,
        &args.dtype,
        &[],
        args.cb_tag.as_deref(),
        args.ub_tag.as_deref(),
    )
}

/// Locate the filtered barcodes file inside a CellRanger output folder.
///
/// Tries two layouts in order:
/// - **CellRanger v2**: `outs/filtered_gene_bc_matrices/<genome>/barcodes.tsv`
/// - **CellRanger v3**: `outs/filtered_feature_bc_matrix/barcodes.tsv.gz`
///
/// Returns an error if neither path can be found.
fn find_barcodes_file(sf: &Path) -> anyhow::Result<String> {
    // CellRanger v2: outs/filtered_gene_bc_matrices/<genome>/barcodes.tsv
    let v2_root = sf.join("outs").join("filtered_gene_bc_matrices");
    if v2_root.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&v2_root) {
            for entry in entries.flatten() {
                let candidate = entry.path().join("barcodes.tsv");
                if candidate.exists() {
                    return Ok(candidate.to_string_lossy().to_string());
                }
            }
        }
    }
    // CellRanger v3: outs/filtered_feature_bc_matrix/barcodes.tsv.gz
    let v3 = sf
        .join("outs")
        .join("filtered_feature_bc_matrix")
        .join("barcodes.tsv.gz");
    if v3.exists() {
        return Ok(v3.to_string_lossy().to_string());
    }
    bail!("Cannot locate barcodes.tsv file in {}", sf.display())
}
