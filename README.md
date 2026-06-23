# velocyto-rs

Faithful Rust translation of [velocyto.py](https://github.com/velocyto-team/velocyto.py) v0.17.16 — RNA velocity analysis for single-cell RNA-seq data.

**Reference snapshot**: velocyto.py v0.17.16 (no git hash available — copied source snapshot).  
**Translation approach**: bottom-up, one Rust function per Python function, source files mirror Python module names.  

## Why
BD Rhapsody single-cell RNA-seq data uses a non-standard, complex cell barcode scheme, which makes it difficult to match cell barcodes between STARsolo output and the BD Rhapsody™ Sequence Analysis Pipeline output. [velocyto.py](https://github.com/velocyto-team/velocyto.py) remains the most straightforward approach for generating spliced and unspliced count matrices required for RNA velocity analysis; however, the project has been unmaintained since June 2019. This project provides a modern, actively maintained, and high-performance Rust reimplementation capable of processing BD Rhapsody BAM files to produce the necessary count matrices.

> 🙏 **Heartfelt thanks to Prof. Johan Henriksson ([@mahogny](https://github.com/mahogny))** for the original software translation work that made this project possible. Your contribution and inspiration are deeply appreciated!

## New Feature

* output `anndata` or `loom`
* add `--cb-tag`, `--ub-tag`, `--sample-tag` to to suport BAM not from 10x (e.g. BD Rhaposody)



## Performance

Measured with `/usr/bin/time -v` on the same machine, same input files, same barcodes file. Both tools produce identical count output (see Output correctness below). Python: velocyto.py v0.17.16, CPython 3.9. Rust: `cargo build --release --features bam`.

> **Backend note:** the Rust numbers in the table below were measured with the original `rust-htslib` (C `htslib`) BAM backend. BAM/SAM reading has since moved to the pure-Rust [`noodles`](https://github.com/zaeleus/noodles) crates — see [BAM reader: noodles vs htslib](#bam-reader-noodles-vs-htslib), which re-baselines both backends on one machine. The Rust-vs-Python rows here are from a separate run and are indicative, not directly comparable to that same-machine table.

| Dataset | Tool | Wall time | Peak RSS | Speed ratio | RAM ratio |
|---|---|---|---|---|---|
| `mini_chr21.bam`<br>(39 MB BAM, 16 MB GTF) | Python | 22.1 s | 390 MB | — | — |
| | Rust | **2.2 s** | **36 MB** | **10× faster** | **11× less** |
| `s04.bam`<br>(1.7 GB BAM, 1.5 GB GTF) | Python | 14 min 47 s | 3.80 GB | — | — |
| | Rust | **2 min 17 s** | **3.26 GB** | **6.5× faster** | **1.2× less** |

Hardware: Linux x86-64, single machine.

The smaller RAM advantage on the large dataset is expected: the 1.5 GB GTF dominates RSS in both tools once read into memory. The speed advantage is consistent because Rust avoids Python's per-object allocation overhead and GIL contention during GTF parsing and BAM scanning.

### BAM reader: noodles vs htslib

BAM/SAM reading was migrated from `rust-htslib` (which links the C `htslib` library and needs `libclang` to build) to the pure-Rust [`noodles`](https://github.com/zaeleus/noodles) crates, removing the last C build dependency for `--features bam`. Re-baselined on one machine, same inputs, `samtools sort` skipped equally, both built `--release --features bam`; output is bit-for-bit identical.

| Dataset | Metric | `rust-htslib` | `noodles` | Improvement |
|---|---|---|---|---|
| `mini_chr21`<br>(39 MB BAM, 16 MB GTF) | Wall time | 1.26 s | **0.76 s** | **1.7× faster** |
| | Peak RSS | 24.6 MB | **18.5 MB** | **1.3× less** |
| `s04`<br>(1.7 GB BAM, 1.5 GB GTF) | Wall time | 2 m 16 s | **1 m 54 s** | **1.2× faster** |
| | Peak RSS | 3.75 GB | **3.68 GB** | ~2% less |
| Release binary | Size | 93 MB | **59 MB** | **1.6× smaller** |

The small-input win is larger because noodles decodes records lazily (only the fields you touch) with no C FFI boundary; on the large input the unchanged 1.5 GB GTF load dominates wall time and RSS, diluting the reader speedup. The smaller binary reflects no statically-linked C `htslib`.

### Output correctness

Comparing Rust vs Python output on the same input (S08, 1974 cells × 39579 genes):

| Dataset dimension | Agreement |
|---|---|
| Gene set | Identical |
| Cell barcode set | Identical |
| Gene row order | Identical |
| Layer counts (spliced/unspliced/ambiguous) | **100% identical** — 0 of 78 M (gene, cell) pairs differ |
| Layer dtype | Rust uses `uint32` same as Python |

**Comparison note:** 69 gene names appear twice in the dataset (same name, different accession/gene ID). A naive gene-name-keyed comparison will silently cross-map these and produce false diffs. Always align by `row_attrs/Accession` (unique gene ID), not by `row_attrs/Gene` (gene name).

### Where the speedup comes from

The dominant cost for large reference genomes is loading the GTF annotation. Three optimisations in `read_transcriptmodels`:

1. **Schwartzian transform for GTF sorting** — Python's `list.sort(key=f)` computes the sort key once per element (built-in). The original Rust `sort_by(|a,b| f(a).cmp(f(b)))` recomputed the key — including a full line clone — on every comparison, O(n log n) times. Replaced with explicit key precomputation (O(n) extractions) then index sort.

2. **Filter to exon-only lines during collection** — The annotation builder only uses `feature_type == "exon"` lines. Gencode v42 is ~⅓ exon lines, so filtering at read time reduces both allocation count and sort size ~3×.

3. **1 MB read buffer** — reduces syscall overhead vs the default 8 KB `BufReader` for large files.

## Build

```bash
cargo build --release
```

BAM support is **on by default** (pure-Rust [`noodles`](https://github.com/zaeleus/noodles) backend — no `libclang`, `htslib`, or any C library to build). Pass `--no-default-features` only if you want the dependency-light stub build without BAM support.

> **No samtools required at runtime.** The cell-barcode sort uses the `samtools` binary when it is on `PATH`, but otherwise falls back to a built-in pure-Rust disk-spilling external merge sort (`src/bam_sort.rs`). It is output-equivalent to `samtools sort -t CB` (verified against a 28.4 M-read fixture: identical per-barcode read counts). So velocyto-rs has no hard dependency on samtools or any C library.

## Usage

BAM support is on by default — no extra feature flag is needed. Run with `--help` for the full option list.

### `run10x` — 10X Chromium (CellRanger output)

The most common entry point. Pass the CellRanger sample folder and a GTF file; the BAM and barcodes list are found automatically.

```bash
velocyto-rs run10x /data/cellranger/sample1 /ref/gencode.v42.gtf
```

With a repeat-mask GTF and verbose logging:

```bash
velocyto-rs run10x /data/cellranger/sample1 /ref/gencode.v42.gtf \
    --mask /ref/repeats.gtf \
    --verbose
```

Output is written to `<samplefolder>/velocyto/<samplename>.h5ad` by default (AnnData).
Pass `--output-format loom` for the legacy loom file, or `--output-format both` for both.

### `run` — generic BAM

Use this for any platform not covered by a dedicated subcommand (BD Rhapsody, Parse Biosciences, etc.).

```bash
velocyto-rs run sample.bam /ref/gencode.v42.gtf \
    --bcfile barcodes.txt \
    --outputfolder ./results \
    --sampleid my_sample
```

Multiple BAMs merged into one loom:

```bash
velocyto-rs run lane1.bam lane2.bam lane3.bam /ref/gencode.v42.gtf \
    --bcfile barcodes.txt \
    --sampleid pooled_run
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `-b / --bcfile` | — | Barcode whitelist (plain text or `.gz`, one per line) |
| `-o / --outputfolder` | `<bam-dir>/velocyto` | Output directory |
| `-e / --sampleid` | derived from BAM name | Loom filename stem |
| `-m / --mask` | — | GTF of genomic intervals to mask (e.g. repeats) |
| `-l / --logic` | `Default` | Molecule-filtering logic class |
| `-U / --without-umi` | false | Read-count mode (no UMI deduplication) |
| `-u / --umi-extension` | `no` | Extend UMI identity: `no`, `chr`, `Gene`, `Cluster`, `all` |
| `-M / --multimap` | false | Count non-uniquely mapped reads (not recommended) |
| `-t / --dtype` | `uint32` | Layer array dtype; use `uint16` for low-depth data |
| `--output-format` | `h5ad` | Output file format: `h5ad` (AnnData), `loom`, or `both` (see [Output formats](#output-formats)) |
| `--samtools-threads` | 16 | Sort threads — used by `samtools sort` if installed, else by the built-in pure-Rust sort |
| `--samtools-memory` | 2048 | MB per sort thread / per in-memory sort chunk before spilling to disk |
| `--cb-tag` | auto | BAM tag for cell barcode (e.g. `CB`, `XC`); skips auto-detection when both tags are set |
| `--ub-tag` | auto | BAM tag for UMI barcode (e.g. `UB`, `XM`); skips auto-detection when both tags are set |
| `--sample-tag` | — | BAM tag carrying sample identity (e.g. BD Rhapsody `ST`); demultiplexes a multi-sample BAM in place (see [Sample demultiplexing](#sample-demultiplexing)) |

## Output formats

`--output-format` is available on all four run commands (`run10x`, `run`, `run_dropest`, `run_smartseq2`) and accepts:

| Value | Writes | Notes |
|---|---|---|
| `h5ad` *(default)* | `<sampleid>.h5ad` | AnnData, the modern scanpy/scVelo on-disk format |
| `loom` | `<sampleid>.loom` | Legacy loompy format (what upstream velocyto.py emits) |
| `both` | `<sampleid>.h5ad` and `<sampleid>.loom` | Both files from a single counting pass |

> **Note:** AnnData (h5ad) output is an extension beyond the v0.17.16 port — upstream velocyto only ever emitted loom. The two files carry identical counts; they differ only in on-disk layout.

Both formats are written directly via the bundled pure-Rust HDF5 implementation (`hdf5-pure-rs`) — no C/HDF5 library and no Python dependency at runtime.

### h5ad layout

The h5ad file targets the current anndata on-disk spec and round-trips through `anndata.read_h5ad`:

- **Cells × genes** (`obs` × `var`), the transpose of loom's genes × cells.
- **Sparse CSR** for `X` and every `layers/{spliced,unspliced,ambiguous}` — a single-cell matrix is mostly zeros, so CSR is far smaller than dense. `X` is the float32 sum of all layers; layer width follows `--dtype` (`uint32` lossless / `uint16` saturating).
- **`var` indexed by `Accession`** (unique Ensembl ID), with `Gene` as a column. Gene *names* collide (see the alignment note above), so the accession must be the index.
- **`obs` indexed by `CellID`**; `SampleID` appears as an `obs` column only when `--sample-tag` demultiplexing is active.

### Loom layout

Matches loompy's layout exactly (genes × cells): a float32 `matrix` (sum of all layers), `layers/{spliced,unspliced,ambiguous}` (`uint32` by default), and `row_attrs` / `col_attrs` string datasets including `Gene`, `Accession`, `Chromosome`, `Strand`, and `CellID`. This is byte-for-byte comparable to upstream velocyto.py output via `h5diff`.

## Sample demultiplexing

`--sample-tag` (generic `run` command only) lets a single multi-sample BAM be counted without `samtools split`. This is an extension beyond the v0.17.16 port — upstream velocyto expects one sample per BAM.

```bash
velocyto-rs run multiplexed.bam /ref/gencode.v42.gtf \
    --bcfile barcodes.txt \
    --sample-tag ST \
    --sampleid pooled_run
```

- Cell identity becomes `(sample, barcode)`, so the same bead barcode in two samples no longer collides. CellIDs are formatted `{sampleid}_{sample}:{bc}` and a `SampleID` column (loom `col_attrs/SampleID`, h5ad `obs` column) is added.
- The BAM is still sorted by `CB` only — no second sort, no split.
- Reads missing the sample tag are dropped (not pooled into a phantom sample). Reads that carry the tag with any value (e.g. BD `"Multiplet"`) are kept as their own sample — filter them downstream via `SampleID`.
- With `--sample-tag` off, output is byte-for-byte identical to before.


## Citation

If you use this software, cite the original velocyto paper:

> La Manno G, Soldatov R, Zeisel A, Braun E, Hochgerner H, Petukhov V, Lidschreiber K, Kastriti ME, Lönnerberg P, Furlan A, Fan J, Borm LE, Liu Z, van Bruggen D, Guo J, He X, Barker R, Sundström E, Castelo-Branco G, Cramer P, Adameyko I, Linnarsson S, Kharchenko PV.  
> **RNA velocity of single cells.**  
> *Nature* 560, 494–498 (2018). https://doi.org/10.1038/s41586-018-0414-6


## License

BSD 2-Clause License. See [LICENSE](LICENSE). Derived from [velocyto.py](https://github.com/velocyto-team/velocyto.py) by Peter Kharchenko, Sten Linnarsson and Gioele La Manno.
