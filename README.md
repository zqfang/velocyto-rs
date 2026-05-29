# velocyto-rs

Faithful Rust translation of [velocyto.py](https://github.com/velocyto-team/velocyto.py) v0.17.16 — RNA velocity analysis for single-cell RNA-seq data.

**Reference snapshot**: velocyto.py v0.17.16 (no git hash available — copied source snapshot).  
**Translation approach**: bottom-up, one Rust function per Python function, source files mirror Python module names.  

## Why
BD Rhapsody single-cell RNA-seq data uses a non-standard, complex cell barcode scheme, which makes it difficult to match cell barcodes between STARsolo output and the BD Rhapsody™ Sequence Analysis Pipeline output. [velocyto.py](https://github.com/velocyto-team/velocyto.py) remains the most straightforward approach for generating spliced and unspliced count matrices required for RNA velocity analysis; however, the project has been unmaintained since June 2019. This project provides a modern, actively maintained, and high-performance Rust reimplementation capable of processing BD Rhapsody BAM files to produce the necessary count matrices.

> 🙏 **Heartfelt thanks to Prof. Johan Henriksson ([@mahogny](https://github.com/mahogny))** for the original software translation work that made this project possible. Your contribution and inspiration are deeply appreciated!


## Citation

If you use this software, cite the original velocyto paper:

> La Manno G, Soldatov R, Zeisel A, Braun E, Hochgerner H, Petukhov V, Lidschreiber K, Kastriti ME, Lönnerberg P, Furlan A, Fan J, Borm LE, Liu Z, van Bruggen D, Guo J, He X, Barker R, Sundström E, Castelo-Branco G, Cramer P, Adameyko I, Linnarsson S, Kharchenko PV.  
> **RNA velocity of single cells.**  
> *Nature* 560, 494–498 (2018). https://doi.org/10.1038/s41586-018-0414-6


## Performance

Measured with `/usr/bin/time -v` on the same machine, same input files, same barcodes file. Both tools produce identical count output (see Output correctness below). Python: velocyto.py v0.17.16, CPython 3.9. Rust: `cargo build --release --features bam`.

| Dataset | Tool | Wall time | Peak RSS | Speed ratio | RAM ratio |
|---|---|---|---|---|---|
| `mini_chr21.bam`<br>(39 MB BAM, 16 MB GTF) | Python | 22.1 s | 390 MB | — | — |
| | Rust | **2.2 s** | **36 MB** | **10× faster** | **11× less** |
| `s04.bam`<br>(1.7 GB BAM, 1.5 GB GTF) | Python | 14 min 47 s | 3.80 GB | — | — |
| | Rust | **2 min 17 s** | **3.26 GB** | **6.5× faster** | **1.2× less** |

Hardware: Linux x86-64, single machine.

The smaller RAM advantage on the large dataset is expected: the 1.5 GB GTF dominates RSS in both tools once read into memory. The speed advantage is consistent because Rust avoids Python's per-object allocation overhead and GIL contention during GTF parsing and BAM scanning.

### Output correctness

Comparing Rust vs Python output on the same input (S08, 1974 cells × 39579 genes):

| Dataset dimension | Agreement |
|---|---|
| Gene set | Identical |
| Cell barcode set | Identical |
| Gene row order | Identical (after fix to `assign_indexes_to_genes`) |
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
cargo build --release --features bam
```

## Usage

All subcommands require the `bam` feature (built in by default via `--features bam`). Run with `--help` for the full option list.

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

Output is written to `<samplefolder>/velocyto/<samplename>.loom`.

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
| `--samtools-threads` | 16 | Threads for `samtools sort` |
| `--samtools-memory` | 2048 | MB per thread for `samtools sort` |
| `--cb-tag` | auto | BAM tag for cell barcode (e.g. `CB`, `XC`); skips auto-detection when both tags are set |
| `--ub-tag` | auto | BAM tag for UMI barcode (e.g. `UB`, `XM`); skips auto-detection when both tags are set |







## License

BSD 2-Clause License. See [LICENSE](LICENSE). Derived from [velocyto.py](https://github.com/velocyto-team/velocyto.py) by Peter Kharchenko, Sten Linnarsson and Gioele La Manno.
