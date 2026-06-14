//! Throwaway bit-for-bit loom comparator for the noodles migration.
//! Aligns rows by `row_attrs/Accession` and columns by bare barcode
//! (`CellID.split(':')[1]`), per the alignment pitfall in CLAUDE.md, then
//! diffs the spliced/unspliced/ambiguous layers.
//!
//! Usage: cargo run --example loom_compare -- <ground_truth.loom> <test.loom>

use hdf5_pure_rust::File;
use std::collections::HashMap;

fn read_layer(f: &File, name: &str) -> (Vec<u64>, Vec<u32>) {
    let ds = f.dataset(name).unwrap_or_else(|_| panic!("missing {name}"));
    let shape = ds.shape().unwrap();
    let data = ds.read::<u32>().unwrap_or_else(|_| panic!("read {name}"));
    (shape, data)
}

fn bare_bc(cell_id: &str) -> String {
    // CellID is `{sampleid}:{bc}{gem_grp}`; normalize away the gem-group marker
    // ('x' or '-<GEM>') so a no-suffix loom aligns with an 'x'-suffixed one.
    let after = cell_id.split(':').nth(1).unwrap_or(cell_id);
    let after = after.split('-').next().unwrap_or(after);
    after.trim_end_matches('x').to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (gt_path, rs_path) = (&args[1], &args[2]);

    let gt = File::open(gt_path).unwrap();
    let rs = File::open(rs_path).unwrap();

    let gt_acc = gt.dataset("row_attrs/Accession").unwrap().read_strings().unwrap();
    let rs_acc = rs.dataset("row_attrs/Accession").unwrap().read_strings().unwrap();
    let gt_cells = gt.dataset("col_attrs/CellID").unwrap().read_strings().unwrap();
    let rs_cells = rs.dataset("col_attrs/CellID").unwrap().read_strings().unwrap();

    println!(
        "ground_truth: {} genes x {} cells | test: {} genes x {} cells",
        gt_acc.len(),
        gt_cells.len(),
        rs_acc.len(),
        rs_cells.len()
    );

    // Row alignment: gt row i -> rs row index, keyed by Accession.
    let rs_acc_idx: HashMap<&str, usize> =
        rs_acc.iter().enumerate().map(|(i, a)| (a.as_str(), i)).collect();
    let gt_row_to_rs: Vec<usize> = gt_acc
        .iter()
        .map(|a| *rs_acc_idx.get(a.as_str()).unwrap_or_else(|| panic!("accession {a} not in test")))
        .collect();

    // Col alignment: gt col j -> rs col index, keyed by bare barcode.
    let rs_bc_idx: HashMap<String, usize> =
        rs_cells.iter().enumerate().map(|(i, c)| (bare_bc(c), i)).collect();
    let gt_col_to_rs: Vec<usize> = gt_cells
        .iter()
        .map(|c| {
            let b = bare_bc(c);
            *rs_bc_idx.get(&b).unwrap_or_else(|| panic!("barcode {b} not in test"))
        })
        .collect();

    let mut total_mismatch = 0u64;
    let mut grand_max = 0i64;
    for layer in ["layers/spliced", "layers/unspliced", "layers/ambiguous"] {
        let (gt_shape, gt_data) = read_layer(&gt, layer);
        let (rs_shape, rs_data) = read_layer(&rs, layer);
        let (gt_g, gt_c) = (gt_shape[0] as usize, gt_shape[1] as usize);
        let rs_c = rs_shape[1] as usize;

        let mut mism = 0u64;
        let mut maxd = 0i64;
        let mut gt_sum = 0u64;
        let mut rs_sum = 0u64;
        for g in 0..gt_g {
            let rg = gt_row_to_rs[g];
            for c in 0..gt_c {
                let rc = gt_col_to_rs[c];
                let a = gt_data[g * gt_c + c] as i64;
                let b = rs_data[rg * rs_c + rc] as i64;
                gt_sum += a as u64;
                rs_sum += b as u64;
                let d = (a - b).abs();
                if d != 0 {
                    mism += 1;
                    if d > maxd {
                        maxd = d;
                    }
                }
            }
        }
        println!(
            "{layer:24} mismatched_cells={mism:6} max_abs_diff={maxd:6} gt_total={gt_sum} test_total={rs_sum}"
        );
        total_mismatch += mism;
        if maxd > grand_max {
            grand_max = maxd;
        }
    }

    if total_mismatch == 0 {
        println!("\nRESULT: IDENTICAL ✓  (all layers bit-for-bit equal after alignment)");
    } else {
        println!("\nRESULT: DIFFERENCES — {total_mismatch} mismatched cells, max abs diff {grand_max}");
        std::process::exit(1);
    }
}
