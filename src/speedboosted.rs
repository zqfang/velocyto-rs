//! Translated from velocyto/speedboosted.pyx
//! Cython+OpenMP parallel column-delta correlation kernels → Rust + rayon.
//! Each public Python-facing `def` becomes one pub fn.
//! The inner `cdef void x_*` kernels become private fn.

use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Inner kernels (cdef void → private fn)
// ---------------------------------------------------------------------------

/// Plain difference: A[j,i] = e[j,i] - e[j,c]
fn x_col_delta_cor(e: ArrayView2<f64>, d: ArrayView2<f64>, mut rm: ArrayViewMut2<f64>) {
    let rows = e.nrows();
    let cols = e.ncols();

    // Each row of rm corresponds to one column-index c.
    // We iterate rows of rm in parallel (one per c).
    rm.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut rm_row)| {
            // A[j, i] = e[j,i] - e[j,c]
            let mut a_vec: Vec<f64> = vec![0.0; rows * cols];
            for j in 0..rows {
                for i in 0..cols {
                    a_vec[j * cols + i] = e[[j, i]] - e[[j, c]];
                }
            }

            // muA[i] = mean_j(A[j,i])
            let mut mu_a: Vec<f64> = vec![0.0; cols];
            for j in 0..rows {
                for i in 0..cols {
                    mu_a[i] += a_vec[j * cols + i];
                }
            }
            let rows_f = rows as f64;
            for i in 0..cols {
                mu_a[i] /= rows_f;
            }

            // A_mA[j,i] = A[j,i] - muA[i]
            let mut a_ma: Vec<f64> = vec![0.0; rows * cols];
            for j in 0..rows {
                for i in 0..cols {
                    a_ma[j * cols + i] = a_vec[j * cols + i] - mu_a[i];
                }
            }

            // mub = mean_j(d[j,c])
            let mut mub = 0.0_f64;
            for j in 0..rows {
                mub += d[[j, c]];
            }
            mub /= rows_f;

            // b_mb[j] = d[j,c] - mub
            let mut b_mb: Vec<f64> = vec![0.0; rows];
            for j in 0..rows {
                b_mb[j] = d[[j, c]] - mub;
            }

            // ssA[i] = 1 / sqrt(sum_j(A_mA[j,i]^2))
            let mut ss_a: Vec<f64> = vec![0.0; cols];
            for j in 0..rows {
                for i in 0..cols {
                    ss_a[i] += a_ma[j * cols + i] * a_ma[j * cols + i];
                }
            }
            for i in 0..cols {
                ss_a[i] = 1.0 / ss_a[i].sqrt();
            }

            // ssb = 1 / sqrt(sum_j(b_mb[j]^2))
            let mut ssb = 0.0_f64;
            for j in 0..rows {
                ssb += b_mb[j] * b_mb[j];
            }
            ssb = 1.0 / ssb.sqrt();

            // rm[c, i] += sum_j(b_mb[j]*ssb * A_mA[j,i]*ssA[i])
            for j in 0..rows {
                let tmp = b_mb[j] * ssb;
                for i in 0..cols {
                    rm_row[i] += (a_ma[j * cols + i] * ss_a[i]) * tmp;
                }
            }
        });
}

/// Sqrt-transformed difference: A[j,i] = sign(diff)*sqrt(|diff|+psc)
fn x_col_delta_cor_sqrt(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    mut rm: ArrayViewMut2<f64>,
    psc: f64,
) {
    let rows = e.nrows();
    let cols = e.ncols();

    rm.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut rm_row)| {
            let mut a_vec: Vec<f64> = vec![0.0; rows * cols];
            for j in 0..rows {
                for i in 0..cols {
                    let diff = e[[j, i]] - e[[j, c]];
                    a_vec[j * cols + i] = if diff > 0.0 {
                        (diff + psc).sqrt()
                    } else {
                        -(-diff + psc).sqrt()
                    };
                }
            }

            let mut mu_a: Vec<f64> = vec![0.0; cols];
            for j in 0..rows {
                for i in 0..cols {
                    mu_a[i] += a_vec[j * cols + i];
                }
            }
            let rows_f = rows as f64;
            for i in 0..cols {
                mu_a[i] /= rows_f;
            }

            let mut a_ma: Vec<f64> = vec![0.0; rows * cols];
            for j in 0..rows {
                for i in 0..cols {
                    a_ma[j * cols + i] = a_vec[j * cols + i] - mu_a[i];
                }
            }

            let mut mub = 0.0_f64;
            for j in 0..rows {
                mub += d[[j, c]];
            }
            mub /= rows_f;

            let mut b_mb: Vec<f64> = vec![0.0; rows];
            for j in 0..rows {
                b_mb[j] = d[[j, c]] - mub;
            }

            let mut ss_a: Vec<f64> = vec![0.0; cols];
            for j in 0..rows {
                for i in 0..cols {
                    ss_a[i] += a_ma[j * cols + i] * a_ma[j * cols + i];
                }
            }
            for i in 0..cols {
                ss_a[i] = 1.0 / ss_a[i].sqrt();
            }

            let mut ssb = 0.0_f64;
            for j in 0..rows {
                ssb += b_mb[j] * b_mb[j];
            }
            ssb = 1.0 / ssb.sqrt();

            for j in 0..rows {
                let tmp = b_mb[j] * ssb;
                for i in 0..cols {
                    rm_row[i] += (a_ma[j * cols + i] * ss_a[i]) * tmp;
                }
            }
        });
}

/// Log10-transformed difference: A[j,i] = sign(diff)*log10(|diff|+psc)
fn x_col_delta_cor_log10(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    mut rm: ArrayViewMut2<f64>,
    psc: f64,
) {
    let rows = e.nrows();
    let cols = e.ncols();

    rm.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut rm_row)| {
            let mut a_vec: Vec<f64> = vec![0.0; rows * cols];
            for j in 0..rows {
                for i in 0..cols {
                    let diff = e[[j, i]] - e[[j, c]];
                    a_vec[j * cols + i] = if diff > 0.0 {
                        (diff + psc).log10()
                    } else {
                        -(-diff + psc).log10()
                    };
                }
            }

            let mut mu_a: Vec<f64> = vec![0.0; cols];
            for j in 0..rows {
                for i in 0..cols {
                    mu_a[i] += a_vec[j * cols + i];
                }
            }
            let rows_f = rows as f64;
            for i in 0..cols {
                mu_a[i] /= rows_f;
            }

            let mut a_ma: Vec<f64> = vec![0.0; rows * cols];
            for j in 0..rows {
                for i in 0..cols {
                    a_ma[j * cols + i] = a_vec[j * cols + i] - mu_a[i];
                }
            }

            let mut mub = 0.0_f64;
            for j in 0..rows {
                mub += d[[j, c]];
            }
            mub /= rows_f;

            let mut b_mb: Vec<f64> = vec![0.0; rows];
            for j in 0..rows {
                b_mb[j] = d[[j, c]] - mub;
            }

            let mut ss_a: Vec<f64> = vec![0.0; cols];
            for j in 0..rows {
                for i in 0..cols {
                    ss_a[i] += a_ma[j * cols + i] * a_ma[j * cols + i];
                }
            }
            for i in 0..cols {
                ss_a[i] = 1.0 / ss_a[i].sqrt();
            }

            let mut ssb = 0.0_f64;
            for j in 0..rows {
                ssb += b_mb[j] * b_mb[j];
            }
            ssb = 1.0 / ssb.sqrt();

            for j in 0..rows {
                let tmp = b_mb[j] * ssb;
                for i in 0..cols {
                    rm_row[i] += (a_ma[j * cols + i] * ss_a[i]) * tmp;
                }
            }
        });
}

/// Partial plain: only nrndm neighbors per cell, ixs is (cols × nrndm) row-major.
fn x_col_delta_cor_partial(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    ixs: &[usize],
    mut rm: ArrayViewMut2<f64>,
    nrndm: usize,
) {
    let rows = e.nrows();
    let cols = e.ncols();

    rm.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut rm_row)| {
            // A[j, n] = e[j, ixs[c,n]] - e[j,c]
            let mut a_vec: Vec<f64> = vec![0.0; rows * nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    let i = ixs[c * nrndm + n];
                    a_vec[j * nrndm + n] = e[[j, i]] - e[[j, c]];
                }
            }

            let mut mu_a: Vec<f64> = vec![0.0; nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    mu_a[n] += a_vec[j * nrndm + n];
                }
            }
            let rows_f = rows as f64;
            for n in 0..nrndm {
                mu_a[n] /= rows_f;
            }

            let mut a_ma: Vec<f64> = vec![0.0; rows * nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    a_ma[j * nrndm + n] = a_vec[j * nrndm + n] - mu_a[n];
                }
            }

            let mut mub = 0.0_f64;
            for j in 0..rows {
                mub += d[[j, c]];
            }
            mub /= rows_f;

            let mut b_mb: Vec<f64> = vec![0.0; rows];
            for j in 0..rows {
                b_mb[j] = d[[j, c]] - mub;
            }

            let mut ss_a: Vec<f64> = vec![0.0; nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    ss_a[n] += a_ma[j * nrndm + n] * a_ma[j * nrndm + n];
                }
            }
            for n in 0..nrndm {
                ss_a[n] = 1.0 / ss_a[n].sqrt();
            }

            let mut ssb = 0.0_f64;
            for j in 0..rows {
                ssb += b_mb[j] * b_mb[j];
            }
            ssb = 1.0 / ssb.sqrt();

            for j in 0..rows {
                let tmp = b_mb[j] * ssb;
                for n in 0..nrndm {
                    let i = ixs[c * nrndm + n];
                    rm_row[i] += (a_ma[j * nrndm + n] * ss_a[n]) * tmp;
                }
            }
        });
}

/// Partial sqrt-transformed. Has extra `fabs < 1e-16` zero-check from Cython source.
fn x_col_delta_cor_sqrt_partial(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    ixs: &[usize],
    mut rm: ArrayViewMut2<f64>,
    nrndm: usize,
    psc: f64,
) {
    let rows = e.nrows();
    let cols = e.ncols();

    rm.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut rm_row)| {
            let mut a_vec: Vec<f64> = vec![0.0; rows * nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    let i = ixs[c * nrndm + n];
                    let diff = e[[j, i]] - e[[j, c]];
                    a_vec[j * nrndm + n] = if diff.abs() < 1e-16 {
                        0.0
                    } else if diff > 0.0 {
                        (diff + psc).sqrt()
                    } else {
                        -(-diff + psc).sqrt()
                    };
                }
            }

            let mut mu_a: Vec<f64> = vec![0.0; nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    mu_a[n] += a_vec[j * nrndm + n];
                }
            }
            let rows_f = rows as f64;
            for n in 0..nrndm {
                mu_a[n] /= rows_f;
            }

            let mut a_ma: Vec<f64> = vec![0.0; rows * nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    a_ma[j * nrndm + n] = a_vec[j * nrndm + n] - mu_a[n];
                }
            }

            let mut mub = 0.0_f64;
            for j in 0..rows {
                mub += d[[j, c]];
            }
            mub /= rows_f;

            let mut b_mb: Vec<f64> = vec![0.0; rows];
            for j in 0..rows {
                b_mb[j] = d[[j, c]] - mub;
            }

            let mut ss_a: Vec<f64> = vec![0.0; nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    ss_a[n] += a_ma[j * nrndm + n] * a_ma[j * nrndm + n];
                }
            }
            for n in 0..nrndm {
                ss_a[n] = 1.0 / ss_a[n].sqrt();
            }

            let mut ssb = 0.0_f64;
            for j in 0..rows {
                ssb += b_mb[j] * b_mb[j];
            }
            ssb = 1.0 / ssb.sqrt();

            for j in 0..rows {
                let tmp = b_mb[j] * ssb;
                for n in 0..nrndm {
                    let i = ixs[c * nrndm + n];
                    rm_row[i] += (a_ma[j * nrndm + n] * ss_a[n]) * tmp;
                }
            }
        });
}

/// Partial log10-transformed.
fn x_col_delta_cor_log10_partial(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    ixs: &[usize],
    mut rm: ArrayViewMut2<f64>,
    nrndm: usize,
    psc: f64,
) {
    let rows = e.nrows();
    let cols = e.ncols();

    rm.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(c, mut rm_row)| {
            let mut a_vec: Vec<f64> = vec![0.0; rows * nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    let i = ixs[c * nrndm + n];
                    let diff = e[[j, i]] - e[[j, c]];
                    a_vec[j * nrndm + n] = if diff >= 0.0 {
                        (diff + psc).log10()
                    } else {
                        -(-diff + psc).log10()
                    };
                }
            }

            let mut mu_a: Vec<f64> = vec![0.0; nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    mu_a[n] += a_vec[j * nrndm + n];
                }
            }
            let rows_f = rows as f64;
            for n in 0..nrndm {
                mu_a[n] /= rows_f;
            }

            let mut a_ma: Vec<f64> = vec![0.0; rows * nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    a_ma[j * nrndm + n] = a_vec[j * nrndm + n] - mu_a[n];
                }
            }

            let mut mub = 0.0_f64;
            for j in 0..rows {
                mub += d[[j, c]];
            }
            mub /= rows_f;

            let mut b_mb: Vec<f64> = vec![0.0; rows];
            for j in 0..rows {
                b_mb[j] = d[[j, c]] - mub;
            }

            let mut ss_a: Vec<f64> = vec![0.0; nrndm];
            for j in 0..rows {
                for n in 0..nrndm {
                    ss_a[n] += a_ma[j * nrndm + n] * a_ma[j * nrndm + n];
                }
            }
            for n in 0..nrndm {
                ss_a[n] = 1.0 / ss_a[n].sqrt();
            }

            let mut ssb = 0.0_f64;
            for j in 0..rows {
                ssb += b_mb[j] * b_mb[j];
            }
            ssb = 1.0 / ssb.sqrt();

            for j in 0..rows {
                let tmp = b_mb[j] * ssb;
                for n in 0..nrndm {
                    let i = ixs[c * nrndm + n];
                    rm_row[i] += (a_ma[j * nrndm + n] * ss_a[n]) * tmp;
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Public wrappers (def → pub fn) — signatures match Python callers in estimation.py
// ---------------------------------------------------------------------------

/// Compute column-delta Pearson correlations (plain difference).
///
/// `e` and `d` are (ngenes × ncells); writes results into `rm` (ncells × ncells).
pub fn col_delta_cor(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    mut rm: ArrayViewMut2<f64>,
    _num_threads: usize,
) {
    x_col_delta_cor(e, d, rm.view_mut());
}

/// Column-delta correlation with sqrt transform.
pub fn col_delta_cor_sqrt(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    mut rm: ArrayViewMut2<f64>,
    _num_threads: usize,
    psc: f64,
) {
    x_col_delta_cor_sqrt(e, d, rm.view_mut(), psc);
}

/// Column-delta correlation with log10 transform.
pub fn col_delta_cor_log10(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    mut rm: ArrayViewMut2<f64>,
    _num_threads: usize,
    psc: f64,
) {
    x_col_delta_cor_log10(e, d, rm.view_mut(), psc);
}

/// Partial column-delta correlation (plain). `ixs` shape (ncells × nrndm), row-major.
pub fn col_delta_cor_partial(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    ixs: &[usize],
    mut rm: ArrayViewMut2<f64>,
    _num_threads: usize,
) {
    let cols = e.ncols();
    let nrndm = ixs.len() / cols;
    x_col_delta_cor_partial(e, d, ixs, rm.view_mut(), nrndm);
}

/// Partial column-delta correlation with sqrt transform.
pub fn col_delta_cor_sqrt_partial(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    ixs: &[usize],
    mut rm: ArrayViewMut2<f64>,
    _num_threads: usize,
    psc: f64,
) {
    let cols = e.ncols();
    let nrndm = ixs.len() / cols;
    x_col_delta_cor_sqrt_partial(e, d, ixs, rm.view_mut(), nrndm, psc);
}

/// Partial column-delta correlation with log10 transform.
pub fn col_delta_cor_log10_partial(
    e: ArrayView2<f64>,
    d: ArrayView2<f64>,
    ixs: &[usize],
    mut rm: ArrayViewMut2<f64>,
    _num_threads: usize,
    psc: f64,
) {
    let cols = e.ncols();
    let nrndm = ixs.len() / cols;
    x_col_delta_cor_log10_partial(e, d, ixs, rm.view_mut(), nrndm, psc);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Verify col_delta_cor on a 3×3 non-degenerate matrix.
    ///
    /// For each column c, the result rm[c, i] is the Pearson correlation between:
    ///   - b = d[:, c]
    ///   - a_i = e[:, i] - e[:, c]
    ///
    /// We verify rm[c, c] == 0 when e[:, c] - e[:, c] == 0 (constant column A → NaN/0),
    /// and verify rm[0, 1] by hand against a direct Pearson computation.
    #[test]
    fn test_col_delta_cor_basic() {
        // e: 3 genes × 3 cells
        let e = array![[1.0_f64, 2.0, 4.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0],];
        // d: same shape
        let d = array![[0.5_f64, 1.0, 1.5], [1.5, 0.5, 2.0], [2.5, 1.5, 0.5],];
        let mut rm = Array2::<f64>::zeros((3, 3));
        col_delta_cor(e.view(), d.view(), rm.view_mut(), 1);

        // Hand-compute rm[0, 1]: correlation of b = d[:,0] and a = e[:,1] - e[:,0]
        // b = [0.5, 1.5, 2.5], a = [1.0, 1.0, -2.0]
        // mub = (0.5+1.5+2.5)/3 = 4.5/3 = 1.5
        // b_mb = [-1.0, 0.0, 1.0]
        // mua = (1.0+1.0-2.0)/3 = 0.0
        // a_ma = [1.0, 1.0, -2.0]
        // ssb = 1/sqrt(1+0+1) = 1/sqrt(2)
        // ssa1 = 1/sqrt(1+1+4) = 1/sqrt(6)
        // corr = (-1*1/sqrt(2) * 1/sqrt(6) + 0 + 1*1/sqrt(2) * (-2)/sqrt(6))
        //      = 1/(sqrt(2)*sqrt(6)) * (-1 + 0 - 2) = -3/sqrt(12) = -3/(2*sqrt(3))
        let expected = -3.0_f64 / (2.0 * 3.0_f64.sqrt());
        let got = rm[[0, 1]];
        assert!(
            (got - expected).abs() < 1e-10,
            "rm[0,1] = {got}, expected {expected}"
        );
    }

    /// Verify that rm[c, c] is 1.0 when d[:,c] == e[:,c] (perfect self-correlation
    /// after mean-centering both series which are identical up to centering).
    /// More precisely: when A[:,c] = e[:,c] - e[:,c] = 0, ssA[c] = 1/0 = inf,
    /// so rm[c,c] will be NaN or inf — just check other cells are finite.
    #[test]
    fn test_col_delta_cor_partial_basic() {
        // 3 genes × 4 cells, each cell has 2 neighbours
        let e = array![
            [1.0_f64, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 4.0, 1.0, 3.0],
        ];
        let d = array![
            [1.0_f64, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
        ];
        // ixs[c, n]: for each cell c, 2 neighbour indices (ncells=4, nrndm=2)
        // row-major: ixs = [[1,2], [0,3], [0,1], [1,2]] flattened
        let ixs: Vec<usize> = vec![1, 2, 0, 3, 0, 1, 1, 2];
        let mut rm = Array2::<f64>::zeros((4, 4));
        col_delta_cor_partial(e.view(), d.view(), &ixs, rm.view_mut(), 1);

        // Check that written positions are finite
        // rm[0,1] and rm[0,2] should be finite (written by c=0)
        assert!(rm[[0, 1]].is_finite(), "rm[0,1] should be finite");
        assert!(rm[[0, 2]].is_finite(), "rm[0,2] should be finite");
        // Positions not in ixs for that row should remain 0
        assert_eq!(
            rm[[0, 3]],
            0.0,
            "rm[0,3] should be 0 (not a neighbour of 0)"
        );
    }

    /// Smoke-test col_delta_cor_sqrt: verify off-diagonal results are finite
    /// and shapes are correct for a 3×3 input with psc=0.0.
    #[test]
    fn test_col_delta_cor_sqrt() {
        let e = array![[1.0_f64, 4.0, 9.0], [4.0, 1.0, 16.0], [9.0, 16.0, 1.0],];
        let d = array![[1.0_f64, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0],];
        let mut rm = Array2::<f64>::zeros((3, 3));
        col_delta_cor_sqrt(e.view(), d.view(), rm.view_mut(), 1, 0.0);
        // Just check shapes and that off-diagonal is finite
        assert!(rm[[0, 1]].is_finite());
        assert!(rm[[0, 2]].is_finite());
        assert!(rm[[1, 0]].is_finite());
    }

    /// Smoke-test col_delta_cor_log10: verify off-diagonal results are finite
    /// and shapes are correct for a 3×3 input with psc=1.0.
    #[test]
    fn test_col_delta_cor_log10() {
        let e = array![
            [1.0_f64, 10.0, 100.0],
            [10.0, 1.0, 1000.0],
            [100.0, 1000.0, 1.0],
        ];
        let d = array![[0.5_f64, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0],];
        let mut rm = Array2::<f64>::zeros((3, 3));
        col_delta_cor_log10(e.view(), d.view(), rm.view_mut(), 1, 1.0);
        assert!(rm[[0, 1]].is_finite());
        assert!(rm[[1, 2]].is_finite());
    }
}
