//! Translated from velocyto/estimation.py

use crate::speedboosted;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

// ---------------------------------------------------------------------------
// colDeltaCor* wrappers
// ---------------------------------------------------------------------------

/// Returns thread count: cpu_count/2 when None, or max(t, cpu_count) when explicit.
fn default_threads(threads: Option<usize>) -> usize {
    match threads {
        Some(t) => t.max(rayon::current_num_threads()),
        None => (rayon::current_num_threads() / 2).max(1),
    }
}

/// Computes column-delta Pearson correlations. For each cell i, correlates d[:,i]
/// with (e - e[:,i]) across genes. Args: emat (ngenes×ncells), dmat (ngenes×ncells).
/// Uses parallel threads.
pub fn col_delta_cor(
    emat: ArrayView2<f64>,
    dmat: ArrayView2<f64>,
    threads: Option<usize>,
) -> Array2<f64> {
    let ncells = emat.ncols();
    let num_threads = default_threads(threads);
    let mut out = Array2::<f64>::zeros((ncells, ncells));
    speedboosted::col_delta_cor(emat, dmat, out.view_mut(), num_threads);
    out
}

/// Partial column-delta correlations (plain difference).
pub fn col_delta_cor_partial(
    emat: ArrayView2<f64>,
    dmat: ArrayView2<f64>,
    ixs: &[usize],
    threads: Option<usize>,
) -> Array2<f64> {
    let ncells = emat.ncols();
    let num_threads = default_threads(threads);
    let mut out = Array2::<f64>::zeros((ncells, ncells));
    speedboosted::col_delta_cor_partial(emat, dmat, ixs, out.view_mut(), num_threads);
    out
}

/// Computes column-delta Pearson correlations with log10 transform. For each cell i,
/// correlates log10(d[:,i]+psc) with log10(e - e[:,i]+psc) across genes.
/// Args: emat (ngenes×ncells), dmat (ngenes×ncells). Uses parallel threads.
pub fn col_delta_cor_log10(
    emat: ArrayView2<f64>,
    dmat: ArrayView2<f64>,
    threads: Option<usize>,
    psc: f64,
) -> Array2<f64> {
    let ncells = emat.ncols();
    let num_threads = default_threads(threads);
    let mut out = Array2::<f64>::zeros((ncells, ncells));
    speedboosted::col_delta_cor_log10(emat, dmat, out.view_mut(), num_threads, psc);
    out
}

/// Partial column-delta correlations with log10 transform.
pub fn col_delta_cor_log10_partial(
    emat: ArrayView2<f64>,
    dmat: ArrayView2<f64>,
    ixs: &[usize],
    threads: Option<usize>,
    psc: f64,
) -> Array2<f64> {
    let ncells = emat.ncols();
    let num_threads = default_threads(threads);
    let mut out = Array2::<f64>::zeros((ncells, ncells));
    speedboosted::col_delta_cor_log10_partial(emat, dmat, ixs, out.view_mut(), num_threads, psc);
    out
}

/// Computes column-delta Pearson correlations with sqrt transform. For each cell i,
/// correlates sqrt(d[:,i]+psc) with sqrt(e - e[:,i]+psc) across genes.
/// Args: emat (ngenes×ncells), dmat (ngenes×ncells). Uses parallel threads.
pub fn col_delta_cor_sqrt(
    emat: ArrayView2<f64>,
    dmat: ArrayView2<f64>,
    threads: Option<usize>,
    psc: f64,
) -> Array2<f64> {
    let ncells = emat.ncols();
    let num_threads = default_threads(threads);
    let mut out = Array2::<f64>::zeros((ncells, ncells));
    speedboosted::col_delta_cor_sqrt(emat, dmat, out.view_mut(), num_threads, psc);
    out
}

/// Partial column-delta correlations with sqrt transform.
pub fn col_delta_cor_sqrt_partial(
    emat: ArrayView2<f64>,
    dmat: ArrayView2<f64>,
    ixs: &[usize],
    threads: Option<usize>,
    psc: f64,
) -> Array2<f64> {
    let ncells = emat.ncols();
    let num_threads = default_threads(threads);
    let mut out = Array2::<f64>::zeros((ncells, ncells));
    speedboosted::col_delta_cor_sqrt_partial(emat, dmat, ixs, out.view_mut(), num_threads, psc);
    out
}

// ---------------------------------------------------------------------------
// Slope-fitting helpers
// ---------------------------------------------------------------------------

/// Fits a single ordinary least squares slope using NNLS. Returns NaN if x is all zeros,
/// 0 if y is all zeros.
pub fn fit1_slope(y: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let xx: f64 = x.iter().map(|&v| v * v).sum();
    if xx == 0.0 {
        return f64::NAN;
    }
    let any_y = y.iter().any(|&v| v != 0.0);
    if !any_y {
        return 0.0;
    }
    let xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    // Non-negative least squares through origin: clip to 0 from below
    (xy / xx).max(0.0)
}

/// Fit linear regression through the origin with weights.
///
/// Minimises sum(w * (x*m - y)^2) over m >= 0.
/// The closed-form weighted OLS solution is m = sum(w*x*y) / sum(w*x*x),
/// clamped to [bounds.0, bounds.1].  When `limit_gamma` is true the upper
/// bound is tightened using the 90th-percentile heuristic from estimation.py.
pub fn fit1_slope_weighted(
    y: ArrayView1<f64>,
    x: ArrayView1<f64>,
    w: ArrayView1<f64>,
    limit_gamma: bool,
    bounds: (f64, f64),
) -> f64 {
    let any_x = x.iter().any(|&v| v != 0.0);
    if !any_x {
        return f64::NAN;
    }
    let any_y = y.iter().any(|&v| v != 0.0);
    if !any_y {
        return 0.0;
    }

    let up_gamma = if limit_gamma {
        let med_y = median_f64(y);
        let med_x = median_f64(x);
        if med_y > med_x {
            // high_x = x > percentile(x, 90)
            let p90 = percentile_f64(x, 90.0);
            let (sum_yh, cnt_yh): (f64, usize) = x
                .iter()
                .zip(y.iter())
                .filter(|(&xi, _)| xi > p90)
                .fold((0.0, 0), |(s, c), (_, &yi)| (s + yi, c + 1));
            let high_x_y: Vec<f64> = x
                .iter()
                .zip(y.iter())
                .filter(|(&xi, _)| xi > p90)
                .map(|(_, &yi)| yi)
                .collect();
            let p10_high_y = if high_x_y.is_empty() {
                1.5
            } else {
                let mut sorted = high_x_y.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                percentile_sorted(&sorted, 10.0)
            };
            let high_x_vals: Vec<f64> = x.iter().filter(|&&xi| xi > p90).copied().collect();
            let med_high_x = if high_x_vals.is_empty() {
                1.0
            } else {
                let mut s = high_x_vals.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                percentile_sorted(&s, 50.0)
            };
            (p10_high_y / med_high_x).max(1.5)
        } else {
            1.5
        }
    } else {
        bounds.1
    };

    let lo = if limit_gamma { 1e-8_f64 } else { bounds.0 };
    let hi = up_gamma;

    // Closed-form weighted OLS through origin: m = sum(w*x*y)/sum(w*x*x)
    let wxsq: f64 = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi * xi).sum();
    let wxy: f64 = w
        .iter()
        .zip(x.iter())
        .zip(y.iter())
        .map(|((&wi, &xi), &yi)| wi * xi * yi)
        .sum();
    let m = if wxsq == 0.0 { 0.0 } else { wxy / wxsq };
    m.clamp(lo, hi)
}

/// Fit linear regression with intercept: y = m*x + q.
///
/// When `fixperc_q` is true: q = median(y where x <= percentile(x,1)),
/// then m = sum((y-q)*x) / sum(x*x) clamped to [0,20].
/// Otherwise: closed-form OLS with intercept.
pub fn fit1_slope_offset(y: ArrayView1<f64>, x: ArrayView1<f64>, fixperc_q: bool) -> (f64, f64) {
    let any_x = x.iter().any(|&v| v != 0.0);
    if !any_x {
        return (f64::NAN, 0.0);
    }
    let any_y = y.iter().any(|&v| v != 0.0);
    if !any_y {
        return (0.0, 0.0);
    }

    if fixperc_q {
        let p1 = percentile_f64(x, 1.0);
        let low_y: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .filter(|(&xi, _)| xi <= p1)
            .map(|(_, &yi)| yi)
            .collect();
        let q = if low_y.is_empty() {
            0.0
        } else {
            let mut s = low_y;
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            percentile_sorted(&s, 50.0)
        };
        // m = argmin sum((x*m - y + q)^2) for m in [0, 20]
        // closed form: m = sum(x*(y-q)) / sum(x*x) clamped to [0,20]
        let xsq: f64 = x.iter().map(|&xi| xi * xi).sum();
        let m = if xsq == 0.0 {
            0.0
        } else {
            let xy_q: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * (yi - q)).sum();
            (xy_q / xsq).clamp(0.0, 20.0)
        };
        (m, q)
    } else {
        // OLS with intercept: m, q = (X^T X)^{-1} X^T y where X = [x | 1]
        ols_with_intercept(y, x)
    }
}

/// Fit weighted linear regression with intercept.
///
/// When `fixperc_q`: q = median(y where x <= p1), m fitted via weighted OLS.
/// Otherwise: weighted OLS with intercept, bounds on m via limit_gamma heuristic.
pub fn fit1_slope_weighted_offset(
    y: ArrayView1<f64>,
    x: ArrayView1<f64>,
    w: ArrayView1<f64>,
    fixperc_q: bool,
    limit_gamma: bool,
) -> (f64, f64) {
    let any_x = x.iter().any(|&v| v != 0.0);
    if !any_x {
        return (f64::NAN, 0.0);
    }
    let any_y = y.iter().any(|&v| v != 0.0);
    if !any_y {
        return (0.0, 0.0);
    }

    if fixperc_q {
        let p1 = percentile_f64(x, 1.0);
        let low_y: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .filter(|(&xi, _)| xi <= p1)
            .map(|(_, &yi)| yi)
            .collect();
        let q = if low_y.is_empty() {
            0.0
        } else {
            let mut s = low_y;
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            percentile_sorted(&s, 50.0)
        };
        // m = argmin sum(w*(x*m - y + q)^2), closed form
        let wxsq: f64 = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi * xi).sum();
        let m = if wxsq == 0.0 {
            0.0
        } else {
            let wxy_q: f64 = w
                .iter()
                .zip(x.iter())
                .zip(y.iter())
                .map(|((&wi, &xi), &yi)| wi * xi * (yi - q))
                .sum();
            (wxy_q / wxsq).clamp(0.0, 20.0)
        };
        (m, q)
    } else {
        let up_gamma = if limit_gamma {
            let med_y = median_f64(y);
            let med_x = median_f64(x);
            if med_y > med_x {
                let p90 = percentile_f64(x, 90.0);
                let high_y: Vec<f64> = x
                    .iter()
                    .zip(y.iter())
                    .filter(|(&xi, _)| xi > p90)
                    .map(|(_, &yi)| yi)
                    .collect();
                let high_x: Vec<f64> = x.iter().filter(|&&xi| xi > p90).copied().collect();
                if high_y.is_empty() {
                    1.5
                } else {
                    let mut sy = high_y.clone();
                    sy.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p10y = percentile_sorted(&sy, 10.0);
                    let mut sx = high_x;
                    sx.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med_hx = percentile_sorted(&sx, 50.0);
                    (p10y / med_hx).max(1.5)
                }
            } else {
                1.5
            }
        } else {
            20.0
        };

        // up_q = 2 * sum(y*w) / sum(w)
        let sum_w: f64 = w.iter().sum();
        let sum_yw: f64 = w.iter().zip(y.iter()).map(|(&wi, &yi)| wi * yi).sum();
        let up_q = if sum_w == 0.0 {
            0.0
        } else {
            2.0 * sum_yw / sum_w
        };

        // Weighted OLS with intercept, clamped
        let (m_raw, q_raw) = weighted_ols_with_intercept(y, x, w);
        let m = m_raw.clamp(1e-8, up_gamma);
        let q = q_raw.clamp(0.0, up_q);
        (m, q)
    }
}

// ---------------------------------------------------------------------------
// Per-gene loop wrappers
// ---------------------------------------------------------------------------

/// Fits slopes for all cells using OLS/NNLS.
/// Applies `fit1_slope` to each gene row of Y,X (ngenes × ncells). Returns slopes vector of length ngenes.
pub fn fit_slope(y_mat: ArrayView2<f64>, x_mat: ArrayView2<f64>) -> Array1<f64> {
    let ngenes = y_mat.nrows();
    let mut slopes = Array1::<f64>::zeros(ngenes);
    for i in 0..ngenes {
        slopes[i] = fit1_slope(y_mat.row(i), x_mat.row(i));
    }
    slopes
}

/// Fits slopes with offsets (intercept) for all cells.
/// Applies `fit1_slope_offset` to each gene row. Returns (slopes, offsets) each of length ngenes.
pub fn fit_slope_offset(
    y_mat: ArrayView2<f64>,
    x_mat: ArrayView2<f64>,
    fixperc_q: bool,
) -> (Array1<f64>, Array1<f64>) {
    let ngenes = y_mat.nrows();
    let mut slopes = Array1::<f64>::zeros(ngenes);
    let mut offsets = Array1::<f64>::zeros(ngenes);
    for i in 0..ngenes {
        let (m, q) = fit1_slope_offset(y_mat.row(i), x_mat.row(i), fixperc_q);
        slopes[i] = m;
        offsets[i] = q;
    }
    (slopes, offsets)
}

/// Fits weighted slopes for all genes.
/// Applies `fit1_slope_weighted` to each gene row.
/// Returns slopes vector of length ngenes.
pub fn fit_slope_weighted(
    y_mat: ArrayView2<f64>,
    x_mat: ArrayView2<f64>,
    w_mat: ArrayView2<f64>,
    limit_gamma: bool,
    bounds: (f64, f64),
) -> Array1<f64> {
    let ngenes = y_mat.nrows();
    let mut slopes = Array1::<f64>::zeros(ngenes);
    for i in 0..ngenes {
        slopes[i] = fit1_slope_weighted(
            y_mat.row(i),
            x_mat.row(i),
            w_mat.row(i),
            limit_gamma,
            bounds,
        );
    }
    slopes
}

/// Fits weighted slopes for all genes, also returning unweighted R² per gene.
/// R² = 1 - SSres/SStot (residuals are unweighted even though the fit is weighted).
/// Genes where R² is not finite get -1e16.
pub fn fit_slope_weighted_r2(
    y_mat: ArrayView2<f64>,
    x_mat: ArrayView2<f64>,
    w_mat: ArrayView2<f64>,
    limit_gamma: bool,
    bounds: (f64, f64),
) -> (Array1<f64>, Array1<f64>) {
    let ngenes = y_mat.nrows();
    let mut slopes = Array1::<f64>::zeros(ngenes);
    let mut r2 = Array1::<f64>::zeros(ngenes);
    for i in 0..ngenes {
        let m = fit1_slope_weighted(
            y_mat.row(i),
            x_mat.row(i),
            w_mat.row(i),
            limit_gamma,
            bounds,
        );
        slopes[i] = m;
        let y = y_mat.row(i);
        let x = x_mat.row(i);
        let y_mean = y.mean().unwrap_or(0.0);
        let ssres: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (m * xi - yi).powi(2))
            .sum();
        let sstot: f64 = y.iter().map(|&yi| (y_mean - yi).powi(2)).sum();
        let v = 1.0 - ssres / sstot;
        r2[i] = if v.is_finite() { v } else { -1e16 };
    }
    (slopes, r2)
}

/// Fits weighted slopes with offsets (intercepts) for all genes.
/// Returns `(slopes, offsets)` each of length ngenes.
pub fn fit_slope_weighted_offset(
    y_mat: ArrayView2<f64>,
    x_mat: ArrayView2<f64>,
    w_mat: ArrayView2<f64>,
    fixperc_q: bool,
    limit_gamma: bool,
) -> (Array1<f64>, Array1<f64>) {
    let ngenes = y_mat.nrows();
    let mut slopes = Array1::<f64>::zeros(ngenes);
    let mut offsets = Array1::<f64>::zeros(ngenes);
    for i in 0..ngenes {
        let (m, q) = fit1_slope_weighted_offset(
            y_mat.row(i),
            x_mat.row(i),
            w_mat.row(i),
            fixperc_q,
            limit_gamma,
        );
        slopes[i] = m;
        offsets[i] = q;
    }
    (slopes, offsets)
}

/// Fits weighted slopes with offsets for all genes, also returning unweighted R² per gene.
/// R² = 1 - SSres/SStot where residuals use `m*x + q - y` (unweighted).
/// Genes where R² is not finite get -1e16.
pub fn fit_slope_weighted_offset_r2(
    y_mat: ArrayView2<f64>,
    x_mat: ArrayView2<f64>,
    w_mat: ArrayView2<f64>,
    fixperc_q: bool,
    limit_gamma: bool,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let ngenes = y_mat.nrows();
    let mut slopes = Array1::<f64>::zeros(ngenes);
    let mut offsets = Array1::<f64>::zeros(ngenes);
    let mut r2 = Array1::<f64>::zeros(ngenes);
    for i in 0..ngenes {
        let (m, q) = fit1_slope_weighted_offset(
            y_mat.row(i),
            x_mat.row(i),
            w_mat.row(i),
            fixperc_q,
            limit_gamma,
        );
        slopes[i] = m;
        offsets[i] = q;
        let y = y_mat.row(i);
        let x = x_mat.row(i);
        let y_mean = y.mean().unwrap_or(0.0);
        let ssres: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (m * xi + q - yi).powi(2))
            .sum();
        let sstot: f64 = y.iter().map(|&yi| (y_mean - yi).powi(2)).sum();
        let v = 1.0 - ssres / sstot;
        r2[i] = if v.is_finite() { v } else { -1e16 };
    }
    (slopes, offsets, r2)
}

// ---------------------------------------------------------------------------
// Cluster statistics
// ---------------------------------------------------------------------------

/// Computes per-cluster mean and standard deviation of spliced/unspliced counts.
///
/// `u`, `s`: unspliced/spliced count matrices (ngenes × ncells)
/// `clusters_uid`: unique cluster labels (length = number of clusters)
/// `cluster_ix`: cluster index for each cell (length = ncells), values in 0..clusters_uid.len()
/// `size_limit`: minimum cluster size; clusters with fewer cells use the global mean instead.
///
/// Returns `(u_avgs, s_avgs)` each of shape (ngenes × nclusters).
pub fn clusters_stats(
    u: ArrayView2<f64>,
    s: ArrayView2<f64>,
    clusters_uid: &[usize],
    cluster_ix: &[usize],
    size_limit: usize,
) -> (Array2<f64>, Array2<f64>) {
    let ngenes = s.nrows();
    let nclusters = clusters_uid.len();
    let mut u_avgs = Array2::<f64>::zeros((ngenes, nclusters));
    let mut s_avgs = Array2::<f64>::zeros((ngenes, nclusters));

    // Pre-compute global means
    let ncells = u.ncols() as f64;
    let u_global: Vec<f64> = (0..ngenes)
        .map(|g| u.row(g).iter().sum::<f64>() / ncells)
        .collect();
    let s_global: Vec<f64> = (0..ngenes)
        .map(|g| s.row(g).iter().sum::<f64>() / ncells)
        .collect();

    for (i, _uid) in clusters_uid.iter().enumerate() {
        let mask: Vec<bool> = cluster_ix.iter().map(|&c| c == i).collect();
        let n_cells: usize = mask.iter().filter(|&&b| b).count();

        if n_cells > size_limit {
            let n_f = n_cells as f64;
            for g in 0..ngenes {
                let row_u = u.row(g);
                let row_s = s.row(g);
                let sum_u: f64 = row_u
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&v, _)| v)
                    .sum();
                let sum_s: f64 = row_s
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&v, _)| v)
                    .sum();
                u_avgs[[g, i]] = sum_u / n_f;
                s_avgs[[g, i]] = sum_s / n_f;
            }
        } else {
            for g in 0..ngenes {
                u_avgs[[g, i]] = u_global[g];
                s_avgs[[g, i]] = s_global[g];
            }
        }
    }

    (u_avgs, s_avgs)
}

// ---------------------------------------------------------------------------
// Internal statistical helpers
// ---------------------------------------------------------------------------

/// OLS with intercept: m, q = argmin sum((y - m*x - q)^2).
/// Normal equations: [sum(x^2), sum(x); sum(x), n] [m; q] = [sum(x*y); sum(y)]
fn ols_with_intercept(y: ArrayView1<f64>, x: ArrayView1<f64>) -> (f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let det = n * sxx - sx * sx;
    if det.abs() < 1e-15 {
        return (0.0, sy / n);
    }
    let m = (n * sxy - sx * sy) / det;
    let q = (sy - m * sx) / n;
    (m, q)
}

/// Weighted OLS with intercept: argmin sum(w*(y - m*x - q)^2).
fn weighted_ols_with_intercept(
    y: ArrayView1<f64>,
    x: ArrayView1<f64>,
    w: ArrayView1<f64>,
) -> (f64, f64) {
    let sw: f64 = w.iter().sum();
    let swx: f64 = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
    let swy: f64 = w.iter().zip(y.iter()).map(|(&wi, &yi)| wi * yi).sum();
    let swxx: f64 = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi * xi).sum();
    let swxy: f64 = w
        .iter()
        .zip(x.iter())
        .zip(y.iter())
        .map(|((&wi, &xi), &yi)| wi * xi * yi)
        .sum();
    let det = sw * swxx - swx * swx;
    if det.abs() < 1e-15 {
        let q = if sw == 0.0 { 0.0 } else { swy / sw };
        return (0.0, q);
    }
    let m = (sw * swxy - swx * swy) / det;
    let q = (swy - m * swx) / sw;
    (m, q)
}

/// Compute the p-th percentile of an ArrayView1 (0 ≤ p ≤ 100).
fn percentile_f64(arr: ArrayView1<f64>, p: f64) -> f64 {
    let mut sorted: Vec<f64> = arr.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile_sorted(&sorted, p)
}

/// Compute the p-th percentile of an already-sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Median of an ArrayView1.
fn median_f64(arr: ArrayView1<f64>) -> f64 {
    percentile_f64(arr, 50.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fit1_slope_basic() {
        // y = 2x → slope should be 2
        let x = array![1.0_f64, 2.0, 3.0];
        let y = array![2.0_f64, 4.0, 6.0];
        let m = fit1_slope(y.view(), x.view());
        assert!((m - 2.0).abs() < 1e-10, "slope={m}");
    }

    #[test]
    fn test_fit1_slope_zero_x() {
        let x = array![0.0_f64, 0.0, 0.0];
        let y = array![1.0_f64, 2.0, 3.0];
        let m = fit1_slope(y.view(), x.view());
        assert!(m.is_nan(), "should be NaN when x is all zero");
    }

    #[test]
    fn test_fit1_slope_zero_y() {
        let x = array![1.0_f64, 2.0, 3.0];
        let y = array![0.0_f64, 0.0, 0.0];
        let m = fit1_slope(y.view(), x.view());
        assert_eq!(m, 0.0, "should be 0 when y is all zero");
    }

    #[test]
    fn test_fit1_slope_offset_ols() {
        // y = 2x + 1
        let x = array![1.0_f64, 2.0, 3.0, 4.0];
        let y = array![3.0_f64, 5.0, 7.0, 9.0];
        let (m, q) = fit1_slope_offset(y.view(), x.view(), false);
        assert!((m - 2.0).abs() < 1e-9, "slope={m}");
        assert!((q - 1.0).abs() < 1e-9, "offset={q}");
    }

    #[test]
    fn test_fit_slope_matrix() {
        // Two genes: y_0 = 3*x, y_1 = 0.5*x
        let x = array![[1.0_f64, 2.0, 3.0], [2.0, 4.0, 6.0]];
        let y = array![[3.0_f64, 6.0, 9.0], [1.0, 2.0, 3.0]];
        let slopes = fit_slope(y.view(), x.view());
        assert!((slopes[0] - 3.0).abs() < 1e-9, "slopes[0]={}", slopes[0]);
        assert!((slopes[1] - 0.5).abs() < 1e-9, "slopes[1]={}", slopes[1]);
    }

    #[test]
    fn test_clusters_stats() {
        // 2 genes, 6 cells, 2 clusters (3 cells each)
        let u = array![
            [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let s = array![
            [2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0],
            [12.0, 10.0, 8.0, 6.0, 4.0, 2.0],
        ];
        // cluster 0: cells 0,1,2; cluster 1: cells 3,4,5
        let clusters_uid = vec![0usize, 1];
        let cluster_ix = vec![0usize, 0, 0, 1, 1, 1];
        let (u_avgs, s_avgs) = clusters_stats(u.view(), s.view(), &clusters_uid, &cluster_ix, 2);
        // cluster 0, gene 0: mean of [1,2,3] = 2.0
        assert!((u_avgs[[0, 0]] - 2.0).abs() < 1e-10);
        // cluster 1, gene 0: mean of [4,5,6] = 5.0
        assert!((u_avgs[[0, 1]] - 5.0).abs() < 1e-10);
        // cluster 0, gene 1: mean of [6,5,4] = 5.0
        assert!((u_avgs[[1, 0]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_col_delta_cor_estimation_wrapper() {
        let e = array![[1.0_f64, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0],];
        let d = array![[1.0_f64, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0],];
        let rm = col_delta_cor(e.view(), d.view(), Some(1));
        assert_eq!(rm.shape(), &[3, 3]);
        // off-diagonal should be finite
        assert!(rm[[0, 1]].is_finite());
        assert!(rm[[1, 2]].is_finite());
    }
}
