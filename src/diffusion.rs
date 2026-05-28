//! Translated from velocyto/diffusion.py

use ndarray::{s, Array1, Array2, Axis};
use rand::Rng;
use sprs::{CsMat, TriMat};
use statrs::distribution::{Continuous, Normal};

/// Computes Markov transition matrices and runs diffusion processes on them
/// (random walks, path integrals, time evolution).
pub struct Diffusion;

impl Default for Diffusion {
    fn default() -> Self {
        Diffusion
    }
}

impl Diffusion {
    /// Creates a new Diffusion instance.
    pub fn new() -> Self {
        Diffusion
    }

    /// Compute a right-stochastic transition matrix using KNN on projected positions.
    ///
    /// Python:
    ///   x1 = x0 ± v
    ///   nn = NearestNeighbors(n_neighbors=20).fit(x0).kneighbors(x1)
    ///   probs = norm.pdf(dists, 0, sigma)
    ///   tr = normalize(coo_matrix((probs, (cells, nearest))), norm='l1')
    pub fn compute_transition_matrix2(
        &self,
        x0: &Array2<f64>,
        v: &Array2<f64>,
        sigma: f64,
        reverse: bool,
    ) -> CsMat<f64> {
        let n_cells = x0.nrows();
        let n_neighbors: usize = 20;

        // Project into future or past
        let x1: Array2<f64> = if reverse { x0 - v } else { x0 + v };

        // KNN: find n_neighbors nearest neighbours in x0 for each row of x1
        let (dists_flat, nearest_flat) = knn_search(x0, &x1, n_neighbors);

        // Normal PDF: probs[i] = N(0, sigma).pdf(dist[i])
        let normal = Normal::new(0.0, sigma.max(f64::EPSILON)).expect("Normal::new failed");
        let probs: Vec<f64> = dists_flat.iter().map(|&d| normal.pdf(d)).collect();

        // Build COO sparse matrix (n_cells × n_cells)
        let mut tri: TriMat<f64> = TriMat::new((n_cells, n_cells));
        for cell in 0..n_cells {
            for nb in 0..n_neighbors {
                let flat_idx = cell * n_neighbors + nb;
                let row = cell;
                let col = nearest_flat[flat_idx];
                tri.add_triplet(row, col, probs[flat_idx]);
            }
        }
        let csr = tri.to_csr();

        // L1-normalise each row (right-stochastic)
        l1_normalize_rows(csr)
    }

    /// Compute a right-stochastic transition matrix from a pre-computed KNN graph.
    ///
    /// Python:
    ///   uv = (x[v1] - x[v0]) / norm        (unit edge vectors)
    ///   scalar_proj = v[v0] · uv            (dot product per edge)
    ///   p = clip(scalar_proj + epsilon, 0) / norms
    ///   tr = normalize(coo_matrix((p, (v0, v1))), norm='l1')
    pub fn compute_transition_matrix(
        &self,
        knn: &CsMat<f64>,
        x: &Array2<f64>,
        v: &Array2<f64>,
        epsilon: f64,
        reverse: bool,
    ) -> CsMat<f64> {
        let n_cells = x.nrows();

        // Extract COO representation via iter()
        // Collect (row, col) pairs from the CSR matrix
        let edge_pairs: Vec<(usize, usize)> = knn.iter().map(|(_val, (r, c))| (r, c)).collect();
        let rows: Vec<usize> = edge_pairs.iter().map(|&(r, _)| r).collect();
        let cols: Vec<usize> = edge_pairs.iter().map(|&(_, c)| c).collect();

        let n_edges = rows.len();
        let mut edge_probs: Vec<f64> = Vec::with_capacity(n_edges);

        for i in 0..n_edges {
            let v0 = rows[i];
            let v1 = cols[i];

            // Edge vector from v0 to v1
            let x0_row = x.row(v0);
            let x1_row = x.row(v1);
            let mut uv: Array1<f64> = &x1_row - &x0_row;
            let norm = uv.iter().map(|&a| a * a).sum::<f64>().sqrt();
            if norm > f64::EPSILON {
                uv.mapv_inplace(|a| a / norm);
            }

            // Scalar projection of velocity onto edge
            let vel_row = v.row(v0);
            let mut sp: f64 = vel_row.iter().zip(uv.iter()).map(|(a, b)| a * b).sum();
            if reverse {
                sp = -sp;
            }
            sp += epsilon;
            sp = sp.max(0.0);

            // Weight = projection / edge_length
            let p = if norm > f64::EPSILON { sp / norm } else { 0.0 };
            edge_probs.push(p);
        }

        // Build sparse matrix and L1-normalise
        let mut tri: TriMat<f64> = TriMat::new((n_cells, n_cells));
        for i in 0..n_edges {
            tri.add_triplet(rows[i], cols[i], edge_probs[i]);
        }
        let csr = tri.to_csr();
        l1_normalize_rows(csr)
    }

    /// Runs a diffusion process on the Markov matrix. Mode `"path_integral"` integrates over time
    /// steps. Mode `"time_evolution"` evolves the distribution. Mode `"trajectory"` samples a
    /// random walk. Also supports `"map_trajectory"` (argmax at each step) and `"frontier"`
    /// (argmax of relative growth).
    ///
    /// Supported modes:
    /// - `"path_integral"` → accumulate intermediate distributions, return Array2
    /// - `"time_evolution"` → return final distribution as Array2
    /// - `"map_trajectory"` → argmax at each step, return Vec<usize>
    /// - `"frontier"` → argmax of relative growth, return Vec<usize>
    /// - `"trajectory"` → random-walk trajectory, return Vec<usize>
    pub fn diffuse(
        &self,
        x_init: &Array2<f64>,
        tr: &CsMat<f64>,
        n_steps: usize,
        mode: &str,
    ) -> DiffuseResult {
        let n_cells = tr.rows();

        match mode {
            "path_integral" => {
                // x = x / x.sum(); accumulate x after each step
                let total = x_init.sum();
                let mut x_flat: Array1<f64> = x_init.clone().into_shape(x_init.len()).unwrap();
                if total > f64::EPSILON {
                    x_flat.mapv_inplace(|v| v / total);
                }

                let mut result: Array1<f64> = Array1::zeros(n_cells);
                for _ in 0..n_steps {
                    x_flat = sprs_matvec_row(tr, &x_flat);
                    result = result + &x_flat;
                }
                DiffuseResult::Matrix(result.into_shape((1, n_cells)).unwrap().into())
            }
            "time_evolution" => {
                let total = x_init.sum();
                let mut x_flat: Array1<f64> = x_init.clone().into_shape(x_init.len()).unwrap();
                if total > f64::EPSILON {
                    x_flat.mapv_inplace(|v| v / total);
                }

                for _ in 0..n_steps {
                    x_flat = sprs_matvec_row(tr, &x_flat);
                }
                DiffuseResult::Matrix(x_flat.into_shape((1, n_cells)).unwrap().into())
            }
            "map_trajectory" => {
                let total = x_init.sum();
                let mut x_flat: Array1<f64> = x_init.clone().into_shape(x_init.len()).unwrap();
                if total > f64::EPSILON {
                    x_flat.mapv_inplace(|v| v / total);
                }

                let mut trajectory = vec![argmax(&x_flat)];
                for _ in 0..n_steps {
                    x_flat = sprs_matvec_row(tr, &x_flat);
                    trajectory.push(argmax(&x_flat));
                }
                DiffuseResult::Trajectory(trajectory)
            }
            "frontier" => {
                let total = x_init.sum();
                let mut x_flat: Array1<f64> = x_init.clone().into_shape(x_init.len()).unwrap();
                if total > f64::EPSILON {
                    x_flat.mapv_inplace(|v| v / total);
                }

                let mut trajectory = vec![argmax(&x_flat)];
                for _ in 0..n_steps {
                    let x_next = sprs_matvec_row(tr, &x_flat);
                    // argmax of (x_next + 1) / (x + 1)
                    let ratio: Vec<f64> = x_next
                        .iter()
                        .zip(x_flat.iter())
                        .map(|(&n, &o)| (n + 1.0) / (o + 1.0))
                        .collect();
                    let ratio = Array1::from(ratio);
                    trajectory.push(argmax(&ratio));
                    x_flat = x_next;
                }
                DiffuseResult::Trajectory(trajectory)
            }
            "trajectory" => {
                // Random-walk: sample node from distribution at each step
                let total = x_init.sum();
                let x_flat: Array1<f64> = x_init.clone().into_shape(x_init.len()).unwrap();
                let x_norm: Array1<f64> = if total > f64::EPSILON {
                    x_flat.mapv(|v| v / total)
                } else {
                    x_flat
                };
                let mut node = weighted_sample(&x_norm);
                let mut trajectory = vec![node];
                for _ in 0..n_steps {
                    // One-hot row vector at `node`
                    let mut x_one: Array1<f64> = Array1::zeros(n_cells);
                    x_one[node] = 1.0;
                    let mut x_next = sprs_matvec_row(tr, &x_one);
                    let s = x_next.sum();
                    if s > f64::EPSILON {
                        x_next.mapv_inplace(|v| v / s);
                    } else {
                        x_next = x_one.clone();
                    }
                    node = weighted_sample(&x_next);
                    trajectory.push(node);
                }
                DiffuseResult::Trajectory(trajectory)
            }
            _ => panic!("diffuse: unknown mode '{mode}'"),
        }
    }
}

/// Return type of `diffuse`: either a dense distribution matrix or a step-by-step trajectory of
/// cell indices.
pub enum DiffuseResult {
    Matrix(Array2<f64>),
    Trajectory(Vec<usize>),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Brute-force KNN: for each row of `query`, find the `k` nearest rows in
/// `data` by Euclidean distance. Returns (distances_flat, indices_flat),
/// both of length query.nrows() * k.
fn knn_search(data: &Array2<f64>, query: &Array2<f64>, k: usize) -> (Vec<f64>, Vec<usize>) {
    let n_data = data.nrows();
    let n_query = query.nrows();
    let k = k.min(n_data);

    let mut dists_out = Vec::with_capacity(n_query * k);
    let mut idx_out = Vec::with_capacity(n_query * k);

    for qi in 0..n_query {
        let qrow = query.row(qi);
        // Compute distances to all data points
        let mut dist_idx: Vec<(f64, usize)> = (0..n_data)
            .map(|di| {
                let drow = data.row(di);
                let d = qrow
                    .iter()
                    .zip(drow.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (d, di)
            })
            .collect();
        dist_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for j in 0..k {
            dists_out.push(dist_idx[j].0);
            idx_out.push(dist_idx[j].1);
        }
    }

    (dists_out, idx_out)
}

/// Left-multiply a row vector by a CSR matrix: result[j] = sum_i x[i] * tr[i,j]
/// This is what Python's `x.dot(tr)` does when x is a row vector.
fn sprs_matvec_row(tr: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
    let n = tr.rows();
    let m = tr.cols();
    let mut out: Array1<f64> = Array1::zeros(m);

    // For each row i: out += x[i] * tr.row(i)
    for (i, row) in tr.outer_iterator().enumerate() {
        if i >= x.len() {
            break;
        }
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        for (&j, &val) in row.indices().iter().zip(row.data().iter()) {
            out[j] += xi * val;
        }
    }
    out
}

/// L1-normalise each row of a CSR matrix in-place (right-stochastic).
fn l1_normalize_rows(mut mat: CsMat<f64>) -> CsMat<f64> {
    let n_rows = mat.rows();
    // Compute row sums
    let mut row_sums = vec![0.0f64; n_rows];
    for (val, (r, _c)) in mat.iter() {
        row_sums[r] += val;
    }
    // Extract owned indptr before taking mutable data reference
    let indptr: Vec<usize> = mat.proper_indptr().into_owned();
    let data = mat.data_mut();
    for r in 0..n_rows {
        let s = row_sums[r];
        if s > f64::EPSILON {
            let start = indptr[r];
            let end = indptr[r + 1];
            for v in &mut data[start..end] {
                *v /= s;
            }
        }
    }
    mat
}

/// Return the index of the maximum value.
fn argmax(arr: &Array1<f64>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Weighted random sample from a probability distribution using a proper PRNG.
fn weighted_sample(probs: &Array1<f64>) -> usize {
    let total: f64 = probs.iter().sum();
    let mut rng = rand::thread_rng();
    let r = rng.gen::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in probs.iter().enumerate() {
        cum += w;
        if r < cum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}
