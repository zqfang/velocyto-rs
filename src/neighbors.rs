//! Translated from velocyto/neighbors.py
//! balance_knn_loop: was @numba.jit → plain Rust (same native speed, no JIT needed)

use linfa_nn::{distance::L2Dist, KdTree, NearestNeighbour};
use ndarray::{s, Array1, Array2, ArrayView2};
use sprs::{CsMat, CsMatBase, TriMat};

// ---------------------------------------------------------------------------
// balance_knn_loop
// ---------------------------------------------------------------------------

/// Balances K-nearest neighbors by iterating over cells and greedily picking neighbors that haven't exceeded their connectivity budget.
///
/// # Arguments
/// * `dsi`  – (samples, K) distance-sorted neighbour indices
/// * `dist` – (samples, K) distances corresponding to `dsi`
/// * `lsi`  – (samples,) indices sorted by ascending connectivity (l)
/// * `maxl` – max in-degree allowed per node
/// * `k`    – desired number of neighbours in the output graph
/// * `return_distance` – whether distances are meaningful
///
/// # Returns
/// `(dsi_new, dist_new, l)` where both `dsi_new` and `dist_new` have
/// shape (samples, k+1); column 0 is the sample itself.
pub fn balance_knn_loop(
    dsi: &Array2<usize>,
    dist: &Array2<f64>,
    lsi: &Array1<usize>,
    maxl: usize,
    k: usize,
    return_distance: bool,
) -> (Array2<usize>, Array2<f64>, Array1<usize>) {
    let n = dsi.shape()[0];
    let sight = dsi.shape()[1];
    assert!(sight >= k, "sight needs to be bigger than k");

    let fill_idx = usize::MAX; // sentinel for "not filled yet" (mirrors Python's -1)
    let mut dsi_new = Array2::<usize>::from_elem((n, k + 1), fill_idx);
    let mut dist_new = Array2::<f64>::zeros((n, k + 1));
    let mut l = Array1::<usize>::zeros(n);

    for i in 0..n {
        let el = lsi[i];
        let mut p: usize = 0;
        let mut j: usize = 0;

        while j < sight {
            if p >= k {
                break;
            }
            let m = dsi[[el, j]];
            if el == m {
                dsi_new[[el, 0]] = el;
                j += 1;
                continue;
            }
            if l[m] >= maxl {
                j += 1;
                continue;
            }
            dsi_new[[el, p + 1]] = m;
            l[m] += 1;
            if return_distance {
                dist_new[[el, p + 1]] = dist[[el, j]];
            }
            p += 1;
            j += 1;
        }

        // If we exhausted sight without finding k neighbours, pad with self
        if j == sight && p < k {
            while p < k {
                dsi_new[[el, p + 1]] = el;
                dist_new[[el, p + 1]] = dist[[el, 0]];
                p += 1;
            }
        }
    }

    if !return_distance {
        // connectivity mode: entire dist_new = 1.0 (mirrors Python's np.ones_like)
        dist_new.fill(1.0);
    }

    (dsi_new, dist_new, l)
}

// ---------------------------------------------------------------------------
// balance_knn_loop_constrained
// ---------------------------------------------------------------------------

/// Like balance_knn_loop but only considers neighbor candidates that are non-zero in the constraint matrix.
///
/// # Additional argument
/// * `groups` – (samples,) integer group label for each node;
///   edges crossing group boundaries are ignored.
pub fn balance_knn_loop_constrained(
    dsi: &Array2<usize>,
    dist: &Array2<f64>,
    lsi: &Array1<usize>,
    groups: &Array1<i64>,
    maxl: usize,
    k: usize,
    return_distance: bool,
) -> (Array2<usize>, Array2<f64>, Array1<usize>) {
    let n = dsi.shape()[0];
    let sight = dsi.shape()[1];
    assert!(sight >= k, "sight needs to be bigger than k");

    let fill_idx = usize::MAX;
    let mut dsi_new = Array2::<usize>::from_elem((n, k + 1), fill_idx);
    let mut dist_new = Array2::<f64>::zeros((n, k + 1));
    let mut l = Array1::<usize>::zeros(n);

    for i in 0..n {
        let el = lsi[i];
        let mut p: usize = 0;
        let mut j: usize = 0;

        while j < sight {
            if p >= k {
                break;
            }
            let m = dsi[[el, j]];
            if el == m {
                dsi_new[[el, 0]] = el;
                j += 1;
                continue;
            }
            if groups[el] != groups[m] {
                j += 1;
                continue;
            }
            if l[m] >= maxl {
                j += 1;
                continue;
            }
            dsi_new[[el, p + 1]] = m;
            l[m] += 1;
            if return_distance {
                dist_new[[el, p + 1]] = dist[[el, j]];
            }
            p += 1;
            j += 1;
        }

        if j == sight && p < k {
            while p < k {
                dsi_new[[el, p + 1]] = el;
                dist_new[[el, p + 1]] = dist[[el, 0]];
                p += 1;
            }
        }
    }

    if !return_distance {
        // connectivity mode: entire dist_new = 1.0 (mirrors Python's np.ones_like)
        dist_new.fill(1.0);
    }

    (dsi_new, dist_new, l)
}

// ---------------------------------------------------------------------------
// knn_balance
// ---------------------------------------------------------------------------

/// Balance a K-NN graph so that no node is the NN to more than `maxl` others.
///
/// # Arguments
/// * `dsi`        – (samples, K) distance-sorted neighbour indices
/// * `dist`       – optional (samples, K) distances; `None` → connectivity mode
/// * `maxl`       – max in-degree
/// * `k`          – desired out-degree
/// * `constraint` – optional group labels (connectivity constrained to same group)
///
/// # Returns
/// `(dist_new, dsi_new, l)`
pub fn knn_balance(
    dsi: &Array2<usize>,
    dist: Option<&Array2<f64>>,
    maxl: usize,
    k: usize,
    constraint: Option<&Array1<i64>>,
) -> (Array2<f64>, Array2<usize>, Array1<usize>) {
    let n = dsi.shape()[0];

    // l = bincount(dsi.flat)  → how many times each index appears
    let mut l_count = vec![0usize; n];
    for &idx in dsi.iter() {
        if idx < n {
            l_count[idx] += 1;
        }
    }
    // lsi = argsort(l)[::-1]  (descending)
    let mut lsi_vec: Vec<usize> = (0..n).collect();
    lsi_vec.sort_by(|&a, &b| l_count[b].cmp(&l_count[a]));
    let lsi = Array1::from_vec(lsi_vec);

    match dist {
        None => {
            // connectivity mode: build a ones matrix with diagonal = 0
            let mut dist_ones = Array2::<f64>::ones((n, dsi.shape()[1]));
            for r in 0..n {
                dist_ones[[r, 0]] = 0.0;
            }
            match constraint {
                Some(g) => {
                    let (dsi_new, dist_new, l) =
                        balance_knn_loop_constrained(dsi, &dist_ones, &lsi, g, maxl, k, false);
                    (dist_new, dsi_new, l)
                }
                None => {
                    let (dsi_new, dist_new, l) =
                        balance_knn_loop(dsi, &dist_ones, &lsi, maxl, k, false);
                    (dist_new, dsi_new, l)
                }
            }
        }
        Some(d) => match constraint {
            Some(g) => {
                let (dsi_new, dist_new, l) =
                    balance_knn_loop_constrained(dsi, d, &lsi, g, maxl, k, true);
                (dist_new, dsi_new, l)
            }
            None => {
                let (dsi_new, dist_new, l) = balance_knn_loop(dsi, d, &lsi, maxl, k, true);
                (dist_new, dsi_new, l)
            }
        },
    }
}

// ---------------------------------------------------------------------------
// knn_distance_matrix
// ---------------------------------------------------------------------------

/// Build a sparse K-NN distance/connectivity matrix using a KD-Tree.
///
/// * `mode = "connectivity"` → values are 1.0
/// * `mode = "distance"`     → values are Euclidean distances
///
/// Returns a CSR matrix of shape (n_samples, n_samples).
pub fn knn_distance_matrix(
    data: ArrayView2<f64>,
    _metric: Option<&str>,
    k: usize,
    mode: &str,
    _n_jobs: i32,
) -> CsMat<f64> {
    let n = data.shape()[0];
    // linfa-nn expects owned Array2; create a view-compatible copy only if needed
    let data_owned = data.to_owned();

    let index = KdTree
        .from_batch(&data_owned, L2Dist)
        .expect("KdTree build failed");

    // We query k+1 neighbours (the point itself is included) then drop self
    let query_k = (k + 1).min(n);

    let mut tri = TriMat::<f64>::new((n, n));

    for row in 0..n {
        let pt = data_owned.row(row);
        let neighbours = index.k_nearest(pt, query_k).expect("k_nearest failed");

        for (_, col) in &neighbours {
            if *col == row {
                continue; // skip self
            }
            let val = if mode == "connectivity" {
                1.0
            } else {
                // recompute Euclidean distance
                let diff = &data_owned.row(row) - &data_owned.row(*col);
                diff.dot(&diff).sqrt()
            };
            tri.add_triplet(row, *col, val);
        }
    }

    tri.to_csr()
}

// ---------------------------------------------------------------------------
// connectivity_to_weights
// ---------------------------------------------------------------------------

/// Normalise a sparse binary connectivity matrix so rows (axis=1) or
/// columns (axis=0) sum to 1.
pub fn connectivity_to_weights(ck: &CsMat<f64>, axis: usize) -> CsMat<f64> {
    let (nrows, ncols) = (ck.rows(), ck.cols());

    if axis == 1 {
        // divide each entry by its row sum
        let mut row_sums = vec![0.0f64; nrows];
        for (&val, (r, _c)) in ck.iter() {
            row_sums[r] += val;
        }
        let mut tri = TriMat::<f64>::new((nrows, ncols));
        for (&val, (r, c)) in ck.iter() {
            let s = row_sums[r];
            if s != 0.0 {
                tri.add_triplet(r, c, val / s);
            }
        }
        tri.to_csr()
    } else {
        // axis == 0: divide each entry by its column sum
        let mut col_sums = vec![0.0f64; ncols];
        for (&val, (_r, c)) in ck.iter() {
            col_sums[c] += val;
        }
        let mut tri = TriMat::<f64>::new((nrows, ncols));
        for (&val, (r, c)) in ck.iter() {
            let s = col_sums[c];
            if s != 0.0 {
                tri.add_triplet(r, c, val / s);
            }
        }
        tri.to_csr()
    }
}

// ---------------------------------------------------------------------------
// convolve_by_sparse_weights
// ---------------------------------------------------------------------------

/// Compute `array @ w.T` where `w` is a CSR sparse weight matrix.
///
/// * `array` – dense (samples, features)
/// * `w`     – sparse weight matrix (samples, samples); columns must sum to 1
///
/// Returns a dense (samples, features) result.
pub fn convolve_by_sparse_weights(array: ArrayView2<f64>, w: &CsMat<f64>) -> Array2<f64> {
    // Python: w_ = w.T; result = data @ w_
    // i.e., result[i, f] = sum_j  array[j, f] * w[i, j]
    // which is: result = w @ array  (treating w rows as output rows)
    let (out_rows, features) = (w.rows(), array.shape()[1]);
    let mut result = Array2::<f64>::zeros((out_rows, features));

    // Iterate CSR rows of w
    for (w_row, row_vec) in w.outer_iterator().enumerate() {
        for (w_col, &w_val) in row_vec.iter() {
            // result[w_row, :] += w_val * array[w_col, :]
            for f in 0..features {
                result[[w_row, f]] += w_val * array[[w_col, f]];
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// make_mutual
// ---------------------------------------------------------------------------

/// Removes edges between neighbours that are not mutual.
///
/// Returns the element-wise minimum of `knn` and `knn.T`, which keeps only
/// entries present in both directions (i.e. mutual connections).
pub fn make_mutual(knn: &CsMat<f64>) -> CsMat<f64> {
    let knn_t = knn.transpose_view().to_csr();
    // element-wise minimum: keep only entries that exist in both knn and knn.T
    let n = knn.rows();
    let mut tri = TriMat::<f64>::new((n, n));
    for ((&v, (r, c)), (&vt, _)) in knn.iter().zip(knn_t.iter()) {
        let m = v.min(vt);
        if m > 0.0 {
            tri.add_triplet(r, c, m);
        }
    }
    // Use outer-iterator approach to correctly compute minimum
    // The zip approach above isn't correct because iterators aren't guaranteed aligned.
    // Redo with a proper sparse minimum.
    let mut result = TriMat::<f64>::new((n, n));
    for (r, row_vec) in knn.outer_iterator().enumerate() {
        for (c, &v) in row_vec.iter() {
            // Check if knn_t has an entry at (r, c), which is knn at (c, r)
            let vt = knn_t.get(r, c).copied().unwrap_or(0.0);
            let m = v.min(vt);
            if m > 0.0 {
                result.add_triplet(r, c, m);
            }
        }
    }
    result.to_csr()
}

// ---------------------------------------------------------------------------
// min_n
// ---------------------------------------------------------------------------

/// Find the `n` smallest values and their corresponding indices in `row_data`.
///
/// Returns `(top_values, top_indices)` sorted by ascending value.
pub fn min_n(row_data: &[f64], row_indices: &[usize], n: usize) -> (Vec<f64>, Vec<usize>) {
    let n = n.min(row_data.len());
    let mut order: Vec<usize> = (0..row_data.len()).collect();
    order.sort_by(|&a, &b| row_data[a].partial_cmp(&row_data[b]).unwrap_or(std::cmp::Ordering::Equal));
    order.truncate(n);
    let top_values: Vec<f64> = order.iter().map(|&i| row_data[i]).collect();
    let top_indices: Vec<usize> = order.iter().map(|&i| row_indices[i]).collect();
    (top_values, top_indices)
}

// ---------------------------------------------------------------------------
// take_top
// ---------------------------------------------------------------------------

/// Filter the top `n` nearest neighbours from a sparse distance matrix.
///
/// For each row keeps only the `n` smallest non-zero entries, discarding the rest.
/// Returns a new CSR sparse matrix of shape matching the input.
pub fn take_top(matrix: &CsMat<f64>, n: usize) -> CsMat<f64> {
    let nrows = matrix.rows();
    let ncols = matrix.cols();
    let mut result = TriMat::<f64>::new((nrows, ncols));
    for (r, row_vec) in matrix.outer_iterator().enumerate() {
        let data: Vec<f64> = row_vec.iter().map(|(_, &v)| v).collect();
        let indices: Vec<usize> = row_vec.iter().map(|(c, _)| c).collect();
        let (top_vals, top_idx) = min_n(&data, &indices, n);
        for (v, c) in top_vals.iter().zip(top_idx.iter()) {
            result.add_triplet(r, *c, *v);
        }
    }
    result.to_csr()
}

// ---------------------------------------------------------------------------
// knn_smooth_weights
// ---------------------------------------------------------------------------

/// Find the weights to smooth the dataset using efficient sparse matrix operations.
///
/// # Arguments
/// * `matrix`   – (genes, cells) expression matrix; transposed to (cells, genes) for KNN
/// * `metric`   – distance metric (passed through to `knn_distance_matrix`)
/// * `k_search` – first k nearest-neighbour search number of neighbours
/// * `k_mutual` – number of mutual neighbours to select
/// * `n_jobs`   – parallelism hint (passed through)
///
/// # Returns
/// `(weights, knn)` where `weights` is a normalised sparse weight matrix and
/// `knn` is the raw distance matrix from the initial search.
pub fn knn_smooth_weights(
    matrix: &Array2<f64>,
    metric: &str,
    k_search: usize,
    k_mutual: usize,
    n_jobs: i32,
) -> (CsMat<f64>, CsMat<f64>) {
    assert!(k_search >= k_mutual, "k_search needs to be bigger than k_mutual");
    // Python: matrix.T is (cells, genes)
    let matrix_t = matrix.t().to_owned();
    let knn = knn_distance_matrix(matrix_t.view(), Some(metric), k_search, "distance", n_jobs);
    let mknn = make_mutual(&knn);
    let mut top_mknn = take_top(&mknn, k_mutual);
    // setdiag(1)
    let n = top_mknn.rows();
    let mut tri = TriMat::<f64>::new((n, n));
    for (r, row_vec) in top_mknn.outer_iterator().enumerate() {
        for (c, &v) in row_vec.iter() {
            tri.add_triplet(r, c, v);
        }
    }
    for i in 0..n {
        tri.add_triplet(i, i, 1.0);
    }
    top_mknn = tri.to_csr();
    // connectivity: binarise
    let mut conn_tri = TriMat::<f64>::new((n, n));
    for (&v, (r, c)) in top_mknn.iter() {
        if v > 0.0 {
            conn_tri.add_triplet(r, c, 1.0);
        }
    }
    let connectivity = conn_tri.to_csr();
    let w = connectivity_to_weights(&connectivity, 1);
    (w, knn)
}

// ---------------------------------------------------------------------------
// BalancedKNN struct
// ---------------------------------------------------------------------------

/// Greedy algorithm to balance a K-nearest-neighbour graph.
///
/// API is similar to scikit-learn estimators.
pub struct BalancedKNN {
    pub k: usize,
    pub maxl: usize,
    pub metric: String,
    pub sight_k: usize,
    pub n_jobs: i32,
    pub mode: String,
    pub constraint: Option<Array1<i64>>,

    // fit state
    data: Option<Array2<f64>>,
    /// Raw KNN distances (n, sight_k+1)
    pub dist: Option<Array2<f64>>,
    /// Raw KNN indices  (n, sight_k+1)
    pub dsi: Option<Array2<usize>>,

    // predict state
    pub dist_new: Option<Array2<f64>>,
    pub dsi_new: Option<Array2<usize>>,
    pub l: Option<Array1<usize>>,
    pub bknn: Option<CsMat<f64>>,
}

impl BalancedKNN {
    /// Creates a BalancedKNN with `k` neighbors per cell, looking up to `sight_k` candidates. `maxl` is the maximum connectivity for trimming. `mode` is 'connectivity' or 'distance'. `metric` is the distance metric.
    pub fn new(
        k: usize,
        maxl: usize,
        metric: &str,
        sight_k: usize,
        n_jobs: i32,
        mode: &str,
    ) -> Self {
        BalancedKNN {
            k,
            maxl,
            metric: metric.to_string(),
            sight_k,
            n_jobs,
            mode: mode.to_string(),
            constraint: None,
            data: None,
            dist: None,
            dsi: None,
            dist_new: None,
            dsi_new: None,
            l: None,
            bknn: None,
        }
    }

    /// Sets an optional constraint matrix for the KNN balancing.
    pub fn with_constraint(mut self, constraint: Array1<i64>) -> Self {
        self.constraint = Some(constraint);
        self
    }

    /// Number of samples in the fitted data.
    pub fn n_samples(&self) -> usize {
        self.data.as_ref().map(|d| d.shape()[0]).unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // fit
    // -----------------------------------------------------------------------

    /// Build the initial (over-sampled) KNN index with `sight_k` neighbours.
    pub fn fit(&mut self, data: ArrayView2<f64>) -> &mut Self {
        let n = data.shape()[0];
        let query_k = (self.sight_k + 1).min(n); // +1 to include self then strip

        let data_owned = data.to_owned();
        let data_for_index = data_owned.clone();
        let index = KdTree
            .from_batch(&data_for_index, L2Dist)
            .expect("KdTree build failed");

        let mut dist_mat = Array2::<f64>::zeros((n, query_k));
        let mut dsi_mat = Array2::<usize>::zeros((n, query_k));

        for row in 0..n {
            let pt = data_owned.row(row);
            let neighbours = index.k_nearest(pt, query_k).expect("k_nearest failed");

            // Sort by distance (linfa-nn returns unsorted for KdTree sometimes)
            let mut nb_sorted: Vec<(f64, usize)> = neighbours
                .iter()
                .map(|(pt_ref, idx)| {
                    let diff = pt_ref.to_owned() - &data_owned.row(row);
                    let d = diff.dot(&diff).sqrt();
                    (d, *idx)
                })
                .collect();
            nb_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (col, (d, idx)) in nb_sorted.iter().enumerate() {
                dist_mat[[row, col]] = *d;
                dsi_mat[[row, col]] = *idx;
            }
        }

        self.data = Some(data_owned);
        self.dist = Some(dist_mat);
        self.dsi = Some(dsi_mat);
        self
    }

    // -----------------------------------------------------------------------
    // kneighbors — raw KNN query (mirrors Python kneighbors)
    // -----------------------------------------------------------------------

    /// Returns (distances, indices) arrays of the raw sight_k nearest neighbours.
    pub fn kneighbors(&self) -> (&Array2<f64>, &Array2<usize>) {
        (
            self.dist.as_ref().expect("call fit() first"),
            self.dsi.as_ref().expect("call fit() first"),
        )
    }

    // -----------------------------------------------------------------------
    // predict — run balance and return sparse weight matrix
    // -----------------------------------------------------------------------

    /// Balance the KNN graph and return a sparse (n, n) weight matrix.
    pub fn predict(&mut self, maxl: Option<usize>) -> CsMat<f64> {
        if let Some(ml) = maxl {
            self.maxl = ml;
        }
        let dsi = self.dsi.as_ref().expect("call fit() first").clone();
        let dist = self.dist.as_ref().expect("call fit() first").clone();

        let use_distance = self.mode == "distance";

        let (dist_new, dsi_new, l) = knn_balance(
            &dsi,
            if use_distance { Some(&dist) } else { None },
            self.maxl,
            self.k,
            self.constraint.as_ref(),
        );

        self.dist_new = Some(dist_new.clone());
        self.dsi_new = Some(dsi_new.clone());
        self.l = Some(l);

        let bknn = self.build_sparse(&dist_new, &dsi_new);
        self.bknn = Some(bknn.clone());
        bknn
    }

    // -----------------------------------------------------------------------
    // fit_predict
    // -----------------------------------------------------------------------

    /// Runs fit() then predict() in sequence.
    pub fn fit_predict(&mut self, data: ArrayView2<f64>, maxl: Option<usize>) -> CsMat<f64> {
        self.fit(data);
        self.predict(maxl)
    }

    // -----------------------------------------------------------------------
    // transform — return (dist, dsi) after balancing
    // -----------------------------------------------------------------------

    /// Returns the balanced (dist_new, dsi_new) after KNN balancing.
    pub fn transform(&self) -> (&Array2<f64>, &Array2<usize>) {
        (
            self.dist_new.as_ref().expect("call predict() first"),
            self.dsi_new.as_ref().expect("call predict() first"),
        )
    }

    // -----------------------------------------------------------------------
    // kneighbors_graph — sparse (n, n) CSR matrix
    // -----------------------------------------------------------------------

    /// Build and return a sparse (n_samples, n_samples) CSR matrix.
    /// Column j of row i holds the distance/connectivity from i to its j-th balanced neighbour.
    pub fn kneighbors_graph(&mut self, maxl: Option<usize>) -> CsMat<f64> {
        self.predict(maxl)
    }

    // -----------------------------------------------------------------------
    // smooth_data
    // -----------------------------------------------------------------------

    /// Use the weights learned from KNN to smooth any data matrix.
    ///
    /// # Arguments
    /// * `data_to_smooth` – (features, samples) — NOTE: transposed relative to the fit data.
    ///   If the data is provided as (samples, features) this is detected automatically when the
    ///   two axes differ; when they are equal (features, samples) is assumed.
    /// * `mutual`        – if true, make the KNN graph mutual before smoothing
    /// * `only_increase` – if true, return `max(result, data_to_smooth)` element-wise
    ///
    /// Returns a smoothed dense matrix with the same shape as `data_to_smooth`.
    pub fn smooth_data(
        &mut self,
        data_to_smooth: &Array2<f64>,
        mutual: bool,
        only_increase: bool,
    ) -> Array2<f64> {
        if self.bknn.is_none() {
            self.kneighbors_graph(None);
        }
        let bknn = self.bknn.as_ref().expect("bknn must be set after kneighbors_graph");

        // connectivity = (bknn.T > 0) or make_mutual(bknn > 0)
        // binarise bknn first
        let n = bknn.rows();
        let mut bknn_bin_tri = TriMat::<f64>::new((n, n));
        for (&v, (r, c)) in bknn.iter() {
            if v > 0.0 {
                bknn_bin_tri.add_triplet(r, c, 1.0);
            }
        }
        let bknn_bin: CsMat<f64> = bknn_bin_tri.to_csr();

        let connectivity_base: CsMat<f64> = if mutual {
            make_mutual(&bknn_bin)
        } else {
            // bknn.T > 0
            let bknn_t = bknn_bin.transpose_view().to_csr();
            bknn_t
        };

        // setdiag(1)
        let mut tri = TriMat::<f64>::new((n, n));
        for (&v, (r, c)) in connectivity_base.iter() {
            tri.add_triplet(r, c, v);
        }
        for i in 0..n {
            tri.add_triplet(i, i, 1.0);
        }
        let connectivity_with_diag = tri.to_csr();

        // w = connectivity_to_weights(connectivity).T  (axis=1 then transpose)
        let w_before_t = connectivity_to_weights(&connectivity_with_diag, 1);
        // Transpose w
        let w = w_before_t.transpose_view().to_csr();

        // Python asserts: w.sum(0) ≈ 1  (columns of w sum to 1 after transpose)
        // result = data_to_smooth @ w
        // Shape matching: data_to_smooth is (features, samples) → w is (samples, samples)
        // If data_to_smooth.shape[1] == w.shape[0] → (features, samples) @ (samples, samples)
        // If data_to_smooth.shape[0] == w.shape[0] → (samples, features).T @ w then .T
        let (nrows, ncols) = (data_to_smooth.shape()[0], data_to_smooth.shape()[1]);
        let result = if ncols == w.rows() {
            // (features, samples) @ (samples, samples) → (features, samples)
            let mut out = Array2::<f64>::zeros((nrows, w.cols()));
            for f in 0..nrows {
                for (w_col, w_row_vec) in w.outer_iterator().enumerate() {
                    let mut val = 0.0f64;
                    for (w_c, &w_v) in w_row_vec.iter() {
                        val += data_to_smooth[[f, w_c]] * w_v;
                    }
                    out[[f, w_col]] = val;
                }
            }
            out
        } else if nrows == w.rows() {
            // (samples, features) case: compute (data.T @ w).T
            let mut out_t = Array2::<f64>::zeros((w.cols(), ncols));
            for (w_col, w_row_vec) in w.outer_iterator().enumerate() {
                for (w_c, &w_v) in w_row_vec.iter() {
                    for f in 0..ncols {
                        out_t[[w_col, f]] += data_to_smooth[[w_c, f]] * w_v;
                    }
                }
            }
            out_t.t().to_owned()
        } else {
            panic!(
                "Incorrect size of matrix, none of the axes correspond to the graph. w.shape=({}, {}), data.shape=({}, {})",
                w.rows(), w.cols(), nrows, ncols
            );
        };

        if only_increase {
            // element-wise maximum of result and data_to_smooth
            let mut out = result.clone();
            for ((r, c), v) in out.indexed_iter_mut() {
                *v = v.max(data_to_smooth[[r, c]]);
            }
            out
        } else {
            result
        }
    }

    // -----------------------------------------------------------------------
    // internal helper
    // -----------------------------------------------------------------------

    fn build_sparse(&self, dist_new: &Array2<f64>, dsi_new: &Array2<usize>) -> CsMat<f64> {
        let n = self.n_samples();
        let cols = dist_new.shape()[1]; // k+1

        // Mirror Python:
        //   sparse.csr_matrix(
        //       (np.ravel(dist_new), np.ravel(dsi_new),
        //        np.arange(0, n_rows * n_cols + 1, n_cols)),
        //       shape=(n, n))
        // This is a CSR constructed with indptr spaced by `cols`.
        let indptr: Vec<usize> = (0..=n).map(|i| i * cols).collect();
        let indices: Vec<usize> = dsi_new
            .iter()
            .map(|&v| if v == usize::MAX { 0 } else { v })
            .collect();
        let data: Vec<f64> = dist_new.iter().copied().collect();

        CsMatBase::new((n, n), indptr, indices, data)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ------------------------------------------------------------------
    // balance_knn_loop tests
    // ------------------------------------------------------------------

    #[test]
    fn test_balance_knn_loop_basic() {
        // 4 nodes, sight = 3, k = 2, maxl = 2
        // dsi: each row lists the 3 nearest neighbours (including self)
        // row 0: [0, 1, 2]  → nearest are 1, 2
        // row 1: [1, 0, 2]
        // row 2: [2, 0, 1]
        // row 3: [3, 0, 1]
        let dsi = array![[0usize, 1, 2], [1, 0, 2], [2, 0, 1], [3, 0, 1],];
        let dist = array![
            [0.0f64, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ];

        // lsi: process in descending connectivity order — just use identity
        let lsi = array![0usize, 1, 2, 3];

        let (dsi_new, dist_new, l) = balance_knn_loop(&dsi, &dist, &lsi, 2, 2, true);

        // Output shape should be (4, 3)  i.e. k+1
        assert_eq!(dsi_new.shape(), &[4, 3]);
        assert_eq!(dist_new.shape(), &[4, 3]);
        assert_eq!(l.len(), 4);

        // Column 0 is self for each row
        assert_eq!(dsi_new[[0, 0]], 0);
        assert_eq!(dsi_new[[1, 0]], 1);

        // l counts should be non-negative and ≤ maxl
        for &v in l.iter() {
            assert!(v <= 2, "l value exceeds maxl: {v}");
        }
    }

    #[test]
    fn test_balance_knn_loop_maxl_respected() {
        // 3 nodes all wanting to connect to node 0, but maxl = 1
        // dsi: rows are [self, 0, other]
        let dsi = array![[0usize, 1, 2], [1, 0, 2], [2, 0, 1],];
        let dist = array![[0.0f64, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],];
        // Process node 1, then 2, then 0
        let lsi = array![1usize, 2, 0];

        let (dsi_new, _dist_new, l) =
            balance_knn_loop(&dsi, &dist, &lsi, /*maxl=*/ 1, /*k=*/ 2, true);

        // Node 0 can only be selected as neighbour at most once (maxl=1)
        assert!(l[0] <= 1, "node 0 in-degree exceeds maxl=1: {}", l[0]);
        let _ = dsi_new; // suppress unused warning
    }

    // ------------------------------------------------------------------
    // convolve_by_sparse_weights tests
    // ------------------------------------------------------------------

    #[test]
    fn test_convolve_identity_weights() {
        // 3 samples, 2 features
        let array = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]];

        // Weight matrix = identity (each sample is its own only neighbour)
        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        tri.add_triplet(2, 2, 1.0);
        let w: CsMat<f64> = tri.to_csr();

        let result = convolve_by_sparse_weights(array.view(), &w);
        // With identity weights, result == array
        for r in 0..3 {
            for c in 0..2 {
                assert!(
                    (result[[r, c]] - array[[r, c]]).abs() < 1e-12,
                    "mismatch at [{r},{c}]: {} vs {}",
                    result[[r, c]],
                    array[[r, c]]
                );
            }
        }
    }

    #[test]
    fn test_convolve_averaging_weights() {
        // 3 samples, 2 features; w averages neighbour 0 and 1 for sample 2
        let array = array![[2.0f64, 4.0], [6.0, 8.0], [0.0, 0.0]];

        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        // row 2: 0.5 * sample0 + 0.5 * sample1
        tri.add_triplet(2, 0, 0.5);
        tri.add_triplet(2, 1, 0.5);
        let w: CsMat<f64> = tri.to_csr();

        let result = convolve_by_sparse_weights(array.view(), &w);
        // row 2 of result = 0.5*[2,4] + 0.5*[6,8] = [4, 6]
        assert!(
            (result[[2, 0]] - 4.0).abs() < 1e-12,
            "expected 4.0, got {}",
            result[[2, 0]]
        );
        assert!(
            (result[[2, 1]] - 6.0).abs() < 1e-12,
            "expected 6.0, got {}",
            result[[2, 1]]
        );
    }

    // ------------------------------------------------------------------
    // connectivity_to_weights test
    // ------------------------------------------------------------------

    #[test]
    fn test_connectivity_to_weights_rows() {
        // 2x2 ones matrix; row sums = 2 so each weight = 0.5
        let mut tri = TriMat::<f64>::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        let ck: CsMat<f64> = tri.to_csr();

        let w = connectivity_to_weights(&ck, 1);
        for (&val, _) in w.iter() {
            assert!((val - 0.5).abs() < 1e-12, "expected 0.5, got {val}");
        }
    }
}
