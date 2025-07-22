import numba
import numpy as np
import time
from typing import Tuple, List, Dict, Union
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import igraph as ig
from joblib import Parallel, delayed
from src.ldpc_post_selection.cluster_tools import label_clusters_igraph


@numba.njit(fastmath=True, cache=True, parallel=True)
def _numba_window_norm_fracs_kernel(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    num_samples: int,
    num_faults: int,
    num_windows: int,
    norm_order: float,
    value_type_int: int,  # 0 = size, 1 = llr
    aggregation_type_int: int,  # 0 = avg, 1 = max
    bit_llrs: np.ndarray,  # Pre-computed LLR values
    total_llr_sum: float,  # Pre-computed total LLR sum
    start_window: int,
    end_window: int,
) -> np.ndarray:
    """
    Calculate window cluster norm fractions using optimized numba kernel.

    Parameters
    ----------
    data : 1D numpy array
        CSR matrix data array containing cluster IDs.
    indices : 1D numpy array
        CSR matrix indices array.
    indptr : 1D numpy array
        CSR matrix indptr array.
    num_samples : int
        Number of samples.
    num_faults : int
        Number of faults per window.
    num_windows : int
        Total number of windows.
    norm_order : float
        Order for L_p norm calculation.
    value_type_int : int
        0 for size calculations, 1 for LLR calculations.
    aggregation_type_int : int
        0 for average, 1 for maximum.
    bit_llrs : 1D numpy array
        Pre-computed LLR values for each fault.
    total_llr_sum : float
        Pre-computed total LLR sum.
    start_window : int
        Starting window index.
    end_window : int
        Ending window index (exclusive).

    Returns
    -------
    norm_fractions : 1D numpy array
        Aggregated norm fractions for each sample.
    """
    if end_window <= start_window:
        return np.zeros(num_samples, dtype=np.float64)

    # Initialize result array
    norm_fractions = np.zeros(num_samples, dtype=np.float64)

    # Process each sample in parallel
    for sample_idx in numba.prange(num_samples):
        sample_start = indptr[sample_idx]
        sample_end = indptr[sample_idx + 1]

        # Skip empty samples
        if sample_end <= sample_start:
            continue

        # Collect norm fractions for evaluated windows
        window_fracs = np.zeros(end_window - start_window, dtype=np.float64)
        window_count = 0

        for window_idx in range(start_window, end_window):
            window_start_col = window_idx * num_faults
            window_end_col = (window_idx + 1) * num_faults

            # Find max cluster ID for this window
            max_cluster_id = 0
            for j in range(sample_start, sample_end):
                col_idx = indices[j]
                if window_start_col <= col_idx < window_end_col:
                    cluster_id = int(data[j])
                    if cluster_id > max_cluster_id:
                        max_cluster_id = cluster_id

            if max_cluster_id == 0:
                window_fracs[window_count] = 0.0
            else:
                # Compute cluster values
                cluster_values = np.zeros(max_cluster_id + 1, dtype=np.float64)

                for j in range(sample_start, sample_end):
                    col_idx = indices[j]
                    if window_start_col <= col_idx < window_end_col:
                        cluster_id = int(data[j])
                        if cluster_id > 0:
                            if value_type_int == 1:  # LLR
                                fault_idx = col_idx - window_start_col
                                cluster_values[cluster_id] += bit_llrs[fault_idx]
                            else:  # Size
                                cluster_values[cluster_id] += 1.0

                # Calculate norm
                norm_val = 0.0
                if norm_order == 1.0:
                    for cid in range(1, max_cluster_id + 1):
                        norm_val += abs(cluster_values[cid])
                elif norm_order == 2.0:
                    sum_sq = 0.0
                    for cid in range(1, max_cluster_id + 1):
                        v = cluster_values[cid]
                        sum_sq += v * v
                    norm_val = np.sqrt(sum_sq)
                elif np.isinf(norm_order):
                    for cid in range(1, max_cluster_id + 1):
                        v = abs(cluster_values[cid])
                        if v > norm_val:
                            norm_val = v
                else:
                    sum_pow = 0.0
                    for cid in range(1, max_cluster_id + 1):
                        v = abs(cluster_values[cid])
                        if v > 0:
                            sum_pow += v**norm_order
                    norm_val = sum_pow ** (1.0 / norm_order)

                # Calculate fraction
                if value_type_int == 1:  # LLR
                    window_fracs[window_count] = (
                        norm_val / total_llr_sum if total_llr_sum > 0 else 0.0
                    )
                else:  # Size
                    window_fracs[window_count] = norm_val / num_faults

            window_count += 1

        # Aggregate across windows
        if window_count > 0:
            if aggregation_type_int == 0:  # Average
                sum_fracs = 0.0
                for i in range(window_count):
                    sum_fracs += window_fracs[i]
                norm_fractions[sample_idx] = sum_fracs / window_count
            else:  # Maximum
                max_frac = 0.0
                for i in range(window_count):
                    if window_fracs[i] > max_frac:
                        max_frac = window_fracs[i]
                norm_fractions[sample_idx] = max_frac

    return norm_fractions


@numba.njit(fastmath=True, cache=True)
def _calculate_size_norm_fraction_numba(
    cluster_labels: np.ndarray, norm_order: float
) -> float:
    """
    Calculate norm fraction for cluster sizes using numba for maximum performance.

    Parameters
    ----------
    cluster_labels : 1D numpy array of int
        Cluster labels for committed faults (all values > 0).
    norm_order : float
        Order for L_p norm calculation.

    Returns
    -------
    norm_value : float
        Norm of cluster sizes.
    """
    if len(cluster_labels) == 0:
        return 0.0

    # Find maximum cluster ID to determine array size
    max_cluster_id = 0
    for i in range(len(cluster_labels)):
        if cluster_labels[i] > max_cluster_id:
            max_cluster_id = cluster_labels[i]

    if max_cluster_id == 0:
        return 0.0

    # Count cluster sizes manually
    cluster_sizes = np.zeros(max_cluster_id + 1, dtype=np.float64)
    for i in range(len(cluster_labels)):
        cluster_id = cluster_labels[i]
        if cluster_id > 0:
            cluster_sizes[cluster_id] += 1.0

    # Calculate norm of cluster sizes
    norm_val = 0.0
    if norm_order == 1.0:
        for cid in range(1, max_cluster_id + 1):
            norm_val += cluster_sizes[cid]
    elif norm_order == 2.0:
        sum_sq = 0.0
        for cid in range(1, max_cluster_id + 1):
            v = cluster_sizes[cid]
            if v > 0:
                sum_sq += v * v
        norm_val = np.sqrt(sum_sq)
    elif np.isinf(norm_order):
        for cid in range(1, max_cluster_id + 1):
            v = cluster_sizes[cid]
            if v > norm_val:
                norm_val = v
    else:
        sum_pow = 0.0
        for cid in range(1, max_cluster_id + 1):
            v = cluster_sizes[cid]
            if v > 0:
                sum_pow += v**norm_order
        norm_val = sum_pow ** (1.0 / norm_order)

    return norm_val


@numba.njit(fastmath=True, cache=True)
def _calculate_llr_norm_fraction_numba(
    cluster_labels: np.ndarray, llr_values: np.ndarray, norm_order: float
) -> float:
    """
    Calculate norm fraction for cluster LLR sums using numba for maximum performance.

    Parameters
    ----------
    cluster_labels : 1D numpy array of int
        Cluster labels for committed faults (all values > 0).
    llr_values : 1D numpy array of float
        LLR values corresponding to each committed fault.
    norm_order : float
        Order for L_p norm calculation.

    Returns
    -------
    norm_value : float
        Norm of cluster LLR sums.
    """
    if len(cluster_labels) == 0 or len(llr_values) == 0:
        return 0.0

    if len(cluster_labels) != len(llr_values):
        return 0.0  # Invalid input

    # Find maximum cluster ID to determine array size
    max_cluster_id = 0
    for i in range(len(cluster_labels)):
        if cluster_labels[i] > max_cluster_id:
            max_cluster_id = cluster_labels[i]

    if max_cluster_id == 0:
        return 0.0

    # Sum LLR values by cluster manually
    cluster_llr_sums = np.zeros(max_cluster_id + 1, dtype=np.float64)
    for i in range(len(cluster_labels)):
        cluster_id = cluster_labels[i]
        if cluster_id > 0:
            cluster_llr_sums[cluster_id] += llr_values[i]

    # Calculate norm of cluster LLR sums
    norm_val = 0.0
    if norm_order == 1.0:
        for cid in range(1, max_cluster_id + 1):
            norm_val += abs(cluster_llr_sums[cid])
    elif norm_order == 2.0:
        sum_sq = 0.0
        for cid in range(1, max_cluster_id + 1):
            v = cluster_llr_sums[cid]
            sum_sq += v * v
        norm_val = np.sqrt(sum_sq)
    elif np.isinf(norm_order):
        for cid in range(1, max_cluster_id + 1):
            v = abs(cluster_llr_sums[cid])
            if v > norm_val:
                norm_val = v
    else:
        sum_pow = 0.0
        for cid in range(1, max_cluster_id + 1):
            v = abs(cluster_llr_sums[cid])
            if v > 0:
                sum_pow += v**norm_order
        norm_val = sum_pow ** (1.0 / norm_order)

    return norm_val


def calculate_window_cluster_norm_fracs_from_csr(
    all_clusters_csr: csr_matrix,
    priors: np.ndarray,
    norm_order: float,
    value_type: str,
    aggregation_type: str,
    eval_windows: Tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Optimized calculation of window cluster norm fracs from CSR matrix.

    Parameters
    ----------
    all_clusters_csr : scipy sparse CSR matrix
        CSR matrix of integers where columns represent "i*num_faults + j".
    priors : 1D numpy array of float
        Prior probabilities for each fault (determines num_faults).
    norm_order : float
        Order for L_p norm calculation.
    value_type : str
        Type of values to calculate: "size" or "llr".
    aggregation_type : str
        Type of aggregation across windows: "avg" or "max".
    eval_windows : tuple of int, optional
        If provided, only consider windows from init_eval_window to final_eval_window.

    Returns
    -------
    norm_fractions : 1D numpy array of float
        Norm fractions for each sample after aggregation across windows.
    """
    num_faults = len(priors)
    num_samples = all_clusters_csr.shape[0]
    total_cols = all_clusters_csr.shape[1]
    num_windows = total_cols // num_faults

    if total_cols % num_faults != 0:
        raise ValueError(
            f"Total columns ({total_cols}) is not divisible by num_faults ({num_faults})"
        )

    # Convert parameters to integers for numba
    value_type_int = {"size": 0, "llr": 1}[value_type]
    aggregation_type_int = {"avg": 0, "max": 1}[aggregation_type]

    # Determine window range
    if eval_windows is not None:
        start_window = max(0, eval_windows[0])
        end_window = min(num_windows, eval_windows[1] + 1)
    else:
        start_window = 0
        end_window = num_windows

    # Pre-compute LLR values if needed
    if value_type == "llr":
        bit_llrs = np.log((1 - priors) / priors)
        total_llr_sum = bit_llrs.sum()
    else:
        bit_llrs = np.zeros(num_faults, dtype=np.float64)
        total_llr_sum = 0.0

    # Use numba kernel for optimized calculation
    return _numba_window_norm_fracs_kernel(
        all_clusters_csr.data.astype(np.float64),
        all_clusters_csr.indices.astype(np.int32),
        all_clusters_csr.indptr.astype(np.int32),
        num_samples,
        num_faults,
        num_windows,
        norm_order,
        value_type_int,
        aggregation_type_int,
        bit_llrs,
        total_llr_sum,
        start_window,
        end_window,
    )


def _split_csr_by_windows(
    csr_matrix: csr_matrix, num_faults: int, eval_windows: Tuple[int, int] | None = None
) -> List[csr_matrix]:
    """
    Split a CSR matrix by windows where columns represent "i*num_faults + j".

    Parameters
    ----------
    csr_matrix : scipy sparse CSR matrix
        Matrix where columns represent "i*num_faults + j" (j-th fault of i-th window).
    num_faults : int
        Number of faults per window.
    eval_windows : tuple of int, optional
        If provided, only return windows from init_eval_window to final_eval_window.

    Returns
    -------
    window_matrices : list of scipy sparse CSR matrix
        List of CSR matrices, one for each window.
    """
    num_samples, total_cols = csr_matrix.shape
    num_windows = total_cols // num_faults

    if total_cols % num_faults != 0:
        raise ValueError(
            f"Total columns ({total_cols}) is not divisible by num_faults ({num_faults})"
        )

    # Determine which windows to extract
    if eval_windows is not None:
        start_window, end_window = eval_windows
        start_window = max(0, start_window)
        end_window = min(num_windows, end_window + 1)  # +1 for inclusive end
    else:
        start_window, end_window = 0, num_windows

    window_matrices = []
    for window_idx in range(start_window, end_window):
        start_col = window_idx * num_faults
        end_col = (window_idx + 1) * num_faults
        window_matrix = csr_matrix[:, start_col:end_col]
        window_matrices.append(window_matrix)

    return window_matrices


def _compute_logical_or_across_windows(window_matrices: List[csr_matrix]) -> csr_matrix:
    """
    Compute logical OR across multiple boolean CSR matrices.

    Parameters
    ----------
    window_matrices : list of scipy sparse CSR matrix
        List of boolean CSR matrices to combine with logical OR.

    Returns
    -------
    combined_matrix : scipy sparse CSR matrix
        Boolean CSR matrix representing logical OR of all input matrices.
    """
    if not window_matrices:
        return csr_matrix((0, 0), dtype=bool)

    if len(window_matrices) == 1:
        return window_matrices[0].astype(bool)

    # Start with the first matrix
    combined = window_matrices[0].astype(bool)

    # Combine with remaining matrices using logical OR
    for matrix in window_matrices[1:]:
        combined = combined + matrix.astype(bool)
        # Convert back to boolean (any non-zero becomes True)
        combined.data = combined.data > 0

    return combined


def _process_sample_batch(
    sample_indices: List[int],
    combined_committed_data: np.ndarray,
    combined_committed_indices: np.ndarray,
    combined_committed_indptr: np.ndarray,
    full_graph: ig.Graph,
    value_type: str,
    norm_order: float,
    bit_llrs: np.ndarray | None,
    total_llr_sum: float,
    total_committed_faults: int,
) -> np.ndarray:
    """
    Process a batch of samples for committed cluster norm fraction calculation.

    Parameters
    ----------
    sample_indices : list of int
        Indices of samples to process in this batch.
    combined_committed_data : 1D numpy array
        CSR matrix data array.
    combined_committed_indices : 1D numpy array
        CSR matrix indices array.
    combined_committed_indptr : 1D numpy array
        CSR matrix indptr array.
    full_graph : igraph Graph
        Pre-constructed graph for cluster labeling.
    value_type : str
        Type of values to calculate: "size" or "llr".
    norm_order : float
        Order for L_p norm calculation.
    bit_llrs : 1D numpy array or None
        LLR values for each fault (None for size calculations).
    total_llr_sum : float
        Total sum of LLR values.
    total_committed_faults : int
        Total number of committed faults.

    Returns
    -------
    batch_norm_fractions : 1D numpy array of float
        Norm fractions for the processed samples.
    """

    batch_norm_fractions = np.zeros(len(sample_indices), dtype=float)

    for batch_idx, sample_idx in enumerate(sample_indices):
        # Find indices of committed faults for this sample directly from sparse matrix
        row_start = combined_committed_indptr[sample_idx]
        row_end = combined_committed_indptr[sample_idx + 1]

        committed_cluster_fault_indices = combined_committed_indices[row_start:row_end]

        if len(committed_cluster_fault_indices) == 0:
            # No committed clusters, norm fraction is 0
            batch_norm_fractions[batch_idx] = 0.0
            continue

        # Label clusters using the adjacency matrix
        cluster_labels = label_clusters_igraph(
            full_graph, committed_cluster_fault_indices
        )

        # Extract cluster labels for committed faults only
        nonzero_cluster_labels = cluster_labels[committed_cluster_fault_indices]

        # Calculate norm fractions based on cluster labels
        if value_type == "size":
            # Use optimized numba kernel for size norm calculation
            inside_norm = _calculate_size_norm_fraction_numba(
                nonzero_cluster_labels, norm_order
            )
            batch_norm_fractions[batch_idx] = (
                inside_norm / total_committed_faults
                if total_committed_faults > 0
                else 0.0
            )

        elif value_type == "llr":
            # Use optimized numba kernel for LLR norm calculation
            llr_values_for_committed = bit_llrs[committed_cluster_fault_indices]
            inside_norm = _calculate_llr_norm_fraction_numba(
                nonzero_cluster_labels, llr_values_for_committed, norm_order
            )
            batch_norm_fractions[batch_idx] = (
                inside_norm / total_llr_sum if total_llr_sum > 0 else 0.0
            )

        else:
            raise ValueError(
                f"Unknown value_type: {value_type}. Must be 'size' or 'llr'."
            )

    return batch_norm_fractions


def calculate_committed_cluster_norm_fractions_from_csr(
    committed_clusters_csr: csr_matrix,
    committed_faults: List[np.ndarray],
    priors: np.ndarray,
    adj_matrix: np.ndarray,
    norm_order: float,
    value_type: str,
    eval_windows: Tuple[int, int] | None = None,
    _benchmarking: bool = False,
    num_jobs: int = 1,
    num_batches: int | None = None,
) -> np.ndarray:
    """
    Optimized calculation of committed cluster norm fractions from CSR matrix.

    This function uses optimized window splitting, logical OR operations, and
    igraph-based connected components analysis for maximum performance.

    Parameters
    ----------
    committed_clusters_csr : scipy sparse CSR matrix
        CSR matrix of booleans where rows correspond to samples and column "i*num_faults + j"
        corresponds to the j-th fault of i-th window.
    committed_faults : list of 1D numpy arrays
        List of boolean arrays representing committed faults for each window.
        Each array has shape (num_faults,) indicating which faults are committed in that window.
    priors : 1D numpy array of float
        Prior probabilities for each fault (determines num_faults).
    adj_matrix : 2D numpy array of bool
        Adjacency matrix for cluster labeling.
    norm_order : float
        Order for L_p norm calculation.
    value_type : str
        Type of values to calculate: "size" or "llr".
    eval_windows : tuple of int, optional
        If provided, only consider windows from init_eval_window to final_eval_window.
    _benchmarking : bool, optional
        If True, print detailed timing information for performance analysis.
    num_jobs : int, optional
        Number of parallel processes to use. Default is 1 (sequential processing).
    num_batches : int, optional
        Number of batches to split samples into. If None, defaults to num_jobs.

    Returns
    -------
    norm_fractions : 1D numpy array of float
        Norm fractions for each sample based on combined committed clusters.
    """

    if _benchmarking:
        start_time = time.perf_counter()
        print(
            f"=== BENCHMARKING: calculate_committed_cluster_norm_fractions_from_csr ==="
        )
        print(
            f"Input: num_samples={committed_clusters_csr.shape[0]}, "
            f"num_cols={committed_clusters_csr.shape[1]}, "
            f"value_type={value_type}, norm_order={norm_order}"
        )

    num_faults = len(priors)
    num_samples = committed_clusters_csr.shape[0]

    # Split CSR matrix by windows
    if _benchmarking:
        split_start = time.perf_counter()

    window_matrices = _split_csr_by_windows(
        committed_clusters_csr, num_faults, eval_windows
    )

    if _benchmarking:
        split_time = time.perf_counter() - split_start
        print(f"Window splitting time: {split_time:.4f}s")

    if not window_matrices:
        return np.zeros(num_samples, dtype=float)

    # Compute logical OR across windows to get combined committed clusters
    # shape: (num_samples, num_faults)
    if _benchmarking:
        or_start = time.perf_counter()

    combined_committed = _compute_logical_or_across_windows(window_matrices)

    if _benchmarking:
        or_time = time.perf_counter() - or_start
        print(f"Logical OR operations time: {or_time:.4f}s")

    # Process committed_faults to get total committed region
    # Determine which windows to consider based on eval_windows
    if _benchmarking:
        faults_start = time.perf_counter()

    if eval_windows is not None:
        start_window = max(0, eval_windows[0])
        end_window = min(len(committed_faults), eval_windows[1] + 1)
        selected_committed_faults = committed_faults[start_window:end_window]
    else:
        selected_committed_faults = committed_faults

    # Combine committed faults across windows using logical OR
    if selected_committed_faults:
        combined_committed_faults = np.logical_or.reduce(selected_committed_faults)
    else:
        combined_committed_faults = np.zeros(num_faults, dtype=bool)

    if _benchmarking:
        faults_time = time.perf_counter() - faults_start
        print(f"Committed faults processing time: {faults_time:.4f}s")

    # Pre-create igraph Graph for maximum performance
    if _benchmarking:
        graph_creation_start = time.perf_counter()
        print("Creating full igraph Graph using efficient CSR → COO conversion...")

    # Use efficient CSR → COO → igraph conversion as suggested
    if hasattr(adj_matrix, "tocoo"):
        # It's already a sparse matrix
        coo = adj_matrix.tocoo()
    else:
        # Convert dense matrix to sparse COO format first
        coo = sp.coo_matrix(adj_matrix)

    # Create igraph using the efficient method: CSR → COO → igraph
    full_graph = ig.Graph(n=coo.shape[0], directed=False)
    full_graph.add_edges(zip(coo.row, coo.col))

    if _benchmarking:
        graph_creation_time = time.perf_counter() - graph_creation_start
        print(f"Graph creation time: {graph_creation_time:.4f}s")
        print(
            f"Created graph with {full_graph.vcount()} vertices and {full_graph.ecount()} edges"
        )

    # Pre-compute commonly used values for performance
    if value_type == "llr":
        bit_llrs = np.log((1 - priors) / priors)
        total_llr_sum = np.sum(bit_llrs[combined_committed_faults])
    else:
        bit_llrs = None
        total_llr_sum = 0.0

    # Pre-compute total committed faults for size calculations
    total_committed_faults = np.sum(combined_committed_faults)

    # Set num_batches to num_jobs if not specified
    if num_batches is None:
        num_batches = num_jobs

    # Process each sample to convert boolean matrix to labeled clusters
    norm_fractions = np.zeros(num_samples, dtype=float)

    if _benchmarking:
        loop_start = time.perf_counter()
        print(f"Starting main sample processing loop for {num_samples} samples...")
        print(f"Using {num_jobs} processes with {num_batches} batches")

    # Prepare sample batches
    if num_jobs > 1 and num_samples > 0:
        # Split samples into batches for parallel processing using numpy
        sample_indices = np.arange(num_samples)
        batches = np.array_split(sample_indices, num_batches)

        # Process batches in parallel
        batch_results = Parallel(n_jobs=num_jobs)(
            delayed(_process_sample_batch)(
                batch,
                combined_committed.data,
                combined_committed.indices,
                combined_committed.indptr,
                full_graph,
                value_type,
                norm_order,
                bit_llrs,
                total_llr_sum,
                total_committed_faults,
            )
            for batch in batches
        )

    else:
        # Sequential processing: use single batch with all samples
        if num_samples > 0:
            sample_indices = np.arange(num_samples)
            batch_results = [
                _process_sample_batch(
                    sample_indices,
                    combined_committed.data,
                    combined_committed.indices,
                    combined_committed.indptr,
                    full_graph,
                    value_type,
                    norm_order,
                    bit_llrs,
                    total_llr_sum,
                    total_committed_faults,
                )
            ]
        else:
            batch_results = []

    # Combine results from all batches using numpy operations
    if batch_results:
        if len(norm_fractions) > 1:
            norm_fractions = np.concatenate(batch_results)
        else:
            norm_fractions = batch_results[0]

    if _benchmarking:
        loop_total_time = time.perf_counter() - loop_start
        total_time = time.perf_counter() - start_time

        print(f"Main loop total time: {loop_total_time:.4f}s")
        print(f"Total function time: {total_time:.4f}s")
        print("=" * 60)

    return norm_fractions


# =============================================================================
# Real-Time Post-Selection Analysis Functions
# =============================================================================


@numba.njit(fastmath=True, cache=True, parallel=True)
def calculate_realtime_metrics_vectorized(
    committed_clusters_data: np.ndarray,
    committed_clusters_indices: np.ndarray,
    committed_clusters_indptr: np.ndarray,
    priors: np.ndarray,
    adj_matrix: np.ndarray,
    num_samples: int,
    num_windows: int,
    num_faults_per_window: int,
    first_eval_window: int,
    metric_windows: int,
    norm_order: float,
    value_type_int: int,  # 0 = size, 1 = llr
) -> np.ndarray:
    """
    Ultra-optimized Numba kernel for real-time metric calculation.

    Calculates committed cluster norm fractions for all samples and evaluatable
    windows simultaneously using advanced vectorization and parallel processing.

    Parameters
    ----------
    committed_clusters_data, committed_clusters_indices, committed_clusters_indptr : np.ndarray
        CSR matrix components for committed cluster assignments.
    priors : np.ndarray
        Prior error probabilities.
    adj_matrix : np.ndarray
        Adjacency matrix for cluster analysis.
    num_samples, num_windows, num_faults_per_window : int
        Data dimensions.
    first_eval_window : int
        First window that can be evaluated.
    metric_windows : int
        Number of windows for metric evaluation.
    norm_order : float
        Order for L_p norm calculation.
    value_type_int : int
        0 for size calculations, 1 for LLR calculations.

    Returns
    -------
    np.ndarray
        Shape (num_samples, num_evaluatable_windows) containing metrics.
    """
    num_eval_windows = num_windows - first_eval_window
    metrics_matrix = np.zeros((num_samples, num_eval_windows), dtype=np.float64)

    if num_eval_windows <= 0:
        return metrics_matrix

    # Pre-compute LLR values if needed
    if value_type_int == 1:  # LLR
        bit_llrs = np.log((1.0 - priors) / priors)
        total_llr_sum = np.sum(bit_llrs)
    else:
        bit_llrs = np.zeros_like(priors)
        total_llr_sum = float(num_faults_per_window)

    # Process each evaluatable window
    for eval_idx in numba.prange(num_eval_windows):
        window_idx = first_eval_window + eval_idx

        # Define evaluation window range
        eval_start = max(0, window_idx - metric_windows + 1)
        eval_end = window_idx + 1

        # Process each sample for this evaluation window
        for sample_idx in range(num_samples):
            sample_start = committed_clusters_indptr[sample_idx]
            sample_end = committed_clusters_indptr[sample_idx + 1]

            if sample_end <= sample_start:
                metrics_matrix[sample_idx, eval_idx] = 0.0
                continue

            # Find max cluster ID across evaluation windows
            max_cluster_id = 0
            for j in range(sample_start, sample_end):
                col_idx = committed_clusters_indices[j]
                window_id = col_idx // num_faults_per_window

                if eval_start <= window_id < eval_end:
                    cluster_id = int(committed_clusters_data[j])
                    if cluster_id > max_cluster_id:
                        max_cluster_id = cluster_id

            if max_cluster_id == 0:
                metrics_matrix[sample_idx, eval_idx] = 0.0
                continue

            # Calculate cluster values
            cluster_values = np.zeros(max_cluster_id + 1, dtype=np.float64)

            for j in range(sample_start, sample_end):
                col_idx = committed_clusters_indices[j]
                window_id = col_idx // num_faults_per_window

                if eval_start <= window_id < eval_end:
                    cluster_id = int(committed_clusters_data[j])
                    if cluster_id > 0:
                        fault_idx = col_idx % num_faults_per_window
                        if value_type_int == 1:  # LLR
                            cluster_values[cluster_id] += bit_llrs[fault_idx]
                        else:  # Size
                            cluster_values[cluster_id] += 1.0

            # Calculate norm
            norm_val = 0.0
            if norm_order == 1.0:
                for cid in range(1, max_cluster_id + 1):
                    norm_val += abs(cluster_values[cid])
            elif norm_order == 2.0:
                sum_sq = 0.0
                for cid in range(1, max_cluster_id + 1):
                    v = cluster_values[cid]
                    sum_sq += v * v
                norm_val = np.sqrt(sum_sq)
            elif np.isinf(norm_order):
                for cid in range(1, max_cluster_id + 1):
                    v = abs(cluster_values[cid])
                    if v > norm_val:
                        norm_val = v
            else:
                sum_pow = 0.0
                for cid in range(1, max_cluster_id + 1):
                    v = abs(cluster_values[cid])
                    if v > 0:
                        sum_pow += v**norm_order
                if sum_pow > 0:
                    norm_val = sum_pow ** (1.0 / norm_order)

            # Calculate fraction
            if total_llr_sum > 0:
                metrics_matrix[sample_idx, eval_idx] = norm_val / total_llr_sum
            else:
                metrics_matrix[sample_idx, eval_idx] = 0.0

    return metrics_matrix


def batch_postselection_analysis(
    data_dir: str,
    param_combinations: List[str],
    cutoffs: np.ndarray,
    metric_windows: int = 1,
    norm_order: float = 2.0,
    value_type: str = "llr",
    num_jobs: int = 1,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    High-performance batch processing of multiple parameter combinations.

    Efficiently processes multiple sliding window parameter combinations in parallel
    for comprehensive post-selection analysis with arbitrary cutoff arrays.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combinations : List[str]
        List of parameter combination strings to process.
    cutoffs : np.ndarray
        Array of cutoff values to test across all combinations.
    metric_windows : int, default=1
        Number of windows for metric evaluation.
    norm_order : float, default=2.0
        Order for L_p norm calculation.
    value_type : str, default="llr"
        Type of cluster value calculation ("size" or "llr").
    num_jobs : int, default=1
        Number of parallel jobs for batch processing.
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Nested dictionary with results for each parameter combination.
        Structure: {param_combo: {result_key: result_array}}
    """
    if verbose:
        print(
            f"Starting batch post-selection analysis for {len(param_combinations)} combinations"
        )
        print(f"Testing {len(cutoffs)} cutoff values with {num_jobs} parallel jobs")

    # Import here to avoid circular imports
    from ..sliding_window_post_selection import analyze_parameter_combination

    def process_single_combination(
        param_combo: str,
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        """Process a single parameter combination."""
        try:
            if verbose:
                print(f"Processing {param_combo}...")

            results = analyze_parameter_combination(
                data_dir=data_dir,
                param_combo=param_combo,
                cutoffs=cutoffs,
                metric_windows=metric_windows,
                norm_order=norm_order,
                value_type=value_type,
            )
            return param_combo, results

        except Exception as e:
            if verbose:
                print(f"Error processing {param_combo}: {e}")
            return param_combo, {}

    # Process combinations in parallel
    if num_jobs > 1:
        results_list = Parallel(n_jobs=num_jobs)(
            delayed(process_single_combination)(combo) for combo in param_combinations
        )
    else:
        results_list = [
            process_single_combination(combo) for combo in param_combinations
        ]

    # Convert to dictionary
    results_dict = {combo: results for combo, results in results_list}

    if verbose:
        successful = sum(1 for _, results in results_list if results)
        print(
            f"Successfully processed {successful}/{len(param_combinations)} combinations"
        )

    return results_dict


def create_cutoff_arrays(
    fine_range: Tuple[float, float] = (0.001, 0.1),
    fine_points: int = 100,
    coarse_range: Tuple[float, float] = (1e-4, 0.5),
    coarse_points: int = 20,
    log_scale: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Create optimized cutoff arrays for parameter sweeps.

    Generates multiple cutoff arrays optimized for different analysis scenarios,
    allowing efficient exploration of the parameter space.

    Parameters
    ----------
    fine_range : Tuple[float, float], default=(0.001, 0.1)
        Range for fine-grained analysis.
    fine_points : int, default=100
        Number of points in fine-grained array.
    coarse_range : Tuple[float, float], default=(1e-4, 0.5)
        Range for coarse parameter sweep.
    coarse_points : int, default=20
        Number of points in coarse array.
    log_scale : bool, default=True
        Whether to use logarithmic spacing for coarse array.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing different cutoff arrays:
        - 'fine': Fine-grained linear spacing
        - 'coarse': Coarse logarithmic or linear spacing
        - 'ultra_fine': Extra fine spacing around typical values
        - 'broad': Very broad range with logarithmic spacing
    """
    cutoff_arrays = {}

    # Fine-grained linear spacing
    cutoff_arrays["fine"] = np.linspace(fine_range[0], fine_range[1], fine_points)

    # Coarse spacing (log or linear)
    if log_scale:
        cutoff_arrays["coarse"] = np.logspace(
            np.log10(coarse_range[0]), np.log10(coarse_range[1]), coarse_points
        )
    else:
        cutoff_arrays["coarse"] = np.linspace(
            coarse_range[0], coarse_range[1], coarse_points
        )

    # Ultra-fine spacing around typical values
    typical_low, typical_high = 0.005, 0.05
    cutoff_arrays["ultra_fine"] = np.linspace(
        typical_low, typical_high, fine_points * 2
    )

    # Broad range with logarithmic spacing
    cutoff_arrays["broad"] = np.logspace(-5, 0, coarse_points * 2)  # 1e-5 to 1.0

    return cutoff_arrays


def optimize_postselection_parameters(
    results: Dict[str, np.ndarray],
    target_abort_rate: float = 0.1,
    target_effective_trials: float = 1.0,
    optimization_metric: str = "p_fail",
) -> Dict[str, Union[float, int]]:
    """
    Find optimal post-selection parameters based on target constraints.

    Analyzes post-selection results to find optimal cutoff values that achieve
    target performance while minimizing logical error rates.

    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Results from post-selection analysis containing statistics arrays.
    target_abort_rate : float, default=0.1
        Target maximum abort rate (0-1).
    target_effective_trials : float, default=1.0
        Target maximum effective average trials.
    optimization_metric : str, default="p_fail"
        Metric to optimize ("p_fail" or "effective_avg_trials").

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary containing optimal parameters:
        - 'optimal_cutoff': Best cutoff value
        - 'optimal_index': Index of optimal cutoff
        - 'achieved_p_fail': Logical error rate at optimal cutoff
        - 'achieved_p_abort': Abort rate at optimal cutoff
        - 'achieved_effective_trials': Effective trials at optimal cutoff
    """
    cutoffs = results["cutoffs"]
    p_fail = results["p_fail"]
    p_abort = results["p_abort"]
    effective_avg_trials = results["effective_avg_trials"]

    # Find valid indices that meet constraints
    valid_mask = (
        (p_abort <= target_abort_rate)
        & (effective_avg_trials <= target_effective_trials)
        & (results["num_accepted"] > 0)  # Ensure we have accepted samples
    )

    if not np.any(valid_mask):
        # No valid solutions - return least constraining option
        best_idx = np.argmin(p_abort)  # Minimize abort rate as fallback
    else:
        # Find optimal solution among valid candidates
        valid_indices = np.where(valid_mask)[0]

        if optimization_metric == "p_fail":
            # Minimize logical error rate among valid solutions
            valid_p_fail = p_fail[valid_indices]
            best_valid_idx = np.argmin(valid_p_fail)
            best_idx = valid_indices[best_valid_idx]
        else:  # optimization_metric == "effective_avg_trials"
            # Minimize effective trials among valid solutions
            valid_trials = effective_avg_trials[valid_indices]
            best_valid_idx = np.argmin(valid_trials)
            best_idx = valid_indices[best_valid_idx]

    return {
        "optimal_cutoff": cutoffs[best_idx],
        "optimal_index": best_idx,
        "achieved_p_fail": p_fail[best_idx],
        "achieved_p_abort": p_abort[best_idx],
        "achieved_effective_trials": effective_avg_trials[best_idx],
        "meets_constraints": (
            bool(valid_mask[best_idx]) if len(valid_mask) > best_idx else False
        ),
    }
