import numba
import numpy as np
import time
from typing import Tuple, List
from scipy.sparse import csr_matrix


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


def calculate_committed_cluster_norm_fractions_from_csr(
    committed_clusters_csr: csr_matrix,
    committed_faults: List[np.ndarray],
    priors: np.ndarray,
    adj_matrix: np.ndarray,
    norm_order: float,
    value_type: str,
    eval_windows: Tuple[int, int] | None = None,
    _benchmarking: bool = False,
) -> np.ndarray:
    """
    Optimized calculation of committed cluster norm fractions from CSR matrix.

    This function uses optimized window splitting and logical OR operations,
    but still relies on scipy for connected components (label_clusters).

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

    Returns
    -------
    norm_fractions : 1D numpy array of float
        Norm fractions for each sample based on combined committed clusters.
    """
    # Import here to avoid circular imports
    from src.ldpc_post_selection.cluster_tools import label_clusters

    if _benchmarking:
        start_time = time.perf_counter()
        print(f"=== BENCHMARKING: calculate_committed_cluster_norm_fractions_from_csr ===")
        print(f"Input: num_samples={committed_clusters_csr.shape[0]}, "
              f"num_cols={committed_clusters_csr.shape[1]}, "
              f"value_type={value_type}, norm_order={norm_order}")

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

    # Process each sample to convert boolean matrix to labeled clusters
    norm_fractions = np.zeros(num_samples, dtype=float)
    
    if _benchmarking:
        loop_start = time.perf_counter()
        loop_times = {
            'nonzero_ops': 0.0,
            'cluster_labeling': 0.0,
            'norm_calculations': 0.0
        }
        print(f"Starting main sample processing loop for {num_samples} samples...")

    for sample_idx in range(num_samples):
        # Find indices of committed faults for this sample directly from sparse matrix
        if _benchmarking:
            nonzero_start = time.perf_counter()
            
        committed_cluster_fault_indices = combined_committed[sample_idx].nonzero()[0]
        
        if _benchmarking:
            loop_times['nonzero_ops'] += time.perf_counter() - nonzero_start

        if len(committed_cluster_fault_indices) == 0:
            # No committed clusters, norm fraction is 0
            norm_fractions[sample_idx] = 0.0
            continue

        # Label clusters using the adjacency matrix
        if _benchmarking:
            cluster_start = time.perf_counter()
            
        cluster_labels = label_clusters(adj_matrix, committed_cluster_fault_indices)
        
        if _benchmarking:
            loop_times['cluster_labeling'] += time.perf_counter() - cluster_start

        # Extract cluster labels for committed faults only
        nonzero_cluster_labels = cluster_labels[committed_cluster_fault_indices]

        # Calculate norm fractions based on cluster labels
        if _benchmarking:
            norm_start = time.perf_counter()
            
        if value_type == "size":
            # Calculate cluster sizes and total
            unique_clusters, cluster_sizes = np.unique(
                nonzero_cluster_labels,  # all > 0
                return_counts=True,
            )

            if len(cluster_sizes) > 0:
                # Calculate norm of cluster sizes
                if norm_order == 1.0:
                    inside_norm = np.sum(cluster_sizes)
                elif norm_order == 2.0:
                    inside_norm = np.sqrt(np.sum(cluster_sizes**2))
                elif np.isinf(norm_order):
                    inside_norm = np.max(cluster_sizes)
                else:
                    inside_norm = np.sum(cluster_sizes**norm_order) ** (
                        1.0 / norm_order
                    )

                # Total number of committed faults
                total_committed_faults = np.sum(combined_committed_faults)
                norm_fractions[sample_idx] = inside_norm / total_committed_faults if total_committed_faults > 0 else 0.0
            else:
                norm_fractions[sample_idx] = 0.0

        elif value_type == "llr":
            # Calculate cluster LLR sums and total
            bit_llrs = np.log((1 - priors) / priors)

            # Group LLRs by cluster
            unique_clusters, inverse_indices = np.unique(
                nonzero_cluster_labels,
                return_inverse=True,
            )

            if len(unique_clusters) > 0:
                # Calculate LLR sum for each cluster
                cluster_llr_sums = np.bincount(
                    inverse_indices,
                    weights=bit_llrs[
                        committed_cluster_fault_indices
                    ],
                )

                # Calculate norm of cluster LLR sums
                if norm_order == 1.0:
                    inside_norm = np.sum(np.abs(cluster_llr_sums))
                elif norm_order == 2.0:
                    inside_norm = np.sqrt(np.sum(cluster_llr_sums**2))
                elif np.isinf(norm_order):
                    inside_norm = np.max(np.abs(cluster_llr_sums))
                else:
                    inside_norm = np.sum(np.abs(cluster_llr_sums) ** norm_order) ** (
                        1.0 / norm_order
                    )

                # Total LLR sum for all committed faults
                total_llr_sum = np.sum(bit_llrs[combined_committed_faults])
                norm_fractions[sample_idx] = (
                    inside_norm / total_llr_sum if total_llr_sum > 0 else 0.0
                )
            else:
                norm_fractions[sample_idx] = 0.0

        else:
            raise ValueError(
                f"Unknown value_type: {value_type}. Must be 'size' or 'llr'."
            )
            
        if _benchmarking:
            loop_times['norm_calculations'] += time.perf_counter() - norm_start

    if _benchmarking:
        loop_total_time = time.perf_counter() - loop_start
        total_time = time.perf_counter() - start_time
        
        print(f"Main loop total time: {loop_total_time:.4f}s")
        print(f"  - Nonzero operations: {loop_times['nonzero_ops']:.4f}s "
              f"({loop_times['nonzero_ops']/loop_total_time*100:.1f}%)")
        print(f"  - Cluster labeling: {loop_times['cluster_labeling']:.4f}s "
              f"({loop_times['cluster_labeling']/loop_total_time*100:.1f}%)")
        print(f"  - Norm calculations: {loop_times['norm_calculations']:.4f}s "
              f"({loop_times['norm_calculations']/loop_total_time*100:.1f}%)")
        print(f"Total function time: {total_time:.4f}s")
        print("=" * 60)

    return norm_fractions