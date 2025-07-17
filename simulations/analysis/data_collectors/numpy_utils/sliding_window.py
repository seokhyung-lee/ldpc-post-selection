import numba
import numpy as np
from typing import Tuple, List
from scipy.sparse import csr_matrix


def _calculate_sliding_window_norm_fractions(
    cluster_data_list: List[List[np.ndarray]],
    norm_order: float,
    aggregation_type: str,
) -> np.ndarray:
    """
    Calculate norm fractions for sliding window cluster data.

    Parameters
    ----------
    cluster_data_list : list of list of numpy arrays
        List where each element is a shot's window data (list of arrays per window).
    norm_order : float
        Order for L_p norm calculation.
    aggregation_type : str
        Type of aggregation: "mean", "max", or "committed".

    Returns
    -------
    norm_fractions : 1D numpy array of float
        Norm fractions for each shot.
    """
    num_shots = len(cluster_data_list)
    if num_shots == 0:
        return np.zeros(0, dtype=np.float64)

    # Validate aggregation type
    if aggregation_type not in {"mean", "max", "committed"}:
        raise ValueError(f"Unknown aggregation_type: {aggregation_type}")

    # For numba compatibility, flatten the data structure
    flat_data = []
    shot_offsets = [0]
    window_offsets = []
    window_shot_indices = []

    for shot_idx, shot_windows in enumerate(cluster_data_list):
        for window_idx, window_data in enumerate(shot_windows):
            window_offsets.append(len(flat_data))
            window_shot_indices.append(shot_idx)
            # Ensure window_data is a numpy array before calling astype
            if not isinstance(window_data, np.ndarray):
                window_data = np.array(window_data)
            flat_data.extend(window_data.astype(np.float64))
        shot_offsets.append(len(window_offsets))

    if len(flat_data) == 0:
        return np.zeros(num_shots, dtype=np.float64)

    # Convert to numpy arrays for numba
    flat_data_np = np.array(flat_data, dtype=np.float64)
    shot_offsets_np = np.array(shot_offsets, dtype=np.int64)
    window_offsets_np = np.array(window_offsets, dtype=np.int64)
    window_shot_indices_np = np.array(window_shot_indices, dtype=np.int64)

    # Add final offset for easy indexing
    window_offsets_np = np.append(window_offsets_np, len(flat_data))

    agg_type_int = {"mean": 0, "max": 1, "committed": 2}[aggregation_type]

    # Process all shots using numba kernel
    return _calculate_sliding_window_norm_fractions_numba(
        flat_data_np,
        shot_offsets_np,
        window_offsets_np,
        window_shot_indices_np,
        norm_order,
        agg_type_int,
        num_shots,
    )


@numba.njit(fastmath=True, cache=True)
def _calculate_sliding_window_norm_fractions_numba(
    flat_data: np.ndarray,
    shot_offsets: np.ndarray,
    window_offsets: np.ndarray,
    window_shot_indices: np.ndarray,
    norm_order: float,
    agg_type_int: int,
    num_shots: int,
) -> np.ndarray:
    """
    Numba-optimized kernel for calculating sliding window norm fractions.

    Parameters
    ----------
    flat_data : 1D numpy array of float
        Flattened window data across all shots and windows.
    shot_offsets : 1D numpy array of int
        Offsets indicating where each shot's windows start in window_offsets.
    window_offsets : 1D numpy array of int
        Offsets indicating where each window's data starts in flat_data.
    window_shot_indices : 1D numpy array of int
        Shot index for each window.
    norm_order : float
        Order for L_p norm calculation.
    agg_type_int : int
        Aggregation type as integer: 0=mean, 1=max, 2=committed.
    num_shots : int
        Number of shots to process.

    Returns
    -------
    norm_fractions : 1D numpy array of float
        Norm fractions for each shot.
    """
    norm_fractions = np.zeros(num_shots, dtype=np.float64)

    # Pre-compute norm type flags for better branch prediction
    use_inf = np.isinf(norm_order)
    use_l1 = norm_order == 1.0
    use_l2 = norm_order == 2.0
    inv_norm_order = 1.0 / norm_order if norm_order != 0.0 and not use_inf else 1.0

    for shot_idx in range(num_shots):
        # Get window indices for this shot
        start_window_idx = shot_offsets[shot_idx]
        end_window_idx = shot_offsets[shot_idx + 1]
        num_windows = end_window_idx - start_window_idx

        if num_windows == 0:
            norm_fractions[shot_idx] = 0.0
            continue

        # For committed aggregation, only process the last window
        if agg_type_int == 2:  # committed
            start_window_idx = end_window_idx - 1

        # Process windows and accumulate for aggregation
        valid_window_count = 0
        sum_fracs = 0.0
        max_frac = 0.0
        last_frac = 0.0

        for window_idx in range(start_window_idx, end_window_idx):
            # Get window data boundaries
            window_start = window_offsets[window_idx]
            window_end = window_offsets[window_idx + 1]
            window_size = window_end - window_start

            window_frac = 0.0

            # Handle empty windows
            if window_size == 0:
                window_frac = 0.0
            elif window_size == 1:
                # Only outside region, norm fraction = 0
                window_frac = 0.0
            else:
                # Calculate total sum
                total_sum = 0.0
                for i in range(window_start, window_end):
                    total_sum += flat_data[i]

                # Handle zero total sum
                if total_sum == 0.0:
                    window_frac = 0.0
                else:
                    # Calculate inside norm (skip first element which is outside region)
                    inside_norm = 0.0

                    if use_inf:
                        # L∞ norm: max value
                        max_val = 0.0
                        for i in range(window_start + 1, window_end):
                            val = flat_data[i]
                            if val > max_val:
                                max_val = val
                        inside_norm = max_val

                    elif use_l1:
                        # L1 norm: sum of values
                        for i in range(window_start + 1, window_end):
                            inside_norm += flat_data[i]

                    elif use_l2:
                        # L2 norm: sqrt of sum of squares
                        sum_sq = 0.0
                        for i in range(window_start + 1, window_end):
                            val = flat_data[i]
                            sum_sq += val * val
                        inside_norm = np.sqrt(sum_sq)

                    else:
                        # General L_p norm
                        sum_pow = 0.0
                        for i in range(window_start + 1, window_end):
                            val = flat_data[i]
                            if val > 0:
                                sum_pow += val**norm_order
                        inside_norm = sum_pow**inv_norm_order

                    window_frac = inside_norm / total_sum

            # Update aggregation variables
            sum_fracs += window_frac
            if window_frac > max_frac:
                max_frac = window_frac
            last_frac = window_frac
            valid_window_count += 1

        # Aggregate across windows based on aggregation_type
        if valid_window_count == 0:
            norm_fractions[shot_idx] = 0.0
        elif agg_type_int == 0:  # mean
            norm_fractions[shot_idx] = sum_fracs / valid_window_count
        elif agg_type_int == 1:  # max
            norm_fractions[shot_idx] = max_frac
        else:  # committed (agg_type_int == 2)
            norm_fractions[shot_idx] = last_frac

    return norm_fractions


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
    priors: np.ndarray,
    adj_matrix: np.ndarray,
    norm_order: float,
    value_type: str,
    eval_windows: Tuple[int, int] | None = None,
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

    num_faults = len(priors)
    num_samples = committed_clusters_csr.shape[0]

    # Split CSR matrix by windows
    window_matrices = _split_csr_by_windows(
        committed_clusters_csr, num_faults, eval_windows
    )

    if not window_matrices:
        return np.zeros(num_samples, dtype=float)

    # Compute logical OR across windows to get combined committed clusters
    # shape: (num_samples, num_faults)
    combined_committed = _compute_logical_or_across_windows(window_matrices)

    # Process each sample to convert boolean matrix to labeled clusters
    norm_fractions = np.zeros(num_samples, dtype=float)

    for sample_idx in range(num_samples):
        # Find indices of committed faults for this sample directly from sparse matrix
        committed_cluster_fault_indices = combined_committed[sample_idx].nonzero()[1]

        if len(committed_cluster_fault_indices) == 0:
            # No committed clusters, norm fraction is 0
            norm_fractions[sample_idx] = 0.0
            continue

        # Label clusters using the adjacency matrix
        cluster_labels = label_clusters(adj_matrix, committed_cluster_fault_indices)

        # Extract cluster labels for committed faults only
        nonzero_cluster_labels = cluster_labels[committed_cluster_fault_indices]

        # Calculate norm fractions based on cluster labels
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
                total_faults = combined_committed.shape[1]
                norm_fractions[sample_idx] = inside_norm / total_faults
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

                # Total LLR sum for committed faults
                total_llr_sum = np.sum(bit_llrs[committed_cluster_fault_indices])
                norm_fractions[sample_idx] = (
                    inside_norm / total_llr_sum if total_llr_sum > 0 else 0.0
                )
            else:
                norm_fractions[sample_idx] = 0.0

        else:
            raise ValueError(
                f"Unknown value_type: {value_type}. Must be 'size' or 'llr'."
            )

    return norm_fractions