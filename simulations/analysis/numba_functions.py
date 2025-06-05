import numba
import numpy as np
from typing import Tuple


@numba.njit(fastmath=True, cache=True)
def _is_uniform_binning(bin_edges: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if bin edges represent uniform binning within tolerance.

    Parameters
    ----------
    bin_edges : 1D numpy array of float
        Bin edges to check.
    tolerance : float
        Relative tolerance for uniformity check.

    Returns
    -------
    is_uniform : bool
        True if binning is uniform within tolerance.
    """
    if len(bin_edges) < 2:
        return True

    expected_width = (bin_edges[-1] - bin_edges[0]) / (len(bin_edges) - 1)
    if expected_width == 0:
        return True

    for i in range(1, len(bin_edges)):
        actual_width = bin_edges[i] - bin_edges[i - 1]
        if abs(actual_width - expected_width) / expected_width > tolerance:
            return False
    return True


@numba.njit(fastmath=True, cache=True)
def _calculate_histograms_bplsd_numba(
    values_np: np.ndarray,  # Expect float array
    fail_mask_np: np.ndarray,  # Expect boolean array
    bin_edges: np.ndarray,  # Expect float array
    total_counts_hist: np.ndarray,  # Expect int64 array
    fail_counts_hist: np.ndarray,  # Expect int64 array
    converge_mask_np: np.ndarray,  # Expect boolean array
    converge_counts_hist: np.ndarray,  # Expect int64 array
    fail_converge_counts_hist: np.ndarray,  # Expect int64 array
    fail_bp_mask_np: np.ndarray,  # Expect boolean array
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized histogram calculation using uniform binning when possible.

    Parameters
    ----------
    values_np : 1D numpy array of float
        Values to be histogrammed (e.g., cluster fractions or other column data).
        NaNs should be removed before calling.
    fail_mask_np : 1D numpy array of bool
        Indicates failures, aligned with values_np.
    bin_edges : 1D numpy array of float
        Defines the histogram bin edges, must be sorted.
    total_counts_hist : 1D numpy array of int64
        Histogram counts for all samples (updated in-place).
    fail_counts_hist : 1D numpy array of int64
        Histogram counts for failed samples (updated in-place).
    converge_mask_np : 1D numpy array of bool
        Indicates convergence, aligned with values_np.
    converge_counts_hist : 1D numpy array of int64
        Histogram counts for converged samples (updated in-place).
    fail_converge_counts_hist : 1D numpy array of int64
        Histogram counts for samples where both fail_bp and converge are true (updated in-place).
    fail_bp_mask_np : 1D numpy array of bool
        Indicates failures based on BP decoder, aligned with values_np.

    Returns
    -------
    total_counts_hist : 1D numpy array of int64
        The updated histogram counts for all samples.
    fail_counts_hist : 1D numpy array of int64
        The updated histogram counts for failed samples.
    converge_counts_hist : 1D numpy array of int64
        The updated histogram counts for converged samples.
    fail_converge_counts_hist : 1D numpy array of int64
        The updated histogram counts for fail_converged samples.

    Notes
    -----
    This optimized version detects uniform binning and uses O(1) bin index calculation
    instead of O(log n) binary search for each value, significantly improving performance.
    """
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:
        return (
            total_counts_hist,
            fail_counts_hist,
            converge_counts_hist,
            fail_converge_counts_hist,
        )

    # Check if we can use uniform binning optimization
    is_uniform = _is_uniform_binning(bin_edges)

    if is_uniform:
        # Uniform binning: O(1) bin calculation
        bin_min = bin_edges[0]
        bin_max = bin_edges[-1]
        bin_width = (bin_max - bin_min) / n_bins
        inv_bin_width = 1.0 / bin_width if bin_width > 0 else 0.0

        for i in range(len(values_np)):
            val = values_np[i]

            if val < bin_min or val > bin_max:
                continue

            # Direct calculation for uniform bins
            if val == bin_max:
                bin_idx = n_bins - 1
            else:
                bin_idx = int((val - bin_min) * inv_bin_width)

            if 0 <= bin_idx < n_bins:
                total_counts_hist[bin_idx] += 1
                if fail_mask_np[i]:
                    fail_counts_hist[bin_idx] += 1
                if converge_mask_np[i]:
                    converge_counts_hist[bin_idx] += 1
                if fail_bp_mask_np[i] and converge_mask_np[i]:
                    fail_converge_counts_hist[bin_idx] += 1
    else:
        # Non-uniform binning: use linear search for small number of bins
        # or optimized binary search for larger number of bins
        if n_bins <= 32:  # Linear search threshold
            for i in range(len(values_np)):
                val = values_np[i]

                if val < bin_edges[0] or val > bin_edges[-1]:
                    continue

                # Linear search for small number of bins
                bin_idx = -1
                for j in range(n_bins):
                    if val < bin_edges[j + 1] or (
                        j == n_bins - 1 and val == bin_edges[j + 1]
                    ):
                        bin_idx = j
                        break

                if bin_idx >= 0:
                    total_counts_hist[bin_idx] += 1
                    if fail_mask_np[i]:
                        fail_counts_hist[bin_idx] += 1
                    if converge_mask_np[i]:
                        converge_counts_hist[bin_idx] += 1
                    if fail_bp_mask_np[i] and converge_mask_np[i]:
                        fail_converge_counts_hist[bin_idx] += 1
        else:
            # Optimized binary search for larger number of bins
            for i in range(len(values_np)):
                val = values_np[i]

                if val < bin_edges[0] or val > bin_edges[-1]:
                    continue

                # Manual binary search (more efficient than np.searchsorted in this context)
                left = 0
                right = n_bins - 1
                bin_idx = -1

                while left <= right:
                    mid = (left + right) // 2
                    if val < bin_edges[mid]:
                        right = mid - 1
                    elif val >= bin_edges[mid + 1] and mid < n_bins - 1:
                        left = mid + 1
                    else:
                        bin_idx = mid
                        break

                # Handle edge case for last bin
                if bin_idx == -1 and val == bin_edges[-1]:
                    bin_idx = n_bins - 1

                if bin_idx >= 0:
                    total_counts_hist[bin_idx] += 1
                    if fail_mask_np[i]:
                        fail_counts_hist[bin_idx] += 1
                    if converge_mask_np[i]:
                        converge_counts_hist[bin_idx] += 1
                    if fail_bp_mask_np[i] and converge_mask_np[i]:
                        fail_converge_counts_hist[bin_idx] += 1

    return (
        total_counts_hist,
        fail_counts_hist,
        converge_counts_hist,
        fail_converge_counts_hist,
    )


@numba.njit(fastmath=True, cache=True)
def _calculate_histograms_matching_numba(
    values_np: np.ndarray,  # Expect float array
    fail_mask_np: np.ndarray,  # Expect boolean array
    bin_edges: np.ndarray,  # Expect float array
    total_counts_hist: np.ndarray,  # Expect int64 array
    fail_counts_hist: np.ndarray,  # Expect int64 array
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized histogram calculation for total and fail counts using uniform binning when possible.

    Parameters
    ----------
    values_np : 1D numpy array of float
        Values to be histogrammed. NaNs should be removed before calling.
    fail_mask_np : 1D numpy array of bool
        Indicates failures, aligned with values_np.
    bin_edges : 1D numpy array of float
        Defines the histogram bin edges, must be sorted.
    total_counts_hist : 1D numpy array of int64
        Histogram counts for all samples (updated in-place).
    fail_counts_hist : 1D numpy array of int64
        Histogram counts for failed samples (updated in-place).

    Returns
    -------
    total_counts_hist : 1D numpy array of int64
        The updated histogram counts for all samples.
    fail_counts_hist : 1D numpy array of int64
        The updated histogram counts for failed samples.

    Notes
    -----
    This optimized version detects uniform binning and uses O(1) bin index calculation
    instead of O(log n) binary search for each value, significantly improving performance.
    """
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:
        return (total_counts_hist, fail_counts_hist)

    # Check if we can use uniform binning optimization
    is_uniform = _is_uniform_binning(bin_edges)

    if is_uniform:
        # Uniform binning: O(1) bin calculation
        bin_min = bin_edges[0]
        bin_max = bin_edges[-1]
        bin_width = (bin_max - bin_min) / n_bins
        inv_bin_width = 1.0 / bin_width if bin_width > 0 else 0.0

        for i in range(len(values_np)):
            val = values_np[i]

            if val < bin_min or val > bin_max:
                continue

            # Direct calculation for uniform bins
            if val == bin_max:
                bin_idx = n_bins - 1
            else:
                bin_idx = int((val - bin_min) * inv_bin_width)

            if 0 <= bin_idx < n_bins:
                total_counts_hist[bin_idx] += 1
                if fail_mask_np[i]:
                    fail_counts_hist[bin_idx] += 1
    else:
        # Non-uniform binning: use linear search for small number of bins
        # or optimized binary search for larger number of bins
        if n_bins <= 32:  # Linear search threshold
            for i in range(len(values_np)):
                val = values_np[i]

                if val < bin_edges[0] or val > bin_edges[-1]:
                    continue

                # Linear search for small number of bins
                bin_idx = -1
                for j in range(n_bins):
                    if val < bin_edges[j + 1] or (
                        j == n_bins - 1 and val == bin_edges[j + 1]
                    ):
                        bin_idx = j
                        break

                if bin_idx >= 0:
                    total_counts_hist[bin_idx] += 1
                    if fail_mask_np[i]:
                        fail_counts_hist[bin_idx] += 1
        else:
            # Optimized binary search for larger number of bins
            for i in range(len(values_np)):
                val = values_np[i]

                if val < bin_edges[0] or val > bin_edges[-1]:
                    continue

                # Manual binary search (more efficient than np.searchsorted in this context)
                left = 0
                right = n_bins - 1
                bin_idx = -1

                while left <= right:
                    mid = (left + right) // 2
                    if val < bin_edges[mid]:
                        right = mid - 1
                    elif val >= bin_edges[mid + 1] and mid < n_bins - 1:
                        left = mid + 1
                    else:
                        bin_idx = mid
                        break

                # Handle edge case for last bin
                if bin_idx == -1 and val == bin_edges[-1]:
                    bin_idx = n_bins - 1

                if bin_idx >= 0:
                    total_counts_hist[bin_idx] += 1
                    if fail_mask_np[i]:
                        fail_counts_hist[bin_idx] += 1

    return (total_counts_hist, fail_counts_hist)


@numba.njit(parallel=True, fastmath=True, cache=True)
def _calculate_norms_for_samples(
    flat_data: np.ndarray,  # Any numeric array type
    offsets: np.ndarray,  # Any numeric array type (will be converted to int)
    norm_order: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized calculation of L_p norms for each sample from flattened cluster data.

    Parameters
    ----------
    flat_data : 1D numpy array of float
        Flattened cluster values (assumed to be positive). For each sample, the last value
        in its segment is considered the outside_cluster_value.
    offsets : 1D numpy array of int | float
        Starting indices in flat_data for each sample.
    norm_order : float
        The order p for the L_p norm. Must be positive (can be np.inf).

    Returns
    -------
    sample_norms : 1D numpy array of float
        The calculated norm for the "inside cluster" values of each sample.
        Norm is 0 if no "inside cluster" values exist or if the sample data is empty.
    outside_cluster_values : 1D numpy array of float
        The last element of each sample's segment in flat_data.
        np.nan if the sample's segment is empty.

    Notes
    -----
    This optimized version assumes all values in flat_data are positive, eliminating
    the need for absolute value calculations and improving performance.
    """
    n = offsets.size
    end_of_data = flat_data.size

    # Pre-allocate output arrays
    norms = np.zeros(n, dtype=np.float32)
    outside = np.full(n, np.nan, dtype=np.float32)

    # Pre-compute norm type flags for better branch prediction
    use_inf = np.isinf(norm_order)
    use_l1 = norm_order == 1.0
    use_l2 = norm_order == 2.0
    use_sqrt = norm_order == 0.5

    # Precompute reciprocal for general case to avoid divisions
    inv_norm_order = 1.0 / norm_order if norm_order != 0.0 and not use_inf else 1.0

    for i in numba.prange(n):
        # Get segment boundaries - ensure int conversion
        start_idx = int(offsets[i])
        if i < n - 1:
            stop_idx = int(offsets[i + 1])
        else:
            stop_idx = int(end_of_data)

        # Early exit for empty segments
        if stop_idx <= start_idx:
            continue

        # Extract outside value (last element) - force int indexing
        last_idx = int(stop_idx - 1)
        outside[i] = flat_data[last_idx]

        # Early exit if no inside values - force int calculation
        inside_length = int(stop_idx - start_idx - 1)
        if inside_length <= 0:
            continue

        # Optimized norm calculations with better memory access
        if use_inf:
            # L∞ norm: max value (assuming positive values)
            max_val = 0.0
            for j in range(start_idx, last_idx):
                val = flat_data[j]
                if val > max_val:
                    max_val = val
            norms[i] = max_val

        elif use_l1:
            # L1 norm: sum of values (assuming positive values)
            sum_val = 0.0
            for j in range(start_idx, last_idx):
                sum_val += flat_data[j]
            norms[i] = sum_val

        elif use_l2:
            # L2 norm: sqrt of sum of squares
            sum_sq = 0.0
            for j in range(start_idx, last_idx):
                val = flat_data[j]
                sum_sq += val * val
            norms[i] = np.sqrt(sum_sq)

        elif use_sqrt:
            # Special case for p = 0.5: optimized sqrt handling
            sum_sqrt = 0.0
            for j in range(start_idx, last_idx):
                val = flat_data[j]
                # Match legacy behavior: sqrt of raw value (assume non-negative for p=0.5)
                sum_sqrt += np.sqrt(val)
            norms[i] = sum_sqrt * sum_sqrt

        else:
            # General L_p norm case
            sum_pow = 0.0
            for j in range(start_idx, last_idx):
                val = flat_data[j]
                # Match legacy behavior: direct power calculation
                sum_pow += val**norm_order
            norms[i] = sum_pow**inv_norm_order

    return norms, outside
