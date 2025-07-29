import numba
import numpy as np
from typing import Tuple


@numba.njit(fastmath=True, cache=True)
def _calculate_histograms_bplsd_legacy(
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
    Calculate total, fail, converge, and fail_converge histograms using Numba with an optimized single loop.

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
    This implementation manually calculates the histogram bins in a single pass
    for potentially better performance within Numba compared to calling
    np.histogram multiple times. It assumes NaNs have been filtered from
    values_np before calling. It replicates np.histogram's behavior
    regarding bin edges: bins are [left, right), except the last bin which
    is [left, right].
    """
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:  # Handle empty or invalid bin_edges
        return (
            total_counts_hist,
            fail_counts_hist,
            converge_counts_hist,
            fail_converge_counts_hist,
        )

    for i in range(len(values_np)):
        val = values_np[i]

        # Check if value is within the histogram range
        if val < bin_edges[0] or val > bin_edges[-1]:
            continue

        # Determine the bin index using searchsorted
        bin_idx = np.searchsorted(bin_edges[:-1], val, side="right")

        if bin_idx == n_bins and val == bin_edges[-1]:
            bin_idx = n_bins - 1
        else:
            bin_idx = bin_idx - 1

        if 0 <= bin_idx < n_bins:
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
def _calculate_histograms_matching_legacy(
    values_np: np.ndarray,  # Expect float array
    fail_mask_np: np.ndarray,  # Expect boolean array
    bin_edges: np.ndarray,  # Expect float array
    total_counts_hist: np.ndarray,  # Expect int64 array
    fail_counts_hist: np.ndarray,  # Expect int64 array
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate total and fail histograms using Numba with an optimized single loop.
    This is a simplified version of _calculate_histograms_bplsd_numba,
    calculating only total and fail counts.

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
    This implementation manually calculates the histogram bins in a single pass.
    It assumes NaNs have been filtered from values_np before calling.
    It replicates np.histogram's behavior regarding bin edges:
    bins are [left, right), except the last bin which is [left, right].
    """
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:  # Handle empty or invalid bin_edges
        return (
            total_counts_hist,
            fail_counts_hist,
        )

    for i in range(len(values_np)):
        val = values_np[i]

        # Check if value is within the histogram range
        if val < bin_edges[0] or val > bin_edges[-1]:
            continue

        # Determine the bin index using searchsorted
        # Find the insertion point for `val` in `bin_edges[:-1]` (all bin starts)
        # `side='right'` means if `val` is equal to an edge, it goes to the right bin
        bin_idx = np.searchsorted(bin_edges[:-1], val, side="right")

        if bin_idx == n_bins and val == bin_edges[-1]:
            bin_idx = n_bins - 1
        else:
            bin_idx = bin_idx - 1

        if 0 <= bin_idx < n_bins:
            total_counts_hist[bin_idx] += 1
            if fail_mask_np[i]:
                fail_counts_hist[bin_idx] += 1

    return (
        total_counts_hist,
        fail_counts_hist,
    )


@numba.njit(parallel=True, fastmath=True, cache=True)
def _calculate_norms_for_samples_legacy(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    norm_order: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the L_p norm for each sample from flattened cluster data,
    excluding the last element of each sample's segment (treated as outside_cluster_value).
    The norm is (sum(value^order))^(1/order) or max(abs(value)) for L-infinity.

    Parameters
    ----------
    flat_data : 1D numpy array
        Flattened cluster values. For each sample, the last value in its segment
        is considered the outside_cluster_value.
    offsets : 1D numpy array
        Starting indices in flat_data for each sample.
    norm_order : float
        The order p for the L_p norm. Must be positive (can be np.inf).

    Returns
    -------
    sample_norms : 1D numpy array
        The calculated norm for the "inside cluster" values of each sample.
        Norm is 0 if no "inside cluster" values exist or if the sample data is empty.
    outside_cluster_values : 1D numpy array
        The last element of each sample's segment in flat_data.
        np.nan if the sample's segment is empty.
    """
    n = offsets.size
    # Pass the sentinel "end" index instead of building it with np.concatenate
    end_of_data = flat_data.size

    # pre-allocate output (NumPy scalars are fine – Numba understands them)
    norms = np.zeros(n, dtype=np.float32)
    outside = np.full(n, np.nan, dtype=np.float32)

    # Constant-fold tests that do not depend on i
    use_inf = np.isinf(norm_order)

    for i in numba.prange(n):  # ← prange enables multi-threaded loop
        # Get raw start and stop from offsets (can be float)
        _raw_start = offsets[i]
        _raw_stop = offsets[i + 1] if i < n - 1 else end_of_data

        # Convert to int for indexing and range()
        start = int(_raw_start)
        stop = int(_raw_stop)

        if stop <= start:  # empty slice – nothing to do
            continue

        # last element belongs to "outside"
        last_val = flat_data[stop - 1]
        outside[i] = last_val

        # all but the last one are inside; bail if length-0
        if stop - start == 1:
            continue

        # NB: working on the original array avoids an extra allocation
        if use_inf:  # L∞
            max_abs = 0.0
            for j in range(start, stop - 1):
                v = flat_data[j]
                if v < 0:
                    v = -v
                if v > max_abs:
                    max_abs = v
            norms[i] = max_abs
        else:
            # special-case the common p = 1, 2 to avoid slow np.power
            if norm_order == 1.0:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    s += v if v >= 0 else -v
                norms[i] = s
            elif norm_order == 2.0:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    s += v * v
                norms[i] = np.sqrt(s)
            elif norm_order == 0.5:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    s += np.sqrt(v)
                norms[i] = s * s
            else:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    # fastmath allows powerf; fall back to np.power otherwise
                    s += v**norm_order
                norms[i] = s ** (1.0 / norm_order)

    return norms, outside
