import numba
import numpy as np
from typing import Tuple, List, Callable
from scipy.sparse import csr_matrix


def _precompile_and_run_numba_kernel(kernel: Callable, precompile: bool, *args) -> any:
    """
    Pre-compiles a Numba kernel (if requested) and then runs it with actual data.

    If `precompile` is True, this function infers the required dtypes from the
    provided arguments, constructs suitable dummy data, and calls the kernel to
    trigger JIT compilation. It then proceeds to run the kernel with the
    actual arguments. It includes specific logic for CSR-like data
    structures that require special handling (e.g., `indptr` arrays).

    Parameters
    ----------
    kernel : Numba jitted function
        The Numba kernel to pre-compile and run.
    precompile : bool
        If True, the kernel is pre-compiled before execution.
    *args :
        Actual arguments to be passed to the kernel.

    Returns
    -------
    return_value : any
        The result of executing the kernel with the provided arguments.
    """
    if precompile:
        dummy_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                # Special handling for CSR-based kernels where `indptr` (arg 2)
                # must be `[0]` for a zero-sample case.
                if "cluster" in kernel.__name__ and i == 2:
                    dummy_args.append(np.zeros(1, dtype=arg.dtype))
                else:
                    dummy_args.append(np.empty(0, dtype=arg.dtype))
            elif isinstance(arg, bool):
                dummy_args.append(False)
            elif isinstance(arg, int):
                dummy_args.append(0)
            elif isinstance(arg, float):
                dummy_args.append(0.0)
            else:
                dummy_args.append(arg)

        try:
            kernel(*dummy_args)
        except Exception:
            # Exceptions are expected with dummy data; the goal is compilation.
            pass

    return kernel(*args)


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


def calculate_cluster_metrics_from_csr(
    clusters: csr_matrix,
    method: str,
    priors: np.ndarray | None = None,
    norm_order: float = 2.0,
    precompile: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Calculate various cluster metrics from a CSR sparse matrix representation.

    Parameters
    ----------
    clusters : scipy sparse CSR matrix
        Sparse matrix where rows represent samples and columns represent bits.
        Non-zero values contain cluster IDs, with 0 representing outside clusters.
    method : str
        The calculation method to use.
    priors : 1D numpy array of float, optional
        Prior probabilities for each bit position. Required for "inv_entropy" and "inv_prior_sum".
    norm_order : float, default 2.0
        The order p for the L_p norm calculation (can be np.inf). Only used for "norm".
    precompile : bool, default False
        If True, pre-compiles the required Numba kernel if it's not already compiled.

    Returns
    -------
    result : tuple[np.ndarray, np.ndarray] | np.ndarray
        For "norm": tuple of (inside_norms, outside_values)
        For "inv_entropy" or "inv_prior_sum": single array of calculated values

    Raises
    ------
    ValueError
        If invalid method is specified or required parameters are missing.
    """
    num_samples, num_bits = clusters.shape

    if method in ["norm", "llr_norm"]:
        calculate_llrs = method == "llr_norm"
        if calculate_llrs:
            if priors is None:
                raise ValueError("priors is required for llr_norm calculation")
            bit_llrs = np.log((1 - priors) / priors)
        else:
            bit_llrs = np.zeros(num_bits)

        if clusters.nnz == 0:
            total_outside = np.sum(bit_llrs) if calculate_llrs else float(num_bits)
            return np.zeros(num_samples), np.full(num_samples, total_outside)

        max_cluster_id = int(clusters.data.max()) if clusters.nnz > 0 else 0

        return _precompile_and_run_numba_kernel(
            _numba_cluster_norm_kernel,
            precompile,
            clusters.data,
            clusters.indices,
            clusters.indptr,
            num_samples,
            num_bits,
            max_cluster_id,
            bit_llrs,
            norm_order,
            calculate_llrs,
        )

    elif "inv_entropy" in method:
        if priors is None:
            raise ValueError("priors is required for inv_entropies calculation")

        if clusters.nnz == 0:
            return np.zeros(num_samples, dtype=np.float64)

        return _precompile_and_run_numba_kernel(
            _numba_inv_entropy_kernel,
            precompile,
            clusters.data,
            clusters.indices,
            clusters.indptr,
            num_samples,
            num_bits,
            priors,
        )

    elif "inv_prior_sum" in method:
        if priors is None:
            raise ValueError("priors is required for inv_priors calculation")

        if clusters.nnz == 0:
            return np.zeros(num_samples, dtype=np.float64)

        return _precompile_and_run_numba_kernel(
            _numba_inv_priors_kernel,
            precompile,
            clusters.data,
            clusters.indices,
            clusters.indptr,
            num_samples,
            num_bits,
            priors,
        )

    else:
        raise ValueError(
            f"Invalid method: {method}. Must be one of 'norm', 'llr_norm', 'inv_entropy', 'inv_prior_sum'"
        )


def _calculate_cluster_norms_from_flat_data_numba(
    flat_data: np.ndarray,  # Any numeric array type
    offsets: np.ndarray,  # Any numeric array type (will be converted to int)
    norm_order: float,
    sample_indices: np.ndarray | None = None,  # Optional sample filtering
    precompile: bool = False,
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
    sample_indices : 1D numpy array of int, optional
        Array of sample indices to include in the calculation. If None, all samples are processed.
    precompile : bool, default False
        If True, pre-compiles the required Numba kernel if it's not already compiled.

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
    if sample_indices is None:
        return _precompile_and_run_numba_kernel(
            _calculate_cluster_norms_numba_kernel,
            precompile,
            flat_data,
            offsets,
            norm_order,
        )
    else:
        return _precompile_and_run_numba_kernel(
            _calculate_cluster_norms_filtered_numba_kernel,
            precompile,
            flat_data,
            offsets,
            norm_order,
            sample_indices,
        )


@numba.njit(parallel=True, fastmath=True, cache=True)
def _calculate_cluster_norms_numba_kernel(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    norm_order: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Original unfiltered numba kernel for calculating cluster norms."""
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


@numba.njit(parallel=True, fastmath=True, cache=True)
def _calculate_cluster_norms_filtered_numba_kernel(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    norm_order: float,
    sample_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filtered numba kernel for calculating cluster norms on selected samples."""
    n = offsets.size
    end_of_data = flat_data.size
    output_size = len(sample_indices)

    # Pre-allocate output arrays
    norms = np.zeros(output_size, dtype=np.float32)
    outside = np.full(output_size, np.nan, dtype=np.float32)

    # Pre-compute norm type flags for better branch prediction
    use_inf = np.isinf(norm_order)
    use_l1 = norm_order == 1.0
    use_l2 = norm_order == 2.0
    use_sqrt = norm_order == 0.5

    # Precompute reciprocal for general case to avoid divisions
    inv_norm_order = 1.0 / norm_order if norm_order != 0.0 and not use_inf else 1.0

    for output_idx in numba.prange(output_size):
        # Get the actual sample index to process
        i = int(sample_indices[output_idx])

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
        outside[output_idx] = flat_data[last_idx]

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
            norms[output_idx] = max_val

        elif use_l1:
            # L1 norm: sum of values (assuming positive values)
            sum_val = 0.0
            for j in range(start_idx, last_idx):
                sum_val += flat_data[j]
            norms[output_idx] = sum_val

        elif use_l2:
            # L2 norm: sqrt of sum of squares
            sum_sq = 0.0
            for j in range(start_idx, last_idx):
                val = flat_data[j]
                sum_sq += val * val
            norms[output_idx] = np.sqrt(sum_sq)

        elif use_sqrt:
            # Special case for p = 0.5: optimized sqrt handling
            sum_sqrt = 0.0
            for j in range(start_idx, last_idx):
                val = flat_data[j]
                # Match legacy behavior: sqrt of raw value (assume non-negative for p=0.5)
                sum_sqrt += np.sqrt(val)
            norms[output_idx] = sum_sqrt * sum_sqrt

        else:
            # General L_p norm case
            sum_pow = 0.0
            for j in range(start_idx, last_idx):
                val = flat_data[j]
                # Match legacy behavior: direct power calculation
                sum_pow += val**norm_order
            norms[output_idx] = sum_pow**inv_norm_order

    return norms, outside


@numba.njit(fastmath=True, cache=True)
def _numba_cluster_norm_kernel(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    num_samples: int,
    num_bits: int,
    max_cluster_id: int,
    bit_llrs: np.ndarray,
    norm_order: float,
    calculate_llrs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Core JIT-compiled function for calculating cluster norms from CSR matrix components.

    Parameters
    ----------
    data : 1D numpy array of float
        CSR matrix data array containing cluster IDs.
    indices : 1D numpy array of int
        CSR matrix indices array containing bit positions.
    indptr : 1D numpy array of int
        CSR matrix index pointer array for sample boundaries.
    num_samples : int
        Number of samples (rows) in the CSR matrix.
    num_bits : int
        Number of bits (columns) in the CSR matrix.
    max_cluster_id : int
        Maximum cluster ID present in the data.
    bit_llrs : 1D numpy array of float
        Log-likelihood ratios for each bit position.
    norm_order : float
        The order p for the L_p norm calculation.
    calculate_llrs : bool
        If True, use LLR values; if False, use counts.

    Returns
    -------
    inside_norms : 1D numpy array of float
        Calculated L_p norms for inside cluster values of each sample.
    outside_values : 1D numpy array of float
        Sum of LLRs or counts for outside cluster values of each sample.
    """
    inside_norms = np.zeros(num_samples, dtype=np.float64)
    outside_values = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        start_idx = indptr[i]
        end_idx = indptr[i + 1]

        # Calculate outside values (matching original logic)
        outside_count = 0
        outside_llr_sum = 0.0
        positions_stored = np.zeros(num_bits, dtype=np.uint8)

        # First pass: mark stored positions and count explicit zeros
        for data_idx in range(start_idx, end_idx):
            cluster_id = int(data[data_idx])
            bit_pos = int(indices[data_idx])

            if bit_pos < num_bits:
                positions_stored[bit_pos] = 1

                if cluster_id == 0:
                    # Explicit outside cluster
                    outside_count += 1
                    if calculate_llrs and bit_pos < len(bit_llrs):
                        outside_llr_sum += bit_llrs[bit_pos]

        # Second pass: count implicit zeros (positions not stored)
        for bit_pos in range(num_bits):
            if positions_stored[bit_pos] == 0:
                # Implicit outside cluster (not stored = cluster_id 0)
                outside_count += 1
                if calculate_llrs and bit_pos < len(bit_llrs):
                    outside_llr_sum += bit_llrs[bit_pos]

        if calculate_llrs:
            outside_values[i] = outside_llr_sum
        else:
            outside_values[i] = outside_count

        # Skip if no data for this sample
        if end_idx <= start_idx:
            continue

        # Find maximum cluster ID to allocate arrays for this sample
        sample_max_cluster_id = 0
        for data_idx in range(start_idx, end_idx):
            cluster_id = int(data[data_idx])
            if cluster_id > sample_max_cluster_id:
                sample_max_cluster_id = cluster_id

        # Skip if no inside clusters (all cluster_id == 0)
        if sample_max_cluster_id == 0:
            continue

        # Use arrays for Numba compatibility
        cluster_sums = np.zeros(sample_max_cluster_id + 1, dtype=np.float64)

        # Count clusters and calculate sums
        for data_idx in range(start_idx, end_idx):
            cluster_id = int(data[data_idx])
            bit_pos = int(indices[data_idx])

            if cluster_id > 0:
                if calculate_llrs and bit_pos < len(bit_llrs):
                    cluster_sums[cluster_id] += bit_llrs[bit_pos]
                elif not calculate_llrs:
                    cluster_sums[cluster_id] += 1.0

        # Calculate norm for inside clusters
        norm_val = 0.0
        if norm_order == 1.0:
            for cluster_id in range(1, sample_max_cluster_id + 1):
                norm_val += abs(cluster_sums[cluster_id])
        elif norm_order == 2.0:
            sum_sq = 0.0
            for cluster_id in range(1, sample_max_cluster_id + 1):
                val = cluster_sums[cluster_id]
                sum_sq += val * val
            norm_val = np.sqrt(sum_sq)
        elif np.isinf(norm_order):
            for cluster_id in range(1, sample_max_cluster_id + 1):
                abs_val = abs(cluster_sums[cluster_id])
                if abs_val > norm_val:
                    norm_val = abs_val
        elif norm_order == 0.5:
            sum_sqrt = 0.0
            for cluster_id in range(1, sample_max_cluster_id + 1):
                if cluster_sums[cluster_id] > 0:
                    sum_sqrt += np.sqrt(abs(cluster_sums[cluster_id]))
            norm_val = sum_sqrt * sum_sqrt
        else:  # Generic case
            sum_pow = 0.0
            for cluster_id in range(1, sample_max_cluster_id + 1):
                if cluster_sums[cluster_id] > 0:
                    sum_pow += abs(cluster_sums[cluster_id]) ** norm_order
            norm_val = sum_pow ** (1.0 / norm_order)

        inside_norms[i] = norm_val

    return inside_norms, outside_values


def _calculate_cluster_norms_from_csr_numba(
    clusters: csr_matrix, bit_llrs: np.ndarray, norm_order: float, calculate_llrs: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate cluster norms from a CSR sparse matrix representation of clusters.

    This is a backward compatibility wrapper for the unified function.
    """
    return calculate_cluster_metrics_from_csr(
        clusters,
        "norm",
        bit_llrs=bit_llrs,
        norm_order=norm_order,
        calculate_llrs=calculate_llrs,
    )


@numba.njit(fastmath=True, cache=True)
def _numba_inv_entropy_kernel(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    num_samples: int,
    num_bits: int,
    priors: np.ndarray,
) -> np.ndarray:
    """
    Core JIT-compiled function for calculating cluster inverse entropy sums from CSR
    matrix components.

    Parameters
    ----------
    data : 1D numpy array of float
        CSR matrix data array containing cluster IDs.
    indices : 1D numpy array of int
        CSR matrix indices array containing bit positions.
    indptr : 1D numpy array of int
        CSR matrix index pointer array for sample boundaries.
    num_samples : int
        Number of samples (rows) in the CSR matrix.
    num_bits : int
        Number of bits (columns) in the CSR matrix.
    priors : 1D numpy array of float
        Prior probabilities for each bit position.

    Returns
    -------
    cluster_inv_entropy_sums : 1D numpy array of float
        Sum of inverse entropies for all inside clusters for each sample.
    """
    cluster_inv_entropy_sums = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        start_idx = indptr[i]
        end_idx = indptr[i + 1]

        # Skip if no data for this sample
        if end_idx <= start_idx:
            continue

        # Find maximum cluster ID for this sample
        sample_max_cluster_id = 0
        for data_idx in range(start_idx, end_idx):
            cluster_id = int(data[data_idx])
            if cluster_id > sample_max_cluster_id:
                sample_max_cluster_id = cluster_id

        # Skip if no inside clusters (all cluster_id == 0)
        if sample_max_cluster_id == 0:
            continue

        # Calculate entropy sum for each inside cluster
        for cluster_id in range(1, sample_max_cluster_id + 1):
            inv_entropy_sum = 0.0

            # Sum entropies of bits in this cluster
            for data_idx in range(start_idx, end_idx):
                if int(data[data_idx]) == cluster_id:
                    bit_pos = int(indices[data_idx])
                    if bit_pos < len(priors):
                        p = priors[bit_pos]
                        # Clamp probability to avoid log(0)
                        if p <= 0.0:
                            p = 1e-15
                        elif p >= 1.0:
                            p = 1.0 - 1e-15

                        # Calculate entropy: -p * log(p) - (1-p) * log(1-p)
                        inv_entropy = 1 / (-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))
                        inv_entropy_sum += inv_entropy

            cluster_inv_entropy_sums[i] += inv_entropy_sum

    return cluster_inv_entropy_sums


def _calculate_cluster_inv_entropies_from_csr(
    clusters: csr_matrix, priors: np.ndarray
) -> np.ndarray:
    """
    Calculate cluster inverse entropy sums from a CSR sparse matrix representation of clusters.

    This is a backward compatibility wrapper for the unified function.
    """
    return calculate_cluster_metrics_from_csr(clusters, "inv_entropy", priors=priors)


@numba.njit(fastmath=True, cache=True)
def _numba_inv_priors_kernel(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    num_samples: int,
    num_bits: int,
    priors: np.ndarray,
) -> np.ndarray:
    """
    Core JIT-compiled function for calculating cluster inverse prior sums from CSR
    matrix components.

    Parameters
    ----------
    data : 1D numpy array of float
        CSR matrix data array containing cluster IDs.
    indices : 1D numpy array of int
        CSR matrix indices array containing bit positions.
    indptr : 1D numpy array of int
        CSR matrix index pointer array for sample boundaries.
    num_samples : int
        Number of samples (rows) in the CSR matrix.
    num_bits : int
        Number of bits (columns) in the CSR matrix.
    priors : 1D numpy array of float
        Prior probabilities for each bit position.

    Returns
    -------
    cluster_inv_entropy_sums : 1D numpy array of float
        Sum of inverse prior probabilities for all inside clusters for each sample.
    """
    cluster_inv_prior_sums = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        start_idx = indptr[i]
        end_idx = indptr[i + 1]

        # Skip if no data for this sample
        if end_idx <= start_idx:
            continue

        # Find maximum cluster ID for this sample
        sample_max_cluster_id = 0
        for data_idx in range(start_idx, end_idx):
            cluster_id = int(data[data_idx])
            if cluster_id > sample_max_cluster_id:
                sample_max_cluster_id = cluster_id

        # Skip if no inside clusters (all cluster_id == 0)
        if sample_max_cluster_id == 0:
            continue

        # Calculate entropy sum for each inside cluster
        for cluster_id in range(1, sample_max_cluster_id + 1):
            inv_prior_sum = 0.0

            # Sum entropies of bits in this cluster
            for data_idx in range(start_idx, end_idx):
                if int(data[data_idx]) == cluster_id:
                    bit_pos = int(indices[data_idx])
                    if bit_pos < len(priors):
                        p = priors[bit_pos]
                        # Clamp probability to avoid log(0)
                        if p <= 0.0:
                            p = 1e-15
                        elif p >= 1.0:
                            p = 1.0 - 1e-15

                        inv_prior_sum += 1 / p

            cluster_inv_prior_sums[i] += inv_prior_sum

    return cluster_inv_prior_sums


def _calculate_cluster_inv_priors_from_csr(
    clusters: csr_matrix, priors: np.ndarray
) -> np.ndarray:
    """
    Calculate cluster inverse prior probability sums from a CSR sparse matrix representation of clusters.

    This is a backward compatibility wrapper for the unified function.
    """
    return calculate_cluster_metrics_from_csr(clusters, "inv_prior_sum", priors=priors)


def get_cluster_size_distribution_from_csr(
    clusters: csr_matrix,
    precompile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distribution of cluster sizes from a CSR sparse matrix representation.

    Parameters
    ----------
    clusters : scipy sparse CSR matrix
        Sparse matrix where rows represent samples and columns represent bits.
        Non-zero values contain cluster IDs, with 0 representing outside clusters.
    precompile : bool, default False
        If True, pre-compiles the required Numba kernel if it's not already compiled.

    Returns
    -------
    cluster_sizes : 1D numpy array of int
        A sorted array of unique cluster sizes found across all samples.
    total_numbers : 1D numpy array of int
        An array where total_numbers[i] is the total count of clusters
        having the size specified in cluster_sizes[i].

    Notes
    -----
    This function processes cluster information for multiple samples, where each sample's
    cluster data is represented as a row in the sparse matrix. Only non-zero cluster IDs
    are considered (cluster ID 0 represents outside clusters and is ignored).
    """
    if clusters.nnz == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return _precompile_and_run_numba_kernel(
        _get_cluster_size_distribution_numba_kernel,
        precompile,
        clusters.data,
        clusters.indices,
        clusters.indptr,
    )


@numba.njit(fastmath=True, cache=True)
def _get_cluster_size_distribution_numba_kernel(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distribution of cluster sizes from the CSR properties of a matrix.

    This function processes cluster information for multiple samples, where each sample's
    cluster data is represented as a row in a conceptual sparse matrix. It is highly
    optimized with Numba for performance.

    Parameters
    ----------
    data : np.ndarray
        The 'data' array of the CSR matrix. It contains the cluster indices (non-zero values).
    indices : np.ndarray
        The 'indices' array of the CSR matrix. It contains the column indices for the 'data' values.
        (This parameter is not directly used in the logic but is part of the CSR format).
    indptr : np.ndarray
        The 'indptr' array of the CSR matrix. It defines the start and end points for each sample (row)
        in the 'data' and 'indices' arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1D NumPy arrays: (cluster_sizes, total_numbers).
        - cluster_sizes: A sorted array of unique cluster sizes.
        - total_numbers: An array where total_numbers[i] is the total count of clusters
                         having the size specified in cluster_sizes[i].
    """
    # Dictionary to store the final distribution: {size: count}
    # Using numba.typed.Dict for JIT compilation.
    size_distribution = numba.typed.Dict.empty(
        key_type=numba.core.types.int64,
        value_type=numba.core.types.int64,
    )

    num_samples = len(indptr) - 1

    # Iterate over each sample (row)
    for i in range(num_samples):
        start = indptr[i]
        end = indptr[i + 1]

        # If the sample has no clusters, skip it
        if start == end:
            continue

        # Dictionary to count cluster sizes for the current sample: {cluster_id: size}
        counts_in_sample = numba.typed.Dict.empty(
            key_type=numba.core.types.int64,
            value_type=numba.core.types.int64,
        )

        # Get all cluster IDs for the current sample
        sample_data = data[start:end]

        # Count the size of each cluster in this sample
        for cluster_id in sample_data:
            counts_in_sample[cluster_id] = counts_in_sample.get(cluster_id, 0) + 1

        # For each size found, update the global distribution count
        for size in counts_in_sample.values():
            size_distribution[size] = size_distribution.get(size, 0) + 1

    # If no clusters were found at all, return empty arrays
    if not size_distribution:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Convert the distribution dictionary to two NumPy arrays
    n_unique_sizes = len(size_distribution)
    cluster_sizes = np.empty(n_unique_sizes, dtype=np.int64)
    total_numbers = np.empty(n_unique_sizes, dtype=np.int64)

    for i, (size, count) in enumerate(size_distribution.items()):
        cluster_sizes[i] = size
        total_numbers[i] = count

    # Sort the results by cluster size
    sort_indices = np.argsort(cluster_sizes)

    return cluster_sizes[sort_indices], total_numbers[sort_indices]
