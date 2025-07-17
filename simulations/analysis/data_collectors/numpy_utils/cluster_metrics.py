import numba
import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix


def calculate_cluster_metrics_from_csr(
    clusters: csr_matrix,
    method: str,
    priors: np.ndarray | None = None,
    norm_order: float = 2.0,
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

        return _numba_cluster_norm_kernel(
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

        return _numba_inv_entropy_kernel(
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

        return _numba_inv_priors_kernel(
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


def calculate_cluster_norms_from_flat_data(
    flat_data: np.ndarray,  # Any numeric array type
    offsets: np.ndarray,  # Any numeric array type (will be converted to int)
    norm_order: float,
    sample_indices: np.ndarray | None = None,  # Optional sample filtering
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
        return _calculate_cluster_norms_numba_kernel(
            flat_data,
            offsets,
            norm_order,
        )
    else:
        return _calculate_cluster_norms_filtered_numba_kernel(
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


@numba.njit(fastmath=True, cache=True, parallel=True)
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
    Core JIT-compiled function for calculating cluster norms from CSR matrix components using optimized single-pass algorithm.

    This function processes cluster data in a single pass per sample, calculating both inside cluster norms
    and outside cluster values efficiently. It uses parallel processing across samples and optimized
    memory access patterns for better performance.

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
        The order p for the L_p norm calculation (can be np.inf).
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

    # Calculate total LLR sum for outside value computation
    total_llr_sum = 0.0
    if calculate_llrs:
        total_llr_sum = bit_llrs.sum()

    # Main processing loop - parallel across samples
    for row in numba.prange(num_samples):
        start_idx = indptr[row]
        end_idx = indptr[row + 1]
        nnz_row = end_idx - start_idx

        # Initialize temporary variables for this sample
        explicit_zero_cnt = 0
        implicit_zero_cnt = num_bits - nnz_row
        inside_llr_sum = 0.0  # Total LLR sum for cluster_id > 0
        sample_max_cluster = 0

        # Scratch array for cluster sums - reused across samples
        cluster_sums = np.zeros(max_cluster_id + 1, dtype=np.float64)

        # Single pass over the current CSR row
        for idx in range(start_idx, end_idx):
            cid = int(data[idx])
            bpos = int(indices[idx])

            if cid == 0:
                # Explicit outside cluster
                explicit_zero_cnt += 1
                # Outside LLR is computed as total_sum - inside_sum, so no need to add LLR here
                continue

            # Inside cluster processing
            if calculate_llrs and bpos < len(bit_llrs):
                val = bit_llrs[bpos]
                cluster_sums[cid] += val
                inside_llr_sum += val
            else:
                cluster_sums[cid] += 1.0  # Count mode

            if cid > sample_max_cluster:
                sample_max_cluster = cid

        # Calculate outside values
        if calculate_llrs:
            outside_values[row] = total_llr_sum - inside_llr_sum
        else:
            outside_values[row] = explicit_zero_cnt + implicit_zero_cnt

        # Calculate inside norm
        if sample_max_cluster == 0:  # All values are outside clusters, norm = 0
            continue

        norm_val = 0.0
        if norm_order == 1.0:
            # L1 norm: sum of absolute values
            for cid in range(1, sample_max_cluster + 1):
                norm_val += abs(cluster_sums[cid])

        elif norm_order == 2.0:
            # L2 norm: square root of sum of squares
            sum_sq = 0.0
            for cid in range(1, sample_max_cluster + 1):
                v = cluster_sums[cid]
                sum_sq += v * v
            norm_val = np.sqrt(sum_sq)

        elif np.isinf(norm_order):
            # L∞ norm: maximum absolute value
            for cid in range(1, sample_max_cluster + 1):
                v = abs(cluster_sums[cid])
                if v > norm_val:
                    norm_val = v

        elif norm_order == 0.5:
            # Special case for p = 0.5: optimized sqrt handling
            sum_sqrt = 0.0
            for cid in range(1, sample_max_cluster + 1):
                v = cluster_sums[cid]
                if v > 0:
                    sum_sqrt += np.sqrt(v)
            norm_val = sum_sqrt * sum_sqrt

        else:
            # Generic L_p norm case
            sum_pow = 0.0
            for cid in range(1, sample_max_cluster + 1):
                v = cluster_sums[cid]
                if v > 0:
                    sum_pow += abs(v) ** norm_order
            norm_val = sum_pow ** (1.0 / norm_order)

        inside_norms[row] = norm_val

    return inside_norms, outside_values


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


def calculate_cluster_inv_entropies_from_csr(
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


def calculate_cluster_inv_priors_from_csr(
    clusters: csr_matrix, priors: np.ndarray
) -> np.ndarray:
    """
    Calculate cluster inverse prior probability sums from a CSR sparse matrix representation of clusters.

    This is a backward compatibility wrapper for the unified function.
    """
    return calculate_cluster_metrics_from_csr(clusters, "inv_prior_sum", priors=priors)
