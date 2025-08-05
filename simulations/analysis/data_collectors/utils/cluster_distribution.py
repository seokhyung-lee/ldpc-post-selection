import numba
import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix


def get_cluster_size_distribution_from_csr(
    clusters: csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distribution of cluster sizes from a CSR sparse matrix representation.

    Parameters
    ----------
    clusters : scipy sparse CSR matrix
        Sparse matrix where rows represent samples and columns represent bits.
        Non-zero values contain cluster IDs, with 0 representing outside clusters.

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

    return _get_cluster_size_distribution_numba_kernel(
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
