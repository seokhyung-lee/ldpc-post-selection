from typing import Tuple

import numpy as np


def compute_cluster_stats(
    clusters: np.ndarray, llrs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cluster statistics from cluster assignments using numpy native functions.

    Parameters
    ----------
    clusters : 1D numpy array of int
        Cluster assignments for each bit (0 = outside cluster, 1+ = cluster ID).
    llrs : 1D numpy array of float
        Log-likelihood ratios for each bit.

    Returns
    -------
    cluster_sizes : 1D numpy array of int
        Size of each cluster (index corresponds to cluster ID).
    cluster_llrs : 1D numpy array of float
        Sum of LLRs for each cluster (index corresponds to cluster ID).
    """
    max_cluster_id = clusters.max()

    # Use bincount to efficiently compute cluster sizes and LLR sums
    cluster_sizes = np.bincount(clusters, minlength=max_cluster_id + 1).astype(np.int_)
    cluster_llrs = np.bincount(clusters, weights=llrs, minlength=max_cluster_id + 1)

    return cluster_sizes, cluster_llrs
