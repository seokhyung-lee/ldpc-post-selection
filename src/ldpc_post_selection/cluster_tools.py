from typing import Tuple

import numba
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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


def label_clusters(
    adj_matrix: np.ndarray, vertices_inside_clusters: np.ndarray
) -> np.ndarray:
    """
    Label connected components (clusters) in an adjacency matrix for specified vertices.

    Parameters
    ----------
    adj_matrix : 2D numpy array of bool with shape (N, N)
        Boolean adjacency matrix of an undirected graph.
    vertices_inside_clusters : 1D numpy array of int
        Array of vertex indices that should be considered for clustering.

    Returns
    -------
    cluster_idx : 1D numpy array of int with shape (N,)
        Cluster label for each vertex (1, 2, ...). Vertices not in clusters have value 0.
    """
    # Input validation
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if len(vertices_inside_clusters) > 0 and (
        vertices_inside_clusters.max() >= adj_matrix.shape[0]
        or vertices_inside_clusters.min() < 0
    ):
        raise ValueError("inside_indices contains invalid vertex indices")

    if len(vertices_inside_clusters) == 0:
        return np.zeros(adj_matrix.shape[0], dtype=int)

    # Create subgraph adjacency matrix for inside vertices only using advanced indexing
    sub_adj = adj_matrix[np.ix_(vertices_inside_clusters, vertices_inside_clusters)]

    # Convert to sparse matrix and find connected components using scipy
    sparse_adj = csr_matrix(sub_adj)
    n_components, labels = connected_components(sparse_adj, directed=False)

    # Map back to original vertex indices using vectorized assignment
    cluster_idx = np.zeros(adj_matrix.shape[0], dtype=int)
    cluster_idx[vertices_inside_clusters] = labels + 1  # +1 to make labels start from 1

    return cluster_idx
