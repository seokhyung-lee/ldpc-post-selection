from typing import Tuple

import numba
import numpy as np
import igraph as ig
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


def compute_cluster_norm_fraction(values: np.ndarray, order: float) -> float:
    """
    Compute norm fraction for given values and order.

    Parameters
    ----------
    values : 1D numpy array of float
        Values to compute norm fraction for (e.g., cluster sizes or LLRs).
        values[0] is assumed to be the outside region and is excluded.
    order : float
        Order for norm computation (can be a positive number or `np.inf`).

    Returns
    -------
    norm_fraction : float
        Norm fraction value for the given order.
    """
    # Get values excluding the outside region (index 0)
    inside_values = values[1:] if values.size > 1 else np.array([], dtype=values.dtype)
    if inside_values.size == 0:
        return 0.0

    total_sum = float(np.sum(values))
    if total_sum == 0.0:
        return 0.0

    inside_norm = compute_lp_norm(inside_values.astype(float, copy=False), order)
    return inside_norm / total_sum


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


def label_clusters_igraph(
    full_graph: ig.Graph, vertices_inside_clusters: np.ndarray
) -> np.ndarray:
    """
    Label connected components (clusters) using python-igraph with pre-created graph.

    This function uses a pre-created igraph Graph and extracts subgraphs efficiently,
    eliminating the need to recreate the graph for each sample. This provides
    significant performance improvements over both the original scipy implementation
    and the previous igraph implementation that recreated graphs.

    Parameters
    ----------
    full_graph : igraph.Graph
        Pre-created igraph Graph object representing the full adjacency structure.
    vertices_inside_clusters : 1D numpy array of int
        Array of vertex indices that should be considered for clustering.

    Returns
    -------
    cluster_idx : 1D numpy array of int with shape (N,)
        Cluster label for each vertex (1, 2, ...). Vertices not in clusters have value 0.
    """
    # Input validation
    if len(vertices_inside_clusters) > 0 and (
        vertices_inside_clusters.max() >= full_graph.vcount()
        or vertices_inside_clusters.min() < 0
    ):
        raise ValueError("inside_indices contains invalid vertex indices")

    if len(vertices_inside_clusters) == 0:
        return np.zeros(full_graph.vcount(), dtype=int)

    # Extract subgraph using igraph's native and efficient subgraph method
    # This is much faster than recreating the graph from adjacency matrix
    subgraph = full_graph.subgraph(vertices_inside_clusters)

    # Find connected components directly on the igraph subgraph
    components = subgraph.connected_components()

    # Prepare result array
    cluster_idx = np.zeros(full_graph.vcount(), dtype=int)

    # Convert membership list to NumPy array and make 1-based labels
    membership = np.asarray(components.membership, dtype=int) + 1

    # Vectorized assignment - much faster than Python for loop
    cluster_idx[vertices_inside_clusters] = membership

    return cluster_idx


def compute_lp_norm(values: np.ndarray, order: float, take_abs: bool = False) -> float:
    """
    Compute an L_p norm for 1D values with optional absolute values.

    Parameters
    ----------
    values : 1D numpy array of float
        Values for which the norm should be computed.
    order : float
        Order for the L_p norm (positive number or `np.inf`).
    take_abs : bool, optional
        If True, take absolute values before the norm calculation. Defaults to False.

    Returns
    -------
    norm_value : float
        Calculated norm of the provided values.
    """
    if values.size == 0:
        return 0.0

    processed = np.abs(values) if take_abs else values

    if processed.size == 0:
        return 0.0

    if order == 1:
        return float(np.sum(processed))
    if order == 2:
        return float(np.sqrt(np.sum(processed**2)))
    if np.isinf(order):
        return float(np.max(processed)) if processed.size > 0 else 0.0

    return float(np.sum(processed**order) ** (1.0 / order))
