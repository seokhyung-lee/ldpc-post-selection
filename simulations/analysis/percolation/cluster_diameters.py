from tqdm import tqdm
import numpy as np
import igraph as ig
from scipy.sparse import csc_matrix, csr_matrix
from itertools import combinations
from joblib import Parallel, delayed


class ClusterDiameterCalculator:
    """
    Calculates the diameter of specified clusters of nodes in a decoding graph.

    The decoding graph is constructed from a binary sparse matrix H, where
    columns represent nodes and rows define hyperedges (or cliques).
    """

    def __init__(self, H: csc_matrix):
        """Initializes the ClusterDiameter class and constructs the decoding graph.

        Parameters
        ----------
        H : binary scipy.sparse.csc_matrix of shape (M, N)
            The parity-check matrix. Columns correspond to nodes, and rows define
            the hyperedges that connect them.
        """
        if not isinstance(H, csc_matrix):
            raise TypeError("H must be a scipy.sparse.csc_matrix.")

        self.H = H
        self.num_nodes = H.shape[1]
        self.decoding_graph = self._build_graph()

    def _build_graph(self) -> ig.Graph:
        """Builds the decoding graph from the parity check matrix H.

        In this graph, each column of H is a node. For each row of H,
        the nodes corresponding to columns with nonzero entries are
        connected to form a clique.

        Returns
        -------
        graph : igraph.Graph
            The constructed decoding graph.
        """
        graph = ig.Graph(n=self.num_nodes, directed=False)
        H_csr = self.H.tocsr()

        edges = []
        for i in range(H_csr.shape[0]):
            nodes_in_row = H_csr.indices[H_csr.indptr[i] : H_csr.indptr[i + 1]]
            for edge in combinations(nodes_in_row, 2):
                edges.append(edge)

        if edges:
            graph.add_edges(edges)
            graph.simplify()  # Remove duplicate edges and loops

        return graph

    def compute_cluster_diameters(
        self, clusters: np.ndarray | csr_matrix
    ) -> dict[int, int]:
        """Computes the diameter for each cluster of nodes.

        The diameter of a cluster is defined as the diameter of the subgraph
        induced by the nodes belonging to that cluster.

        Parameters
        ----------
        clusters : 1D numpy array of int with shape (N,) or scipy.sparse.csr_matrix of shape (1, N)
            An array where each element is an integer indicating the cluster ID
            for the corresponding node. A value of 0 indicates that the node
            is not in any cluster. For `csr_matrix`, only non-zero cluster IDs are stored.

        Returns
        -------
        diameters : dict[int, int]
            A dictionary mapping each cluster ID (keys) to its
            computed diameter (values).
        """
        if isinstance(clusters, csr_matrix):
            if clusters.shape[0] != 1 or clusters.shape[1] != self.num_nodes:
                raise ValueError(
                    f"If `clusters` is a csr_matrix, its shape must be (1, {self.num_nodes}). Got {clusters.shape}"
                )

            cluster_ids = np.unique(clusters.data)

            def get_node_indices(c_id):
                return clusters.indices[clusters.data == c_id].tolist()

        elif isinstance(clusters, np.ndarray):
            if clusters.ndim != 1:
                raise TypeError("If `clusters` is a numpy array, it must be 1D.")
            if clusters.shape[0] != self.num_nodes:
                raise ValueError(
                    f"The length of `clusters` must be equal to {self.num_nodes}. Got {clusters.shape[0]}."
                )

            cluster_ids = np.unique(clusters)
            cluster_ids = cluster_ids[cluster_ids > 0]

            def get_node_indices(c_id):
                return np.where(clusters == c_id)[0].tolist()

        else:
            raise TypeError(
                "`clusters` must be a 1D numpy array or a scipy.sparse.csr_matrix."
            )

        diameters = {}
        for c_id in cluster_ids:
            node_indices = get_node_indices(c_id)

            if not node_indices:
                diameters[int(c_id)] = 0
                continue

            subgraph = self.decoding_graph.subgraph(node_indices)

            if subgraph.vcount() > 0:
                diameters[int(c_id)] = subgraph.diameter(unconn=True)
            else:
                diameters[int(c_id)] = 0

        return diameters

    def _compute_max_diameter_for_sample(
        self, clusters: np.ndarray | csr_matrix
    ) -> int:
        """Computes the maximum diameter for a single sample of clusters."""
        diameters_dict = self.compute_cluster_diameters(clusters)
        if not diameters_dict:
            return 0
        return max(diameters_dict.values())

    def compute_cluster_diameters_batch(
        self,
        clusters: np.ndarray | csr_matrix,
        n_jobs: int = 1,
        use_tqdm: bool = False,
        return_max_diameter: bool = False,
    ) -> list[dict[int, int]] | list[int]:
        """Computes cluster diameters for multiple samples in batch.

        This method processes each row of the `clusters` input as a separate sample
        and computes the cluster diameters for it, with optional parallelization
        using joblib.

        Parameters
        ----------
        clusters : 2D numpy array of int or 2D scipy.sparse.csr_matrix
            A collection of cluster assignments, with shape (num_samples, N),
            where N is the number of nodes. Each row represents a single sample.
        n_jobs : int, optional
            The number of jobs to run in parallel. If -1, all available CPUs are
            used. Defaults to 1 (no parallelization).
        use_tqdm : bool, optional
            Whether to use tqdm to display progress. Defaults to False.
        return_max_diameter : bool, optional
            If True, returns only the maximum diameter for each sample.
            Defaults to False.

        Returns
        -------
        list_of_diameters : list[dict[int, int]] | list[int]
            If `return_max_diameter` is False, a list where each element is a
            dictionary of cluster diameters for the corresponding sample.
            If `return_max_diameter` is True, a list of the maximum diameter
            for each sample.
        """
        if not isinstance(clusters, (np.ndarray, csr_matrix)):
            raise TypeError(
                "`clusters` must be a 2D numpy array or a scipy.sparse.csr_matrix."
            )

        if clusters.ndim != 2:
            raise ValueError("`clusters` must be a 2D array or matrix.")

        if clusters.shape[1] != self.num_nodes:
            raise ValueError(
                f"The number of columns in `clusters` ({clusters.shape[1]}) must match "
                f"the number of nodes in H ({self.num_nodes})."
            )

        num_samples = clusters.shape[0]

        if num_samples == 0:
            return []

        target_func = (
            self._compute_max_diameter_for_sample
            if return_max_diameter
            else self.compute_cluster_diameters
        )

        iterator = range(num_samples)
        if use_tqdm:
            iterator = tqdm(list(iterator))

        results = Parallel(n_jobs=n_jobs)(
            delayed(target_func)(clusters[i]) for i in iterator
        )

        if results is None:
            return []

        return results
