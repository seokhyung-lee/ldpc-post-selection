import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple

from ..toric_code_bitflip_simulation import (
    generate_toric_code,
)
from .numpy_utils import (
    calculate_cluster_metrics_from_csr,
    get_cluster_size_distribution_from_csr,
)
from .percolation_utils import (
    calculate_com_periodic,
    calculate_distance_periodic,
)


class ToricCodePercolation:
    """
    A class to analyze percolation properties of a toric code.

    This class encapsulates the structure of a toric code of a given distance `d`
    and provides methods to check for spanning clusters of qubits.

    Parameters
    ----------
    d : int
        The code distance of the toric code.
    """

    def __init__(self, d: int):
        """
        Initializes the ToricCodePercolation instance.
        """
        if not isinstance(d, int) or d <= 0:
            raise ValueError("Code distance 'd' must be a positive integer.")

        self.d = d
        self.hz, _ = generate_toric_code(d)
        self.matching_graph = self._build_matching_graph()

        # in the unit of half lattice spacing
        self.qubit_coordinates = self._compute_qubit_coordinates()

    def _build_matching_graph(self) -> ig.Graph:
        """
        Builds the matching graph from the toric code check matrix.

        The matching graph has checks as vertices and qubits as edges. It also
        annotates certain edges with flags 'H' and 'V' to identify non-trivial
        loops on the torus.

        Returns
        -------
        igraph.Graph
            The constructed matching graph.
        """
        n_checks, n_qubits = self.hz.shape
        matching_graph = ig.Graph(n_checks, directed=False)

        rows, cols = self.hz.nonzero()
        qubit_to_checks = [[] for _ in range(n_qubits)]
        for r, c in zip(rows, cols):
            qubit_to_checks[c].append(r)

        edges = []
        qubit_indices_for_edges = []
        for q_idx, c_list in enumerate(qubit_to_checks):
            if len(c_list) == 2:
                edges.append(tuple(c_list))
                qubit_indices_for_edges.append(q_idx)

        attrs = {
            "qubit_index": qubit_indices_for_edges,
            "flag": None,
        }
        matching_graph.add_edges(edges, attrs)

        h_flag_indices = {i * self.d for i in range(self.d)}
        v_flag_indices = {self.d * self.d + i for i in range(self.d)}
        matching_graph.es.select(qubit_index_in=h_flag_indices)["flag"] = "H"
        matching_graph.es.select(qubit_index_in=v_flag_indices)["flag"] = "V"

        return matching_graph

    def _compute_qubit_coordinates(self) -> np.ndarray:
        """
        Computes the coordinates (in the unit of half lattice spacing) for all qubits in
        the toric code.

        Returns
        -------
        numpy array of shape (num_qubits, 2)
            Array where each row contains the (x, y) coordinates of a qubit.
            For qubit index i:
            - If i < d**2: coordinates are (2 * (i % d) + 1, i // d)
            - If i >= d**2: coordinates are (2 * (i % d) + 2, 2 * (i // d - d) + 1)
        """
        num_qubits = self.hz.shape[1]
        coordinates = np.zeros((num_qubits, 2), dtype=int)

        # Vectorized computation for all qubit indices
        qubit_indices = np.arange(num_qubits)

        # Split into two groups based on the condition i < d**2
        mask_first_group = qubit_indices < self.d**2
        mask_second_group = ~mask_first_group

        # First group: i < d**2
        first_group_indices = qubit_indices[mask_first_group]
        coordinates[mask_first_group, 0] = 2 * (first_group_indices % self.d) + 1
        coordinates[mask_first_group, 1] = first_group_indices // self.d

        # Second group: i >= d**2
        second_group_indices = qubit_indices[mask_second_group]
        coordinates[mask_second_group, 0] = 2 * (second_group_indices % self.d) + 2
        coordinates[mask_second_group, 1] = (
            2 * (second_group_indices // self.d - self.d) + 1
        )

        return coordinates

    def _extract_clusters_from_config(
        self, cluster_config: np.ndarray | csr_matrix
    ) -> dict[int, np.ndarray]:
        """
        Extracts cluster information from a cluster configuration.

        Parameters
        ----------
        cluster_config : 1D numpy array or scipy csr_matrix
            Cluster configuration where values >= 1 indicate the cluster index
            that each qubit belongs to, and 0 indicates qubits outside clusters.

        Returns
        -------
        dict[int, numpy array]
            Dictionary mapping cluster indices to arrays of qubit indices in each cluster.
        """
        if isinstance(cluster_config, csr_matrix):
            if cluster_config.shape[0] != 1:
                raise ValueError("csr_matrix should have shape (1, num_qubits)")

            # Use csr_matrix internal properties directly
            qubit_indices = cluster_config.indices
            cluster_values = cluster_config.data

            # Build clusters using sparse structure
            clusters = {}
            for qubit_idx, cluster_idx in zip(qubit_indices, cluster_values):
                cluster_idx = int(cluster_idx)
                if cluster_idx not in clusters:
                    clusters[cluster_idx] = []
                clusters[cluster_idx].append(qubit_idx)

            # Convert lists to numpy arrays
            for cluster_idx in clusters:
                clusters[cluster_idx] = np.array(clusters[cluster_idx], dtype=int)

        else:
            # Handle numpy array
            cluster_array = cluster_config.flatten()

            # Get unique cluster indices (excluding 0)
            cluster_indices = np.unique(cluster_array)
            cluster_indices = cluster_indices[cluster_indices > 0]

            # Build clusters using vectorized operations
            clusters = {}
            for cluster_idx in cluster_indices:
                qubit_cluster = np.where(cluster_array == cluster_idx)[0]
                if len(qubit_cluster) > 0:  # Skip empty clusters
                    clusters[int(cluster_idx)] = qubit_cluster

        return clusters

    @staticmethod
    def _check_for_spanning_cycles(graph: ig.Graph) -> bool:
        """
        Checks for the presence of a spanning cycle in a graph.

        A cycle is spanning if it intersects a non-trivial loop of the torus
        an odd number of times. This is checked by counting edges flagged
        as 'H' or 'V'.

        Parameters
        ----------
        graph : igraph.Graph
            The graph (or subgraph) to check for spanning cycles.

        Returns
        -------
        bool
            True if a spanning cycle is found, False otherwise.
        """
        cycle_basis = graph.minimum_cycle_basis()

        if not cycle_basis:
            return False

        for cycle_edge_indices in cycle_basis:
            h_count = 0
            v_count = 0
            cycle_edges = graph.es[cycle_edge_indices]

            for flag in cycle_edges["flag"]:
                if flag == "H":
                    h_count += 1
                elif flag == "V":
                    v_count += 1

            if h_count % 2 != 0 or v_count % 2 != 0:
                return True

        return False

    def is_spanning_cluster(self, qubit_cluster: list[int]) -> bool:
        """
        Checks if a qubit cluster in the toric code forms a spanning cluster.

        Parameters
        ----------
        qubit_cluster : list of int
            A list of qubit indices forming the cluster.

        Returns
        -------
        bool
            True if the qubit cluster is a spanning cluster, False otherwise.
        """
        if not qubit_cluster:
            return False

        edges_in_cluster = self.matching_graph.es.select(qubit_index_in=qubit_cluster)

        subgraph: ig.Graph = self.matching_graph.subgraph_edges(
            edges_in_cluster, delete_vertices=True
        )

        if not subgraph.is_connected():
            raise ValueError("qubit_cluster is not a connected cluster.")

        return self._check_for_spanning_cycles(subgraph)

    def check_percolation(self, cluster_config: np.ndarray | csr_matrix) -> bool:
        """
        Checks if a cluster configuration contains any spanning clusters.

        A configuration is percolating if it contains at least one spanning cluster.

        Parameters
        ----------
        cluster_config : 1D numpy array or scipy csr_matrix
            Cluster configuration where values >= 1 indicate the cluster index
            that each qubit belongs to, and 0 indicates qubits outside clusters.
            If numpy array: shape should be (num_qubits,)
            If csr_matrix: shape should be (1, num_qubits)

        Returns
        -------
        bool
            True if the configuration is percolating (contains at least one
            spanning cluster), False otherwise.
        """
        # Parse cluster configuration
        if isinstance(cluster_config, csr_matrix):
            if cluster_config.shape[0] != 1:
                raise ValueError("csr_matrix should have shape (1, num_qubits)")

            # Use csr_matrix internal properties directly
            qubit_indices = cluster_config.indices
            cluster_values = cluster_config.data

            # Get unique cluster indices
            cluster_indices = np.unique(cluster_values)

            # Build clusters using sparse structure
            clusters = {}
            for qubit_idx, cluster_idx in zip(qubit_indices, cluster_values):
                if cluster_idx not in clusters:
                    clusters[cluster_idx] = []
                clusters[cluster_idx].append(int(qubit_idx))

        else:
            # Handle numpy array
            cluster_array = cluster_config.flatten()

            # Get unique cluster indices (excluding 0)
            cluster_indices = np.unique(cluster_array)
            cluster_indices = cluster_indices[cluster_indices > 0]

            # Build clusters
            clusters = {}
            for cluster_idx in cluster_indices:
                qubit_cluster = np.where(cluster_array == cluster_idx)[0].tolist()
                if qubit_cluster:  # Skip empty clusters
                    clusters[cluster_idx] = qubit_cluster

        # Check each cluster for spanning property
        for cluster_idx, qubit_cluster in clusters.items():
            try:
                if self.is_spanning_cluster(qubit_cluster):
                    return True
            except ValueError:
                raise ValueError(f"Cluster {cluster_idx} is not connected.")

        return False

    def check_percolation_batch(
        self, cluster_configs: np.ndarray | csr_matrix
    ) -> list[bool]:
        """
        Checks if cluster configurations contain any spanning clusters for multiple samples.

        Parameters
        ----------
        cluster_configs : 2D numpy array or scipy csr_matrix
            Batch of cluster configurations where each row represents a sample.
            Values >= 1 indicate the cluster index that each qubit belongs to,
            and 0 indicates qubits outside clusters.
            Shape should be (num_samples, num_qubits)

        Returns
        -------
        list of bool
            List of boolean values indicating whether each sample configuration
            is percolating (contains at least one spanning cluster).
        """
        if isinstance(cluster_configs, csr_matrix):
            num_samples = cluster_configs.shape[0]
            results = []

            for sample_idx in range(num_samples):
                # Extract single row as csr_matrix
                sample_config = cluster_configs.getrow(sample_idx)
                results.append(self.check_percolation(sample_config))

        else:
            # Handle 2D numpy array
            if cluster_configs.ndim != 2:
                raise ValueError("cluster_configs must be a 2D array")

            results = []
            for sample_config in cluster_configs:
                results.append(self.check_percolation(sample_config))

        return results

    def calculate_kth_moment_estimation(
        self, cluster_config: np.ndarray | csr_matrix, k: float | int
    ) -> float:
        """
        Calculate sample estimation of the k-th moment of the cluster number distribution.

        This computes (Lk-norm of cluster sizes)^k / num_qubits where num_qubits
        is the total number of qubits in the toric code.

        Parameters
        ----------
        cluster_config : 1D numpy array or scipy csr_matrix
            Cluster configuration where values >= 1 indicate the cluster index
            that each qubit belongs to, and 0 indicates qubits outside clusters.
            If numpy array: shape should be (num_qubits,)
            If csr_matrix: shape should be (1, num_qubits)
        k : float or int
            The moment order k.

        Returns
        -------
        float
            Sample estimation of the k-th moment of the cluster number distribution.
        """
        # Ensure single sample format for calculate_cluster_metrics_from_csr
        if isinstance(cluster_config, np.ndarray):
            if cluster_config.ndim == 1:
                # Convert 1D array to 2D with single row
                cluster_csr = csr_matrix(cluster_config.reshape(1, -1))
            else:
                raise ValueError("numpy array must be 1D for single sample")
        else:
            # Already csr_matrix
            if cluster_config.shape[0] != 1:
                raise ValueError(
                    "csr_matrix should have shape (1, num_qubits) for single sample"
                )
            cluster_csr = cluster_config

        # Calculate Lk-norm using calculate_cluster_metrics_from_csr
        inside_norms, _ = calculate_cluster_metrics_from_csr(
            cluster_csr, method="norm", norm_order=k
        )

        # inside_norms is array with single element for single sample
        lk_norm = inside_norms[0]

        # Calculate k-th moment estimation: (Lk-norm)^k / num_qubits
        num_qubits = self.hz.shape[1]
        kth_moment_estimation = (lk_norm**k) / num_qubits

        return kth_moment_estimation

    def calculate_kth_moment_estimation_batch(
        self, cluster_configs: np.ndarray | csr_matrix, k: float | int
    ) -> np.ndarray:
        """
        Calculate sample estimation of the k-th moment of the cluster number distribution for multiple samples.

        This computes (Lk-norm of cluster sizes)^k / num_qubits for each sample.

        Parameters
        ----------
        cluster_configs : 2D numpy array or scipy csr_matrix
            Batch of cluster configurations where each row represents a sample.
            Values >= 1 indicate the cluster index that each qubit belongs to,
            and 0 indicates qubits outside clusters.
            Shape should be (num_samples, num_qubits)
        k : float or int
            The moment order k.

        Returns
        -------
        numpy array of float
            Array of k-th moment estimations for each sample.
        """
        # Ensure proper format for calculate_cluster_metrics_from_csr
        if isinstance(cluster_configs, np.ndarray):
            if cluster_configs.ndim != 2:
                raise ValueError("numpy array must be 2D for batch processing")
            cluster_csr = csr_matrix(cluster_configs)
        else:
            # Already csr_matrix
            cluster_csr = cluster_configs

        # Calculate Lk-norms for all samples using calculate_cluster_metrics_from_csr
        inside_norms, _ = calculate_cluster_metrics_from_csr(
            cluster_csr, method="norm", norm_order=k, precompile=True
        )

        # Calculate k-th moment estimation for each sample: (Lk-norm)^k / num_qubits
        num_qubits = self.hz.shape[1]
        kth_moment_estimations = (inside_norms**k) / num_qubits

        return kth_moment_estimations

    def calculate_cluster_number_density(
        self, cluster_config: np.ndarray | csr_matrix
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate cluster number density for a single cluster configuration.

        The cluster number density is calculated as total_numbers / num_qubits / num_samples,
        where total_numbers is the count of clusters for each cluster size.

        Parameters
        ----------
        cluster_config : 1D numpy array or scipy csr_matrix
            Cluster configuration where values >= 1 indicate the cluster index
            that each qubit belongs to, and 0 indicates qubits outside clusters.
            If numpy array: shape should be (num_qubits,)
            If csr_matrix: shape should be (1, num_qubits)

        Returns
        -------
        cluster_sizes : 1D numpy array of int
            A sorted array of unique cluster sizes found in the configuration.
        cluster_number_density : 1D numpy array of float
            Array where cluster_number_density[i] is the number density of clusters
            having the size specified in cluster_sizes[i].
        """
        # Ensure single sample format
        if isinstance(cluster_config, np.ndarray):
            if cluster_config.ndim == 1:
                # Convert 1D array to 2D with single row
                cluster_csr = csr_matrix(cluster_config.reshape(1, -1))
            else:
                raise ValueError("numpy array must be 1D for single sample")
        else:
            # Already csr_matrix
            if cluster_config.shape[0] != 1:
                raise ValueError(
                    "csr_matrix should have shape (1, num_qubits) for single sample"
                )
            cluster_csr = cluster_config

        # Get cluster size distribution
        cluster_sizes, total_numbers = get_cluster_size_distribution_from_csr(
            cluster_csr, precompile=False
        )

        # Calculate cluster number density: total_numbers / num_qubits / num_samples
        num_qubits = self.hz.shape[1]
        num_samples = 1  # Single sample
        cluster_number_density = (
            total_numbers.astype(np.float64) / num_qubits / num_samples
        )

        return cluster_sizes, cluster_number_density

    def calculate_cluster_number_density_batch(
        self, cluster_configs: np.ndarray | csr_matrix
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate cluster number density for multiple cluster configurations.

        The cluster number density is calculated as total_numbers / num_qubits / num_samples,
        where total_numbers is the count of clusters for each cluster size across all samples.

        Parameters
        ----------
        cluster_configs : 2D numpy array or scipy csr_matrix
            Batch of cluster configurations where each row represents a sample.
            Values >= 1 indicate the cluster index that each qubit belongs to,
            and 0 indicates qubits outside clusters.
            Shape should be (num_samples, num_qubits)

        Returns
        -------
        cluster_sizes : 1D numpy array of int
            A sorted array of unique cluster sizes found across all samples.
        cluster_number_density : 1D numpy array of float
            Array where cluster_number_density[i] is the number density of clusters
            having the size specified in cluster_sizes[i].
        """
        # Ensure proper format
        if isinstance(cluster_configs, np.ndarray):
            if cluster_configs.ndim != 2:
                raise ValueError("numpy array must be 2D for batch processing")
            cluster_csr = csr_matrix(cluster_configs)
        else:
            # Already csr_matrix
            cluster_csr = cluster_configs

        # Get cluster size distribution
        cluster_sizes, total_numbers = get_cluster_size_distribution_from_csr(
            cluster_csr, precompile=True
        )

        # Calculate cluster number density: total_numbers / num_qubits / num_samples
        num_qubits = self.hz.shape[1]
        num_samples = cluster_csr.shape[0]
        cluster_number_density = (
            total_numbers.astype(np.float64) / num_qubits / num_samples
        )

        return cluster_sizes, cluster_number_density

    def calculate_max_cluster_size(
        self, cluster_config: np.ndarray | csr_matrix
    ) -> int:
        """
        Calculate the maximum cluster size for a single cluster configuration.

        Parameters
        ----------
        cluster_config : 1D numpy array or scipy csr_matrix
            Cluster configuration where values >= 1 indicate the cluster index
            that each qubit belongs to, and 0 indicates qubits outside clusters.
            If numpy array: shape should be (num_qubits,)
            If csr_matrix: shape should be (1, num_qubits)

        Returns
        -------
        int
            The maximum cluster size in the configuration.
        """
        # Ensure single sample format for calculate_cluster_metrics_from_csr
        if isinstance(cluster_config, np.ndarray):
            if cluster_config.ndim == 1:
                # Convert 1D array to 2D with single row
                cluster_csr = csr_matrix(cluster_config.reshape(1, -1))
            else:
                raise ValueError("numpy array must be 1D for single sample")
        else:
            # Already csr_matrix
            if cluster_config.shape[0] != 1:
                raise ValueError(
                    "csr_matrix should have shape (1, num_qubits) for single sample"
                )
            cluster_csr = cluster_config

        # Calculate L∞-norm using calculate_cluster_metrics_from_csr
        # L∞-norm gives the maximum cluster size
        max_cluster_sizes, _ = calculate_cluster_metrics_from_csr(
            cluster_csr, method="norm", norm_order=np.inf
        )

        # max_cluster_sizes is array with single element for single sample
        max_cluster_size = int(max_cluster_sizes[0])

        return max_cluster_size

    def calculate_max_cluster_size_batch(
        self, cluster_configs: np.ndarray | csr_matrix
    ) -> np.ndarray:
        """
        Calculate the maximum cluster size for multiple cluster configurations.

        Parameters
        ----------
        cluster_configs : 2D numpy array or scipy csr_matrix
            Batch of cluster configurations where each row represents a sample.
            Values >= 1 indicate the cluster index that each qubit belongs to,
            and 0 indicates qubits outside clusters.
            Shape should be (num_samples, num_qubits)

        Returns
        -------
        numpy array of int
            Array of maximum cluster sizes for each sample.
        """
        # Ensure proper format for calculate_cluster_metrics_from_csr
        if isinstance(cluster_configs, np.ndarray):
            if cluster_configs.ndim != 2:
                raise ValueError("numpy array must be 2D for batch processing")
            cluster_csr = csr_matrix(cluster_configs)
        else:
            # Already csr_matrix
            cluster_csr = cluster_configs

        # Calculate L∞-norms for all samples using calculate_cluster_metrics_from_csr
        # L∞-norm gives the maximum cluster size for each sample
        max_cluster_sizes, _ = calculate_cluster_metrics_from_csr(
            cluster_csr, method="norm", norm_order=np.inf, precompile=True
        )

        # Convert to integers
        max_cluster_sizes = max_cluster_sizes.astype(int)

        return max_cluster_sizes

    def calculate_sample_corr_length(
        self, cluster_config: np.ndarray | csr_matrix, use_manhattan: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate sample correlation length from a single cluster configuration.

        The sample correlation length is calculated as the square root of the weighted
        average of squared radii of gyration (Rg^2) multipled by sqrt(2), where the weights
        are the squares of cluster sizes.

        Parameters
        ----------
        cluster_config : 1D numpy array or scipy csr_matrix
            Cluster configuration where values >= 1 indicate the cluster index
            that each qubit belongs to, and 0 indicates qubits outside clusters.
            If numpy array: shape should be (num_qubits,)
            If csr_matrix: shape should be (1, num_qubits)
        use_manhattan : bool, optional
            If True, uses Manhattan distance for radius calculations.
            Otherwise, uses Euclidean distance (default is False).

        Returns
        -------
        correlation_length : float
            Calculated sample correlation length.
        total_weight : float
            Total weight (= sum of cluster sizes squared).
        """
        # Extract clusters from configuration
        clusters = self._extract_clusters_from_config(cluster_config)

        if not clusters:
            return 0.0, 0.0

        # Calculate weighted sum of Rg^2 for all clusters
        total_weighted_rg_squared = 0.0
        total_weight = 0.0

        dims = (2 * self.d, 2 * self.d)  # Periodic dimensions

        for cluster_idx, qubit_indices in clusters.items():
            cluster_size = len(qubit_indices)
            if cluster_size <= 1:
                continue  # Skip single-qubit clusters

            # Get coordinates for qubits in this cluster
            cluster_coords = self.qubit_coordinates[qubit_indices]

            # Calculate center of mass using periodic boundary conditions
            com = calculate_com_periodic(cluster_coords.astype(float), dims)
            com_array = np.array(com).reshape(1, 2)

            # Calculate distances from each qubit to COM
            distances = calculate_distance_periodic(
                cluster_coords.astype(float),
                com_array,
                dims,
                use_manhattan=use_manhattan,
            )

            # Calculate Rg^2 (mean squared distance from COM)
            rg_squared = np.mean(distances**2)

            # Weight by cluster size squared
            weight = cluster_size**2
            total_weighted_rg_squared += weight * rg_squared
            total_weight += weight

        if total_weight == 0.0:
            return 0.0, 0.0

        # Calculate weighted average of Rg^2
        avg_rg_squared = total_weighted_rg_squared / total_weight

        # corr_length = sqrt(avg_rg_squared) * sqrt(2) / 2
        # "/ 2" is for adjusting its unit to be the same as the lattice spacing.
        correlation_length = np.sqrt(avg_rg_squared) / np.sqrt(2)

        return correlation_length, total_weight

    def calculate_sample_corr_length_batch(
        self, cluster_configs: np.ndarray | csr_matrix, use_manhattan: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate sample correlation length for multiple cluster configurations.

        Parameters
        ----------
        cluster_configs : 2D numpy array or scipy csr_matrix
            Batch of cluster configurations where each row represents a sample.
            Values >= 1 indicate the cluster index that each qubit belongs to,
            and 0 indicates qubits outside clusters.
            Shape should be (num_samples, num_qubits)
        use_manhattan : bool, optional
            If True, uses Manhattan distance for radius calculations.
            Otherwise, uses Euclidean distance (default is False).

        Returns
        -------
        corr_lengths : numpy array of float
            Array of estimated correlation lengths for each sample.
        weights : numpy array of float
            Array of total weights for each sample.
        """
        if isinstance(cluster_configs, csr_matrix):
            num_samples = cluster_configs.shape[0]
            corr_lengths = np.zeros(num_samples, dtype="float32")
            weights = np.zeros(num_samples, dtype="float32")

            for sample_idx in range(num_samples):
                # Extract single row as csr_matrix
                sample_config = cluster_configs.getrow(sample_idx)
                corr_length, weight = self.calculate_sample_corr_length(
                    sample_config, use_manhattan=use_manhattan
                )
                corr_lengths[sample_idx] = corr_length
                weights[sample_idx] = weight
        else:
            # Handle 2D numpy array
            if cluster_configs.ndim != 2:
                raise ValueError("cluster_configs must be a 2D array")

            num_samples = cluster_configs.shape[0]
            corr_lengths = np.zeros(num_samples, dtype="float32")
            weights = np.zeros(num_samples, dtype="float32")

            for sample_idx in range(num_samples):
                sample_config = cluster_configs[sample_idx]
                corr_length, weight = self.calculate_sample_corr_length(
                    sample_config, use_manhattan=use_manhattan
                )
                corr_lengths[sample_idx] = corr_length
                weights[sample_idx] = weight

        return corr_lengths, weights
