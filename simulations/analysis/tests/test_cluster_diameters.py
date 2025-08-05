import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix
from simulations.analysis.cluster_diameters import ClusterDiameterCalculator


class TestClusterDiameter:
    """Test cases for the ClusterDiameter class."""

    @pytest.fixture
    def simple_graph(self):
        """Returns a ClusterDiameter instance for a simple linear graph 0-1-2-3."""
        H = csc_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))
        return ClusterDiameterCalculator(H)

    def test_init_with_valid_matrix(self, simple_graph):
        """Test initialization with a valid CSC matrix."""
        assert simple_graph.num_nodes == 4
        assert simple_graph.decoding_graph.vcount() == 4
        assert simple_graph.decoding_graph.ecount() == 3

    def test_init_with_invalid_matrix_type(self):
        """Test initialization with invalid matrix type raises TypeError."""
        H = np.array([[1, 0], [0, 1]])
        with pytest.raises(TypeError, match="H must be a scipy.sparse.csc_matrix."):
            ClusterDiameterCalculator(H)

    def test_build_graph_clique(self):
        """Test graph construction with a row that forms a clique."""
        H = csc_matrix(np.array([[1, 1, 1, 1]]))
        cluster_diameter = ClusterDiameterCalculator(H)
        graph = cluster_diameter.decoding_graph
        assert graph.vcount() == 4
        assert graph.ecount() == 6

    @pytest.mark.parametrize("as_csr", [False, True])
    def test_compute_cluster_diameters_simple(self, simple_graph, as_csr):
        """Test diameter computation with simple clusters."""
        clusters_np = np.array([1, 1, 1, 2])
        clusters = csr_matrix(clusters_np) if as_csr else clusters_np

        diameters = simple_graph.compute_cluster_diameters(clusters)
        assert diameters == {1: 2, 2: 0}

    @pytest.mark.parametrize("as_csr", [False, True])
    def test_compute_cluster_diameters_disconnected(self, as_csr):
        """Test diameter computation with disconnected clusters."""
        H = csc_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]))
        cluster_diameter = ClusterDiameterCalculator(H)
        clusters_np = np.array([1, 0, 1, 0])
        clusters = csr_matrix(clusters_np) if as_csr else clusters_np
        diameters = cluster_diameter.compute_cluster_diameters(clusters)
        # Diameter of a disconnected graph with two nodes is 0
        assert diameters == {1: 0}

    def test_compute_cluster_diameters_no_clusters(self, simple_graph):
        """Test diameter computation when no clusters are specified."""
        clusters = np.array([0, 0, 0, 0])
        diameters = simple_graph.compute_cluster_diameters(clusters)
        assert len(diameters) == 0

    def test_compute_cluster_diameters_invalid_input(self, simple_graph):
        """Test diameter computation with invalid input."""
        with pytest.raises(ValueError, match="The length of `clusters` must be equal"):
            simple_graph.compute_cluster_diameters(np.array([1, 2, 3]))

        with pytest.raises(TypeError, match="must be a 1D numpy array"):
            simple_graph.compute_cluster_diameters([1, 2, 3, 4])

        # csr_matrix must have shape (1, N)
        with pytest.raises(ValueError, match="its shape must be"):
            simple_graph.compute_cluster_diameters(
                csr_matrix(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))
            )

    @pytest.mark.parametrize("as_csr", [False, True])
    def test_compute_cluster_diameters_single_node_clusters(self, as_csr):
        """Test diameter computation with single-node clusters."""
        H = csc_matrix(np.identity(3))
        cluster_diameter = ClusterDiameterCalculator(H)
        clusters_np = np.array([1, 2, 3])
        clusters = csr_matrix(clusters_np) if as_csr else clusters_np
        diameters = cluster_diameter.compute_cluster_diameters(clusters)
        assert diameters == {1: 0, 2: 0, 3: 0}

    def test_empty_matrix(self):
        """Test with an empty matrix."""
        H = csc_matrix((0, 3))
        cluster_diameter = ClusterDiameterCalculator(H)
        clusters = np.array([1, 1, 2])
        diameters = cluster_diameter.compute_cluster_diameters(clusters)
        assert diameters == {1: 0, 2: 0}

    @pytest.mark.parametrize("as_csr", [False, True])
    @pytest.mark.parametrize("return_max", [False, True])
    def test_compute_cluster_diameters_batch(self, simple_graph, as_csr, return_max):
        """Test batch computation of cluster diameters."""
        clusters_np = np.array([[1, 1, 1, 2], [1, 0, 2, 1]])
        clusters = csr_matrix(clusters_np) if as_csr else clusters_np

        results = simple_graph.compute_cluster_diameters_batch(
            clusters, return_max_diameter=return_max
        )
        if return_max:
            expected = [2, 0]
        else:
            expected = [{1: 2, 2: 0}, {1: 0, 2: 0}]
        assert results == expected

    @pytest.mark.parametrize("n_jobs", [2, -1])
    @pytest.mark.parametrize("return_max", [False, True])
    def test_compute_cluster_diameters_batch_parallel(
        self, simple_graph, n_jobs, return_max
    ):
        """Test parallel batch computation."""
        clusters_np = np.array([[1, 1, 1, 2], [1, 0, 2, 1], [3, 3, 0, 0], [0, 4, 4, 4]])
        results = simple_graph.compute_cluster_diameters_batch(
            clusters_np, n_jobs=n_jobs, return_max_diameter=return_max
        )
        if return_max:
            expected = [2, 0, 1, 2]
        else:
            expected = [{1: 2, 2: 0}, {1: 0, 2: 0}, {3: 1}, {4: 2}]
        assert results == expected

    def test_compute_cluster_diameters_batch_invalid_input(self, simple_graph):
        """Test batch computation with invalid input."""
        with pytest.raises(ValueError, match="number of columns"):
            simple_graph.compute_cluster_diameters_batch(np.array([[1, 2, 3]]))
        with pytest.raises(ValueError, match="must be a 2D array"):
            simple_graph.compute_cluster_diameters_batch(np.array([1, 2, 3, 4]))

    def test_compute_cluster_diameters_batch_empty_input(self, simple_graph):
        """Test batch computation with empty input."""
        clusters_np = np.empty((0, simple_graph.num_nodes))
        assert simple_graph.compute_cluster_diameters_batch(clusters_np) == []

        clusters_csr = csr_matrix((0, simple_graph.num_nodes))
        assert simple_graph.compute_cluster_diameters_batch(clusters_csr) == []
