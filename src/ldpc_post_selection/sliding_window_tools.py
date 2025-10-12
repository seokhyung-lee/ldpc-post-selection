from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import igraph as ig
import scipy.sparse as sp

from ldpc_post_selection.cluster_tools import (
    compute_lp_norm,
    label_clusters_igraph,
)


class CommittedClusterNormCalculator:
    """
    Efficiently evaluate committed cluster norm fractions for multiple samples.

    This class pre-computes all sample-independent structures (committed faults,
    igraph representation, log-likelihood ratios) and caches sliding-window
    reductions so that repeated metric evaluations avoid redundant work.

    Parameters
    ----------
    committed_faults : list of 1D numpy arrays of bool with shape (num_faults,)
        Per-window indicators defining the committed fault region.
    priors : 1D numpy array of float with shape (num_faults,)
        Prior fault probabilities used to derive log-likelihood ratios.
    H : 2D array-like of bool or int with shape (num_checks, num_faults)
        Parity-check matrix whose columns correspond to faults. The adjacency
        matrix used for clustering is derived internally as `H.T @ H`.
    """

    def __init__(
        self,
        committed_faults: list[np.ndarray],
        priors: np.ndarray,
        H: np.ndarray | sp.spmatrix,
    ) -> None:
        self._priors = np.asarray(priors, dtype=float)
        if self._priors.ndim != 1:
            raise ValueError("priors must be a 1D array.")

        self._num_faults = int(self._priors.shape[0])
        self._committed_faults = committed_faults
        self._fault_matrix = self._stack_boolean_windows(
            committed_faults, "committed_faults"
        )
        self._num_windows = int(self._fault_matrix.shape[0])
        self._full_fault_mask = (
            self._fault_matrix.any(axis=0)
            if self._num_windows > 0
            else np.zeros(self._num_faults, dtype=bool)
        )

        self._parity_matrix = sp.csr_matrix(H, dtype=bool)
        if self._parity_matrix.shape[1] != self._num_faults:
            raise ValueError(
                "The number of columns in H must match the length of priors."
            )
        adjacency = self._parity_matrix.transpose().dot(self._parity_matrix)
        adjacency = adjacency.tolil()
        adjacency.setdiag(False)
        adjacency = adjacency.astype(bool).tocsr()
        adjacency.eliminate_zeros()

        self._graph = self._build_full_graph(adjacency)
        self._bit_llrs = np.log((1.0 - self._priors) / self._priors)

        # Cache sliding-window faults keyed by (target_windows, lookback_size)
        self._fault_sliding_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def compute(
        self,
        committed_clusters: list[np.ndarray],
        norm_order: float,
        value_type: str,
        lookback_window_size: int | None = None,
    ) -> np.ndarray:
        """
        Calculate committed cluster norm fractions for a sample.

        Parameters
        ----------
        committed_clusters : list of 1D numpy arrays of bool with shape (num_faults,)
            Per-window indicators showing which faults belong to a committed cluster
            for the evaluated sample.
        norm_order : float
            Order of the L_p norm to evaluate (positive number or `np.inf`).
        value_type : str
            Type of value to aggregate inside clusters, either 'size' or 'llr'.
        lookback_window_size : int or None, optional
            If provided, number of consecutive windows to include per evaluation;
            sliding metrics are returned for each position.

        Returns
        -------
        norm_fractions : 1D numpy array of float
            Norm fractions for each evaluation. When `lookback_window_size` is None,
            the array has length 1 containing the single-sample norm fraction.
        """
        if value_type not in {"size", "llr"}:
            raise ValueError("value_type must be either 'size' or 'llr'.")

        cluster_matrix = self._stack_boolean_windows(
            committed_clusters, "committed_clusters"
        )
        target_windows = max(cluster_matrix.shape[0], self._num_windows)

        if self._num_windows not in (0, target_windows):
            raise ValueError(
                "Number of committed cluster windows does not match committed faults."
            )

        if cluster_matrix.shape[0] == 0 and target_windows > 0:
            cluster_matrix = np.zeros((target_windows, self._num_faults), dtype=bool)
        elif cluster_matrix.shape[0] not in (0, target_windows):
            raise ValueError(
                "Committed cluster data must provide either zero or all windows."
            )

        if lookback_window_size is None:
            if target_windows == 0:
                return np.zeros(1, dtype=float)

            cluster_mask = (
                cluster_matrix.any(axis=0)
                if cluster_matrix.size > 0
                else np.zeros(self._num_faults, dtype=bool)
            )
            fault_mask = (
                self._full_fault_mask
                if self._num_windows == target_windows
                else np.zeros(self._num_faults, dtype=bool)
            )

            norm_value = self._compute_norm_fraction_for_masks(
                committed_cluster_mask=cluster_mask,
                committed_fault_mask=fault_mask,
                norm_order=norm_order,
                value_type=value_type,
            )
            return np.array([norm_value], dtype=float)

        if target_windows == 0:
            return np.zeros(0, dtype=float)

        if lookback_window_size <= 0:
            raise ValueError("lookback_window_size must be a positive integer.")
        if lookback_window_size > target_windows:
            raise ValueError(
                "lookback_window_size cannot exceed the number of available windows."
            )

        sliding_clusters = self._sliding_window_any(cluster_matrix, lookback_window_size)
        sliding_faults = self._get_sliding_faults(target_windows, lookback_window_size)

        num_positions = sliding_clusters.shape[0]
        norm_fractions = np.zeros(num_positions, dtype=float)

        for idx in range(num_positions):
            norm_fractions[idx] = self._compute_norm_fraction_for_masks(
                committed_cluster_mask=sliding_clusters[idx],
                committed_fault_mask=sliding_faults[idx],
                norm_order=norm_order,
                value_type=value_type,
            )

        return norm_fractions

    def _get_sliding_faults(
        self, target_windows: int, lookback_window_size: int
    ) -> np.ndarray:
        """
        Retrieve (or build) cached sliding-window committed fault masks.

        Parameters
        ----------
        target_windows : int
            Number of windows represented in the evaluation.
        lookback_window_size : int
            Sliding window size requested by the metric.

        Returns
        -------
        sliding_faults : 2D numpy array of bool
            Logical-OR committed fault masks for each sliding position.
        """
        cache_key = (target_windows, lookback_window_size)
        if cache_key in self._fault_sliding_cache:
            return self._fault_sliding_cache[cache_key]

        if target_windows == self._num_windows:
            if self._num_windows == 0:
                sliding_faults = np.zeros((0, self._num_faults), dtype=bool)
            else:
                sliding_faults = self._sliding_window_any(
                    self._fault_matrix, lookback_window_size
                )
        else:
            # Only possible when committed faults were not supplied per-window.
            sliding_faults = np.zeros(
                (target_windows - lookback_window_size + 1, self._num_faults),
                dtype=bool,
            )

        self._fault_sliding_cache[cache_key] = sliding_faults
        return sliding_faults

    def _stack_boolean_windows(
        self, window_data: list[np.ndarray], label: str
    ) -> np.ndarray:
        """
        Normalize per-window boolean indicators to a dense matrix representation.

        Parameters
        ----------
        window_data : list of 1D numpy arrays of bool
            Window-wise boolean indicators.
        label : str
            Descriptive label used in validation error messages.

        Returns
        -------
        window_matrix : 2D numpy array of bool
            Stacked window data with shape (num_windows, num_faults).
        """
        if not window_data:
            return np.zeros((0, self._num_faults), dtype=bool)

        stacked: list[np.ndarray] = []
        for window in window_data:
            window_array = np.asarray(window, dtype=bool)
            if window_array.ndim != 1 or window_array.shape[0] != self._num_faults:
                raise ValueError(
                    f"Each {label} entry must be a 1D array matching priors length."
                )
            stacked.append(window_array)

        return np.vstack(stacked)

    def _build_full_graph(self, adj_matrix: sp.spmatrix | np.ndarray) -> ig.Graph:
        """
        Construct an igraph Graph object from the stored adjacency matrix.

        Parameters
        ----------
        adj_matrix : scipy sparse matrix or numpy array of bool
            Adjacency matrix describing fault connectivity, typically derived
            from `H.T @ H`.

        Returns
        -------
        igraph_graph : igraph Graph
            Undirected igraph representation of the adjacency matrix.
        """
        if adj_matrix.shape[0] != self._num_faults or adj_matrix.shape[1] != self._num_faults:
            raise ValueError(
                "Adjacency matrix must be square with size matching the number of faults."
            )

        if hasattr(adj_matrix, "tocoo"):
            coo_matrix = adj_matrix.tocoo()
        else:
            coo_matrix = sp.coo_matrix(adj_matrix)

        graph = ig.Graph(n=self._num_faults, directed=False)
        if coo_matrix.nnz > 0:
            graph.add_edges(zip(coo_matrix.row, coo_matrix.col))

        return graph

    def _compute_norm_fraction_for_masks(
        self,
        committed_cluster_mask: np.ndarray,
        committed_fault_mask: np.ndarray,
        norm_order: float,
        value_type: str,
    ) -> float:
        """
        Compute norm fraction for a combined committed region.

        Parameters
        ----------
        committed_cluster_mask : 1D numpy array of bool
            Boolean mask identifying committed cluster faults.
        committed_fault_mask : 1D numpy array of bool
            Boolean mask identifying committed faults for normalization.
        norm_order : float
            Order for the L_p norm calculation.
        value_type : str
            Either "size" or "llr".

        Returns
        -------
        norm_fraction : float
            Norm fraction for the provided committed masks.
        """
        committed_indices = np.flatnonzero(committed_cluster_mask)
        if committed_indices.size == 0:
            return 0.0

        cluster_labels = label_clusters_igraph(
            self._graph, committed_indices.astype(np.int_)
        )
        sample_cluster_labels = cluster_labels[committed_indices]

        if sample_cluster_labels.size == 0:
            return 0.0

        max_cluster_label = int(sample_cluster_labels.max())
        if max_cluster_label <= 0:
            return 0.0

        if value_type == "size":
            total_committed_faults = int(np.count_nonzero(committed_fault_mask))
            if total_committed_faults == 0:
                return 0.0

            cluster_sizes = np.bincount(
                sample_cluster_labels,
                minlength=max_cluster_label + 1,
            ).astype(np.float64)

            inside_norm = compute_lp_norm(cluster_sizes[1:], norm_order, take_abs=False)
            return (
                inside_norm / total_committed_faults
                if total_committed_faults > 0
                else 0.0
            )

        bit_llrs = self._bit_llrs

        total_llr_sum = float(np.sum(bit_llrs[committed_fault_mask]))
        if total_llr_sum <= 0:
            return 0.0

        cluster_llr_sums = np.bincount(
            sample_cluster_labels,
            weights=bit_llrs[committed_indices],
            minlength=max_cluster_label + 1,
        ).astype(np.float64)

        inside_norm = compute_lp_norm(cluster_llr_sums[1:], norm_order, take_abs=True)
        return inside_norm / total_llr_sum if total_llr_sum > 0 else 0.0

    def _sliding_window_any(
        self, boolean_matrix: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Compute sliding-window logical OR along the first dimension.

        Parameters
        ----------
        boolean_matrix : 2D numpy array of bool
            Boolean matrix of shape (num_windows, num_faults).
        window_size : int
            Size of the sliding window.

        Returns
        -------
        window_any : 2D numpy array of bool
            Boolean matrix of shape (num_windows - window_size + 1, num_faults)
            where each row is the logical OR across a sliding window.
        """
        if boolean_matrix.shape[0] == 0:
            return np.zeros((0, boolean_matrix.shape[1]), dtype=bool)

        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        if window_size > boolean_matrix.shape[0]:
            raise ValueError(
                "window_size cannot exceed the number of available windows."
            )

        cumulative = np.cumsum(boolean_matrix.astype(np.int32), axis=0)
        cumulative = np.vstack(
            [np.zeros((1, boolean_matrix.shape[1]), dtype=np.int32), cumulative]
        )
        window_sums = cumulative[window_size:] - cumulative[:-window_size]
        return window_sums > 0
