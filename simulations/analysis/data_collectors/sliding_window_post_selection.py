"""
Ultra-efficient retrospective real-time post-selection analysis for sliding window decoding.

This module implements high-performance vectorized analysis of existing sliding window
simulation data to evaluate real-time post-selection strategies without re-simulation.
The implementation uses advanced numpy broadcasting and vectorization to process
multiple cutoff values and samples simultaneously.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path

from .numpy_utils.sliding_window import (
    calculate_committed_cluster_norm_fractions_from_csr,
)
from simulations.analysis.plotting_helpers import get_confint


class RealTimePostSelectionAnalyzer:
    """
    Ultra-efficient analyzer for retrospective real-time post-selection analysis.

    This class processes existing sliding window simulation data to evaluate
    real-time post-selection performance across multiple cutoff values simultaneously.
    Uses advanced vectorization and numpy broadcasting for maximum efficiency.
    """

    def __init__(
        self,
        committed_clusters_csr: csr_matrix,
        committed_faults: List[np.ndarray],
        priors: np.ndarray,
        H: np.ndarray,
        F: int,
        T: int,
        num_faults_per_window: int,
    ):
        """
        Initialize the real-time post-selection analyzer.

        Parameters
        ----------
        committed_clusters_csr : csr_matrix
            Sparse matrix of committed cluster assignments.
            Shape: (num_samples, num_windows * num_faults_per_window)
        committed_faults : List[np.ndarray]
            List of committed faults for each window.
        priors : np.ndarray
            Prior error probabilities for each fault.
        H : np.ndarray
            Parity check matrix for cluster analysis.
        F : int
            Commit size (number of rounds committed per window).
        T : int
            Total number of rounds.
        num_faults_per_window : int
            Number of faults per window.
        """
        self.committed_clusters_csr = committed_clusters_csr
        self.committed_faults = committed_faults
        self.priors = priors
        self.H = H
        self.F = F
        self.T = T
        self.num_faults_per_window = num_faults_per_window

        # Derived properties
        self.num_samples = committed_clusters_csr.shape[0]
        self.num_windows = len(committed_faults)

        # Pre-compute adjacency matrix for cluster analysis
        self.adj_matrix = (H.T @ H) == 1

        # Cache for computed metrics
        self._metrics_cache = {}

    def analyze_postselection_vectorized(
        self,
        cutoffs: Union[np.ndarray, List[float]],
        metric_windows: int = 1,
        norm_order: float = 2.0,
        value_type: str = "llr",
        disable_cache: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Perform ultra-efficient vectorized real-time post-selection analysis.

        This function processes all cutoff values simultaneously using advanced
        numpy broadcasting and vectorization techniques for maximum performance.

        Parameters
        ----------
        cutoffs : array-like
            Array of cutoff values to test simultaneously.
        metric_windows : int, default=1
            Number of windows for metric evaluation.
        norm_order : float, default=2.0
            Order for L_p norm calculation.
        value_type : str, default="llr"
            Type of cluster value calculation ("size" or "llr").
        disable_cache : bool, default=False
            If True, disable caching of computed metrics.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing analysis results:
            - 'abort_windows': Shape (num_samples, num_cutoffs) - window where cutoff exceeded
            - 'effective_trials': Shape (num_samples, num_cutoffs) - effective number of trials
            - 'accepted_mask': Shape (num_samples, num_cutoffs) - True if sample accepted
            - 'cutoffs': Shape (num_cutoffs,) - cutoff values used
            - 'metrics_matrix': Shape (num_samples, num_evaluatable_windows) - pre-computed metrics
        """
        cutoffs = np.asarray(cutoffs, dtype=np.float64)
        num_cutoffs = len(cutoffs)

        # Determine evaluatable windows (windows >= metric_windows - 1)
        first_eval_window = metric_windows - 1
        num_eval_windows = max(0, self.num_windows - first_eval_window)

        if num_eval_windows == 0:
            warnings.warn(
                f"No evaluatable windows with metric_windows={metric_windows}"
            )
            return self._empty_results(num_cutoffs, cutoffs)

        # Step 1: Pre-compute metrics matrix for all samples and evaluatable windows
        metrics_matrix = self._compute_metrics_matrix_vectorized(
            first_eval_window, metric_windows, norm_order, value_type, disable_cache
        )

        # Step 2: Vectorized cutoff evaluation using broadcasting
        # Shape: (num_samples, num_eval_windows, num_cutoffs)
        cutoffs_broadcast = cutoffs.reshape(1, 1, -1)
        exceeds_cutoff = metrics_matrix[:, :, np.newaxis] > cutoffs_broadcast

        # Step 3: Vectorized abort window detection
        abort_windows, accepted_mask = self._detect_abort_windows_vectorized(
            exceeds_cutoff, first_eval_window
        )

        # Step 4: Vectorized effective trials calculation
        effective_trials = self._calculate_effective_trials_vectorized(
            abort_windows, accepted_mask, first_eval_window
        )

        return {
            "abort_windows": abort_windows,
            "effective_trials": effective_trials,
            "accepted_mask": accepted_mask,
            "cutoffs": cutoffs,
            "metrics_matrix": metrics_matrix,
            "first_eval_window": first_eval_window,
            "metric_windows": metric_windows,
            "norm_order": norm_order,
            "value_type": value_type,
        }

    def _compute_metrics_matrix_vectorized(
        self,
        first_eval_window: int,
        metric_windows: int,
        norm_order: float,
        value_type: str,
        disable_cache: bool,
    ) -> np.ndarray:
        """
        Pre-compute metrics matrix for all samples and evaluatable windows.

        Returns
        -------
        np.ndarray
            Shape (num_samples, num_evaluatable_windows) containing
            committed cluster norm fractions for each sample/window combination.
        """
        cache_key = (first_eval_window, metric_windows, norm_order, value_type)

        if not disable_cache and cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        num_eval_windows = self.num_windows - first_eval_window
        metrics_matrix = np.zeros(
            (self.num_samples, num_eval_windows), dtype=np.float64
        )

        # Compute metrics for each evaluatable window
        for i, window_idx in enumerate(range(first_eval_window, self.num_windows)):
            # Evaluation window range: (window_idx - metric_windows + 1, window_idx + 1)
            eval_start = max(0, window_idx - metric_windows + 1)
            eval_end = window_idx + 1
            eval_windows = (eval_start, eval_end)

            # Calculate committed cluster norm fractions for this evaluation window
            window_metrics = calculate_committed_cluster_norm_fractions_from_csr(
                self.committed_clusters_csr,
                self.committed_faults,
                self.priors,
                self.adj_matrix,
                norm_order=norm_order,
                value_type=value_type,
                eval_windows=eval_windows,
                _benchmarking=False,
                num_jobs=1,
                num_batches=None,
            )

            metrics_matrix[:, i] = window_metrics

        if not disable_cache:
            self._metrics_cache[cache_key] = metrics_matrix

        return metrics_matrix

    def _detect_abort_windows_vectorized(
        self, exceeds_cutoff: np.ndarray, first_eval_window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized detection of abort windows for all samples and cutoffs.

        Parameters
        ----------
        exceeds_cutoff : np.ndarray
            Shape (num_samples, num_eval_windows, num_cutoffs) boolean array
            indicating where metrics exceed cutoffs.
        first_eval_window : int
            Index of first evaluatable window.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            abort_windows: Shape (num_samples, num_cutoffs) - absolute window indices where cutoff exceeded
            accepted_mask: Shape (num_samples, num_cutoffs) - True if sample never exceeded cutoff
        """
        # Find first occurrence of True along window axis (axis=1)
        # If no True found, argmax returns 0, so we need to handle this case
        first_exceed_indices = np.argmax(
            exceeds_cutoff, axis=1
        )  # Shape: (num_samples, num_cutoffs)

        # Check if cutoff was actually exceeded (handle cases where argmax returns 0 but no True exists)
        any_exceeded = np.any(
            exceeds_cutoff, axis=1
        )  # Shape: (num_samples, num_cutoffs)

        # Convert relative indices to absolute window indices
        abort_windows = first_exceed_indices + first_eval_window

        # Samples are accepted if cutoff was never exceeded
        accepted_mask = ~any_exceeded

        # For accepted samples, set abort_windows to a sentinel value (e.g., -1)
        abort_windows = np.where(accepted_mask, -1, abort_windows)

        return abort_windows, accepted_mask

    def _calculate_effective_trials_vectorized(
        self,
        abort_windows: np.ndarray,
        accepted_mask: np.ndarray,
        first_eval_window: int,
    ) -> np.ndarray:
        """
        Vectorized calculation of effective trials for all samples and cutoffs.

        Uses the formula: (window_exceeding_cutoff + 1) * F / T
        For accepted samples, uses the total number of windows.

        Parameters
        ----------
        abort_windows : np.ndarray
            Shape (num_samples, num_cutoffs) - absolute window indices where cutoff exceeded.
        accepted_mask : np.ndarray
            Shape (num_samples, num_cutoffs) - True if sample was accepted.
        first_eval_window : int
            Index of first evaluatable window.

        Returns
        -------
        np.ndarray
            Shape (num_samples, num_cutoffs) - effective number of trials.
        """
        # For aborted samples: (abort_window + 1) * F / T
        # For accepted samples: total_windows * F / T
        effective_trials = np.where(
            accepted_mask,
            self.num_windows * self.F / self.T,  # Accepted: use all windows
            (abort_windows + 1) * self.F / self.T,  # Aborted: use abort window + 1
        )

        return effective_trials

    def _empty_results(
        self, num_cutoffs: int, cutoffs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Return empty results dictionary when no evaluatable windows exist."""
        return {
            "abort_windows": np.full((self.num_samples, num_cutoffs), -1, dtype=int),
            "effective_trials": np.zeros(
                (self.num_samples, num_cutoffs), dtype=np.float64
            ),
            "accepted_mask": np.ones((self.num_samples, num_cutoffs), dtype=bool),
            "cutoffs": cutoffs,
            "metrics_matrix": np.zeros((self.num_samples, 0), dtype=np.float64),
            "first_eval_window": 0,
            "metric_windows": 0,
            "norm_order": 2.0,
            "value_type": "llr",
        }

    def compute_postselection_statistics(
        self, results: Dict[str, np.ndarray], fails: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive post-selection statistics from analysis results.

        Parameters
        ----------
        results : Dict[str, np.ndarray]
            Results from analyze_postselection_vectorized().
        fails : np.ndarray
            Boolean array indicating decoding failures for each sample.
            Shape: (num_samples,)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing statistics for each cutoff:
            - 'p_fail': Logical error rate (failures among accepted samples)
            - 'delta_p_fail': Margin of error for p_fail (95% confidence interval)
            - 'p_abort': Abort rate (fraction of samples aborted)
            - 'effective_avg_trials': Average effective trials among accepted samples
            - 'num_accepted': Number of accepted samples
            - 'num_failed_accepted': Number of failed samples among accepted
        """
        cutoffs = results["cutoffs"]
        accepted_mask = results["accepted_mask"]
        effective_trials = results["effective_trials"]
        num_cutoffs = len(cutoffs)

        # Initialize output arrays
        p_fail = np.zeros(num_cutoffs, dtype=np.float64)
        delta_p_fail = np.zeros(num_cutoffs, dtype=np.float64)
        p_abort = np.zeros(num_cutoffs, dtype=np.float64)
        effective_avg_trials = np.zeros(num_cutoffs, dtype=np.float64)
        num_accepted = np.zeros(num_cutoffs, dtype=int)
        num_failed_accepted = np.zeros(num_cutoffs, dtype=int)

        # Vectorized computation across cutoffs
        for i in range(num_cutoffs):
            accepted_samples = accepted_mask[:, i]
            num_accepted[i] = np.sum(accepted_samples)

            if num_accepted[i] > 0:
                # Logical error rate among accepted samples
                failed_accepted = fails & accepted_samples
                num_failed_accepted[i] = np.sum(failed_accepted)

                # Average effective trials among accepted samples
                effective_avg_trials[i] = np.mean(effective_trials[accepted_samples, i])
            else:
                num_failed_accepted[i] = 0
                effective_avg_trials[i] = 0.0

            # Abort rate
            p_abort[i] = 1.0 - (num_accepted[i] / self.num_samples)

        # Compute confidence intervals for p_fail using get_confint
        # Handle edge cases where num_accepted is zero
        valid_mask = num_accepted > 0
        if np.any(valid_mask):
            # Compute confidence intervals only for valid cases
            p_fail_valid, delta_p_fail_valid = get_confint(
                num_failed_accepted[valid_mask], 
                num_accepted[valid_mask]
            )
            p_fail[valid_mask] = p_fail_valid
            delta_p_fail[valid_mask] = delta_p_fail_valid
        
        # For invalid cases (num_accepted == 0), p_fail and delta_p_fail remain 0

        return {
            "p_fail": p_fail,
            "delta_p_fail": delta_p_fail,
            "p_abort": p_abort,
            "effective_avg_trials": effective_avg_trials,
            "num_accepted": num_accepted,
            "num_failed_accepted": num_failed_accepted,
            "cutoffs": cutoffs,
        }

    def clear_cache(self):
        """Clear the metrics cache to free memory."""
        self._metrics_cache.clear()


def load_sliding_window_data(
    data_dir: str, param_combo: str
) -> Tuple[csr_matrix, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sliding window simulation data for post-selection analysis.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combo : str
        Parameter combination string (e.g., "n144_T12_p0.003_W3_F1").

    Returns
    -------
    Tuple containing:
        - committed_clusters_csr: CSR matrix of committed cluster assignments
        - committed_faults: List of committed fault arrays per window
        - priors: Prior error probabilities
        - H: Parity check matrix
        - fails: Boolean array of decoding failures
    """
    param_dir = Path(data_dir) / param_combo

    # Load basic matrices
    H = sp.load_npz(param_dir / "H.npz").toarray()
    priors = np.load(param_dir / "priors.npy")
    committed_faults_data = np.load(
        param_dir / "committed_faults.npz", allow_pickle=True
    )
    committed_faults = [
        committed_faults_data[f"arr_{i}"]
        for i in range(len(committed_faults_data.files))
    ]

    # Find and load batch data
    batch_dirs = [
        d for d in param_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")
    ]

    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found in {param_dir}")

    # Load data from all batches
    all_committed_clusters = []
    all_fails = []

    for batch_dir in sorted(batch_dirs):
        # Load committed clusters and failures for this batch
        committed_clusters = sp.load_npz(batch_dir / "committed_clusters.npz")
        fails = np.load(batch_dir / "fails.npy")

        all_committed_clusters.append(committed_clusters)
        all_fails.append(fails)

    # Concatenate data from all batches
    committed_clusters_csr = sp.vstack(all_committed_clusters, format="csr")
    fails_combined = np.concatenate(all_fails)

    return committed_clusters_csr, committed_faults, priors, H, fails_combined


def analyze_parameter_combination(
    data_dir: str,
    param_combo: str,
    cutoffs: np.ndarray,
    metric_windows: int = 1,
    norm_order: float = 2.0,
    value_type: str = "llr",
) -> Dict[str, np.ndarray]:
    """
    Perform complete post-selection analysis for a single parameter combination.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combo : str
        Parameter combination string (e.g., "n144_T12_p0.003_W3_F1").
    cutoffs : np.ndarray
        Array of cutoff values to analyze.
    metric_windows : int, default=1
        Number of windows for metric evaluation.
    norm_order : float, default=2.0
        Order for L_p norm calculation.
    value_type : str, default="llr"
        Type of cluster value calculation ("size" or "llr").

    Returns
    -------
    Dict[str, np.ndarray]
        Complete post-selection statistics for all cutoffs.
    """
    # Load simulation data
    committed_clusters_csr, committed_faults, priors, H, fails = (
        load_sliding_window_data(data_dir, param_combo)
    )

    # Parse parameters from combo string
    parts = param_combo.split("_")
    F = int(parts[-1][1:])  # Extract F from "F1"
    T = int(parts[-3][1:])  # Extract T from "T12"

    # Determine number of faults per window from data structure
    total_faults = committed_clusters_csr.shape[1]
    num_windows = len(committed_faults)
    num_faults_per_window = total_faults // num_windows

    # Create analyzer and perform analysis
    analyzer = RealTimePostSelectionAnalyzer(
        committed_clusters_csr=committed_clusters_csr,
        committed_faults=committed_faults,
        priors=priors,
        H=H,
        F=F,
        T=T,
        num_faults_per_window=num_faults_per_window,
    )

    # Perform vectorized analysis
    results = analyzer.analyze_postselection_vectorized(
        cutoffs=cutoffs,
        metric_windows=metric_windows,
        norm_order=norm_order,
        value_type=value_type,
    )

    # Compute comprehensive statistics
    statistics = analyzer.compute_postselection_statistics(results, fails)

    # Combine results and statistics
    combined_results = {**results, **statistics}

    return combined_results
