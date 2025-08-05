"""
Real-time post-selection analysis for sliding window decoding.

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
from joblib import Parallel, delayed
import numba

from .utils.sliding_window import (
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
        num_jobs: int = 1,
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
        num_jobs : int, default=1
            Number of parallel jobs for sample-level processing.

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
            first_eval_window,
            metric_windows,
            norm_order,
            value_type,
            disable_cache,
            num_jobs,
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
        num_jobs: int = 1,
    ) -> np.ndarray:
        """
        Pre-compute metrics matrix for all samples and evaluatable windows.

        Parameters
        ----------
        first_eval_window : int
            Index of first evaluatable window.
        metric_windows : int
            Number of windows for metric evaluation.
        norm_order : float
            Order for L_p norm calculation.
        value_type : str
            Type of cluster value calculation ("size" or "llr").
        disable_cache : bool
            If True, disable caching of computed metrics.
        num_jobs : int, default=1
            Number of parallel jobs for sample-level processing.

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
            # Evaluation window range: [window_idx - metric_windows + 1, window_idx] (inclusive)
            eval_start = max(0, window_idx - metric_windows + 1)
            eval_end = window_idx
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
                num_jobs=num_jobs,
                # num_batches=num_jobs * 5,
                verbose=3,
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

        Effective trials calculation:
        - Accepted samples: always 1.0
        - Aborted at last window (abort_windows == num_windows - 1): 1.0
        - Other aborted samples: (abort_window + 1) * F / T

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
        # For accepted samples: always 1.0
        # For aborted samples: 1.0 if aborted at last window, otherwise (abort_window + 1) * F / T

        # Handle aborted samples with special case for last window
        aborted_at_last_window = (abort_windows == self.num_windows - 1) & (
            ~accepted_mask
        )
        regular_aborted = (~accepted_mask) & (~aborted_at_last_window)

        effective_trials = np.zeros_like(abort_windows, dtype=np.float64)
        effective_trials[accepted_mask] = 1.0  # Accepted samples: always 1.0
        effective_trials[aborted_at_last_window] = 1.0  # Aborted at last window: 1.0
        effective_trials[regular_aborted] = (
            (abort_windows[regular_aborted] + 1) * self.F / self.T
        )  # Other aborted: formula

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
            - 'effective_avg_trials': Sum of all effective trials divided by number of accepted samples
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

                # Average effective trials: sum of all trials divided by number of accepted samples
                effective_avg_trials[i] = (
                    np.sum(effective_trials[:, i]) / num_accepted[i]
                )
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
                num_failed_accepted[valid_mask], num_accepted[valid_mask]
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


def load_single_batch_data(
    data_dir: str, param_combo: str, batch_dir_name: str
) -> Tuple[csr_matrix, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sliding window simulation data for a single batch.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combo : str
        Parameter combination string (e.g., "n144_T12_p0.003_W3_F1").
    batch_dir_name : str
        Name of the specific batch directory to load (e.g., "batch_1_1000000").

    Returns
    -------
    Tuple containing:
        - committed_clusters_csr: CSR matrix of committed cluster assignments for this batch
        - committed_faults: List of committed fault arrays per window
        - priors: Prior error probabilities
        - H: Parity check matrix
        - fails: Boolean array of decoding failures for this batch
    """
    param_dir = Path(data_dir) / param_combo
    batch_dir = param_dir / batch_dir_name

    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    # Load basic matrices (shared across all batches)
    H = sp.load_npz(param_dir / "H.npz").toarray()
    priors = np.load(param_dir / "priors.npy")
    committed_faults_data = np.load(
        param_dir / "committed_faults.npz", allow_pickle=True
    )
    committed_faults = [
        committed_faults_data[f"arr_{i}"]
        for i in range(len(committed_faults_data.files))
    ]

    # Load data for this specific batch
    committed_clusters = sp.load_npz(batch_dir / "committed_clusters.npz")
    fails = np.load(batch_dir / "fails.npy")

    return committed_clusters, committed_faults, priors, H, fails


def combine_batch_statistics(
    batch_stats_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Combine post-selection statistics from multiple batches.

    Parameters
    ----------
    batch_stats_list : List[Dict[str, np.ndarray]]
        List of statistics dictionaries from individual batches.
        Each dictionary should contain statistics from compute_postselection_statistics().

    Returns
    -------
    Dict[str, np.ndarray]
        Combined statistics across all batches:
        - 'p_fail': Logical error rate (failures among accepted samples)
        - 'delta_p_fail': Margin of error for p_fail (95% confidence interval)
        - 'p_abort': Abort rate (fraction of samples aborted)
        - 'effective_avg_trials': Weighted average of effective trials across batches
        - 'num_accepted': Total number of accepted samples
        - 'num_failed_accepted': Total number of failed samples among accepted
        - 'total_samples': Total number of samples across all batches
    """
    if not batch_stats_list:
        raise ValueError("batch_stats_list cannot be empty")

    # Get cutoffs from first batch (should be the same for all batches)
    cutoffs = batch_stats_list[0]["cutoffs"]
    num_cutoffs = len(cutoffs)

    # Initialize combined arrays
    total_num_accepted = np.zeros(num_cutoffs, dtype=int)
    total_num_failed_accepted = np.zeros(num_cutoffs, dtype=int)
    total_samples = 0

    # Arrays to store effective trials weighted by number of accepted samples
    weighted_effective_trials_sum = np.zeros(num_cutoffs, dtype=np.float64)

    # Sum across all batches
    for batch_stats in batch_stats_list:
        total_num_accepted += batch_stats["num_accepted"]
        total_num_failed_accepted += batch_stats["num_failed_accepted"]

        # For effective_avg_trials, we need to weight by the number of accepted samples
        # batch_stats["effective_avg_trials"] is already the average for that batch
        # So we multiply by num_accepted to get the total effective trials for that batch
        batch_total_effective_trials = (
            batch_stats["effective_avg_trials"] * batch_stats["num_accepted"]
        )
        weighted_effective_trials_sum += batch_total_effective_trials

        # Count total samples directly from batch sample count
        total_samples += batch_stats["batch_samples"]

    # Calculate combined effective average trials
    effective_avg_trials = np.where(
        total_num_accepted > 0, weighted_effective_trials_sum / total_num_accepted, 0.0
    )

    # Calculate combined p_fail and confidence intervals using get_confint
    p_fail = np.zeros(num_cutoffs, dtype=np.float64)
    delta_p_fail = np.zeros(num_cutoffs, dtype=np.float64)

    # Only calculate confidence intervals where we have accepted samples
    valid_mask = total_num_accepted > 0
    if np.any(valid_mask):
        p_fail_valid, delta_p_fail_valid = get_confint(
            total_num_failed_accepted[valid_mask], total_num_accepted[valid_mask]
        )
        p_fail[valid_mask] = p_fail_valid
        delta_p_fail[valid_mask] = delta_p_fail_valid

    # Calculate combined abort rate
    p_abort = (
        1.0 - (total_num_accepted / total_samples)
        if total_samples > 0
        else np.ones(num_cutoffs)
    )

    return {
        "p_fail": p_fail,
        "delta_p_fail": delta_p_fail,
        "p_abort": p_abort,
        "effective_avg_trials": effective_avg_trials,
        "num_accepted": total_num_accepted,
        "num_failed_accepted": total_num_failed_accepted,
        "cutoffs": cutoffs,
        "total_samples": total_samples,
    }


def analyze_parameter_combination_batch_by_batch(
    data_dir: str,
    param_combo: str,
    cutoffs: np.ndarray,
    metric_windows: int = 1,
    norm_order: float = 2.0,
    value_type: str = "llr",
    num_jobs: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Perform batch-by-batch post-selection analysis for a single parameter combination.

    This function processes each batch individually to avoid memory issues with large datasets.
    For each batch, it loads data, computes statistics, stores only the statistics,
    and deletes the raw data from memory before processing the next batch.

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
    num_jobs : int, default=1
        Number of parallel jobs for sample-level processing.

    Returns
    -------
    Dict[str, np.ndarray]
        Combined post-selection statistics across all batches.
    """
    param_dir = Path(data_dir) / param_combo

    # Find all batch directories
    batch_dirs = [
        d for d in param_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")
    ]

    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found in {param_dir}")

    # Parse parameters from combo string (needed for analyzer initialization)
    parts = param_combo.split("_")
    F = int(parts[-1][1:])  # Extract F from "F1"
    T = int(parts[-4][1:])  # Extract T from "T12"

    batch_stats_list = []

    for batch_dir in sorted(batch_dirs):
        print(f"Processing batch: {batch_dir.name}")

        # Load data for this batch only
        committed_clusters_csr, committed_faults, priors, H, fails = (
            load_single_batch_data(data_dir, param_combo, batch_dir.name)
        )

        # Determine number of faults per window from data structure
        total_faults = committed_clusters_csr.shape[1]
        num_windows = len(committed_faults)
        num_faults_per_window = total_faults // num_windows

        # Create analyzer for this batch
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
            num_jobs=num_jobs,
        )

        # Compute statistics for this batch
        batch_stats = analyzer.compute_postselection_statistics(results, fails)
        # Add batch sample count for accurate total sample tracking
        batch_stats["batch_samples"] = analyzer.num_samples
        batch_stats_list.append(batch_stats)

        # Clear memory - delete large objects
        del committed_clusters_csr, committed_faults, priors, H, fails
        del analyzer, results

        print(f"Completed batch: {batch_dir.name}")

    # Combine statistics from all batches
    combined_stats = combine_batch_statistics(batch_stats_list)

    return combined_stats


def analyze_parameter_combination(
    data_dir: str,
    param_combo: str,
    cutoffs: np.ndarray,
    metric_windows: int = 1,
    norm_order: float = 2.0,
    value_type: str = "llr",
    num_jobs: int = 1,
    stats_only: bool = True,
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
    num_jobs : int, default=1
        Number of parallel jobs for sample-level processing.
    stats_only : bool, default=True
        If True, return only statistics (p_fail, p_abort, etc.). If False,
        return both raw analysis results and statistics.

    Returns
    -------
    Dict[str, np.ndarray]
        Post-selection results for all cutoffs. If stats_only=True, returns
        only statistical results. If stats_only=False, returns both raw
        analysis results and statistics.
    """
    # Load simulation data
    committed_clusters_csr, committed_faults, priors, H, fails = (
        load_sliding_window_data(data_dir, param_combo)
    )

    # Parse parameters from combo string
    parts = param_combo.split("_")
    F = int(parts[-1][1:])  # Extract F from "F1"
    T = int(parts[-4][1:])  # Extract T from "T12"

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
        num_jobs=num_jobs,
    )

    # Compute comprehensive statistics
    statistics = analyzer.compute_postselection_statistics(results, fails)

    if stats_only:
        # Return only statistics
        return statistics
    else:
        # Combine results and statistics
        combined_results = {**results, **statistics}
        return combined_results


def batch_postselection_analysis(
    data_dir: str,
    param_combinations: List[str],
    cutoffs: np.ndarray,
    metric_windows: int = 1,
    norm_order: float = 2.0,
    value_type: str = "llr",
    num_jobs: int = 1,
    batch_mode: bool = False,
    stats_only: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    High-performance batch processing of multiple parameter combinations.

    Efficiently processes multiple sliding window parameter combinations with
    support for memory-efficient batch-by-batch processing for large datasets.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combinations : List[str]
        List of parameter combination strings to process.
    cutoffs : np.ndarray
        Array of cutoff values to test across all combinations.
    metric_windows : int, default=1
        Number of windows for metric evaluation.
    norm_order : float, default=2.0
        Order for L_p norm calculation.
    value_type : str, default="llr"
        Type of cluster value calculation ("size" or "llr").
    num_jobs : int, default=1
        Number of parallel jobs for sample-level processing.
    batch_mode : bool, default=False
        If True, use batch-by-batch processing to handle large datasets that
        exceed memory capacity. Each batch is processed individually with
        statistics combined at the end. If False, load all batches at once
        (original behavior).
    stats_only : bool, default=True
        If True, return only statistics (p_fail, p_abort, etc.). If False,
        return both raw results and statistics. Note: stats_only=False is not
        supported when batch_mode=True due to memory constraints.
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Nested dictionary with results for each parameter combination.
        Structure: {param_combo: {result_key: result_array}}

        If stats_only=True (default), only statistical results are returned:
        - 'p_fail', 'delta_p_fail', 'p_abort', 'effective_avg_trials',
          'num_accepted', 'num_failed_accepted', 'cutoffs'

        If stats_only=False, both raw analysis results and statistics are returned:
        - All statistical results (above) plus raw results like 'abort_windows',
          'effective_trials', 'accepted_mask', 'metrics_matrix', etc.
    """
    # Validate parameter combination
    if batch_mode and not stats_only:
        raise ValueError(
            "stats_only=False is not supported when batch_mode=True due to memory constraints. "
            "Raw results would be too large to store in memory for batch processing."
        )

    if verbose:
        mode_str = "batch-by-batch" if batch_mode else "all-at-once"
        output_str = "statistics only" if stats_only else "full results + statistics"
        print(
            f"Starting {mode_str} post-selection analysis for {len(param_combinations)} combinations"
        )
        print(
            f"Testing {len(cutoffs)} cutoff values with {num_jobs} parallel jobs for sample processing"
        )
        print(f"Output mode: {output_str}")
        if batch_mode:
            print("Using memory-efficient batch-by-batch processing mode")

    # Process combinations sequentially (since typically only 1 subdir) but with parallel sample processing
    results_list = []
    for param_combo in param_combinations:
        if verbose:
            print(f"Processing {param_combo}...")

        if batch_mode:
            # Use batch-by-batch processing for memory efficiency
            results = analyze_parameter_combination_batch_by_batch(
                data_dir=data_dir,
                param_combo=param_combo,
                cutoffs=cutoffs,
                metric_windows=metric_windows,
                norm_order=norm_order,
                value_type=value_type,
                num_jobs=num_jobs,
            )
        else:
            # Use all-at-once processing
            results = analyze_parameter_combination(
                data_dir=data_dir,
                param_combo=param_combo,
                cutoffs=cutoffs,
                metric_windows=metric_windows,
                norm_order=norm_order,
                value_type=value_type,
                num_jobs=num_jobs,
                stats_only=stats_only,
            )

        results_list.append((param_combo, results))

    # Convert to dictionary
    results_dict = {combo: results for combo, results in results_list}

    if verbose:
        successful = sum(1 for _, results in results_list if results)
        print(
            f"Successfully processed {successful}/{len(param_combinations)} combinations"
        )

    return results_dict
