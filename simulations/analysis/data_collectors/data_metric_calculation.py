import os
from typing import Tuple
import time

import numpy as np
import pandas as pd
from scipy import sparse

from simulations.analysis.data_collectors.utils import (
    calculate_cluster_norms_from_flat_data,
    calculate_cluster_metrics_from_csr,
    calculate_window_cluster_norm_fracs_from_csr,
    calculate_committed_cluster_norm_fractions_from_csr,
)


def _detect_data_format(
    batch_dir_path: str, by: str, verbose: bool
) -> Tuple[bool, bool, bool]:
    """
    Detect which data format is used in the batch directory.

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    by : str
        The aggregation method.
    verbose : bool
        If True, prints detailed format information.

    Returns
    -------
    use_sliding_window_format : bool
        True if sliding window format is detected.
    use_legacy_format : bool
        True if legacy format is detected.
    use_new_format : bool
        True if new format is detected.
    """
    # File paths
    scalars_path = os.path.join(batch_dir_path, "scalars.feather")

    # Legacy format paths
    cluster_sizes_path = os.path.join(batch_dir_path, "cluster_sizes.npy")
    cluster_llrs_path = os.path.join(batch_dir_path, "cluster_llrs.npy")
    offsets_path = os.path.join(batch_dir_path, "offsets.npy")

    # New format paths
    clusters_path = os.path.join(batch_dir_path, "clusters.npz")

    # Sliding window format paths
    fails_path = os.path.join(batch_dir_path, "fails.npy")
    # New sliding window format (CSR-based)
    all_clusters_path = os.path.join(batch_dir_path, "all_clusters.npz")
    committed_clusters_path = os.path.join(batch_dir_path, "committed_clusters.npz")

    # Check for scalars.feather first, as it's always needed for regular format
    # For sliding window format, we'll use fails.npy instead
    use_sliding_window_format = False
    use_legacy_format = False
    use_new_format = False

    if not os.path.isfile(scalars_path):
        # Check if this is sliding window format
        if os.path.isfile(fails_path):
            # Check for new CSR-based sliding window format
            if os.path.isfile(all_clusters_path) and os.path.isfile(
                committed_clusters_path
            ):
                use_sliding_window_format = True
                if verbose:
                    print(
                        f"  Using CSR-based sliding window format for {batch_dir_path}"
                    )
            else:
                if verbose:
                    print(
                        f"  Error: fails.npy found but missing sliding window cluster files in {batch_dir_path}. Cannot process this batch."
                    )
                raise FileNotFoundError(
                    f"fails.npy found but missing sliding window cluster files in {batch_dir_path}. Cannot process this batch."
                )
        else:
            if verbose:
                print(
                    f"  Error: Neither scalars.feather nor sliding window format files found in {batch_dir_path}. Cannot process this batch."
                )
            # Raise FileNotFoundError directly as per new requirement
            raise FileNotFoundError(
                f"Neither scalars.feather nor sliding window format files found in {batch_dir_path}. Cannot process this batch."
            )

    # Check for cluster data availability
    if "cluster" in by and not use_sliding_window_format:
        # Check legacy format first
        legacy_files_exist = all(
            os.path.isfile(fpath)
            for fpath in [cluster_sizes_path, cluster_llrs_path, offsets_path]
        )
        new_format_exists = os.path.isfile(clusters_path)

        if legacy_files_exist:
            use_legacy_format = True
            if verbose:
                print(f"  Using legacy format for {batch_dir_path}")
        elif new_format_exists:
            use_new_format = True
            if verbose:
                print(f"  Using new format for {batch_dir_path}")
        else:
            raise FileNotFoundError(
                f"Error: Neither legacy format (cluster_sizes.npy, cluster_llrs.npy, offsets.npy) "
                f"nor new format (clusters.npz) found for 'by={by}' method in {batch_dir_path}."
            )

    return use_sliding_window_format, use_legacy_format, use_new_format


def _load_scalars_data(
    batch_dir_path: str, use_sliding_window: bool, verbose: bool
) -> Tuple[pd.DataFrame, int]:
    """
    Load scalars data from either scalars.feather or sliding window format.

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    use_sliding_window : bool
        Whether to use sliding window format.
    verbose : bool
        If True, prints detailed loading information.

    Returns
    -------
    df_scalars : pd.DataFrame
        The DataFrame containing scalar data.
    original_batch_size : int
        The original size of the batch before any filtering.
    """
    if use_sliding_window:
        # Load fails.npy for sliding window format
        fails_path = os.path.join(batch_dir_path, "fails.npy")
        fails = np.load(fails_path)
        original_batch_size = len(fails)

        # Create minimal df_scalars with fail column
        df_scalars = pd.DataFrame({"fail": fails})

        if verbose:
            print(f"  Loaded {len(fails)} samples from sliding window format")
    else:
        # Load regular scalars.feather
        scalars_path = os.path.join(batch_dir_path, "scalars.feather")
        df_scalars = pd.read_feather(scalars_path)
        original_batch_size = len(
            df_scalars
        )  # Store original size before any filtering

        if df_scalars.empty:
            if verbose:
                print(f"  Warning: scalars.feather in {batch_dir_path} is empty.")
            # Still return the empty df_scalars as it exists, the caller can decide to skip.
            # The series_to_bin will likely be empty or None.

    return df_scalars, original_batch_size


def _filter_samples_by_indices(
    df_scalars: pd.DataFrame,
    sample_indices: np.ndarray | None,
    batch_start_idx: int,
    verbose: bool,
) -> pd.DataFrame | None:
    """
    Filter samples based on sample_indices if provided.

    Parameters
    ----------
    df_scalars : pd.DataFrame
        DataFrame containing scalar metrics for each sample.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include. If None, no filtering is applied.
    batch_start_idx : int
        Starting global index for this batch.
    verbose : bool
        If True, prints filtering information.

    Returns
    -------
    df_filtered : pd.DataFrame or None
        Filtered DataFrame with only requested samples, or None if no samples match.
    """
    if sample_indices is None or df_scalars.empty:
        return df_scalars

    batch_end_idx = batch_start_idx + len(df_scalars)

    # Find which global indices fall within this batch's range
    mask = (sample_indices >= batch_start_idx) & (sample_indices < batch_end_idx)
    batch_sample_indices = sample_indices[mask]

    if len(batch_sample_indices) == 0:
        # No samples from this batch are in the requested indices
        if verbose:
            print(
                f"  No requested samples found in batch (range: {batch_start_idx}-{batch_end_idx-1})"
            )
        return None

    # Convert global indices to local batch indices
    local_indices = batch_sample_indices - batch_start_idx

    # Filter the dataframe to only include requested samples
    df_filtered = df_scalars.iloc[local_indices].reset_index(drop=True)

    if verbose:
        print(f"  Filtered to {len(df_filtered)} samples from batch")

    return df_filtered


def _load_and_calculate_sliding_window_metrics(
    batch_dir_path: str,
    by: str,
    norm_order: float | None,
    sample_indices: np.ndarray | None,
    batch_start_idx: int,
    original_batch_size: int,
    priors: np.ndarray | None = None,
    eval_windows: Tuple[int, int] | None = None,
    adj_matrix: np.ndarray | None = None,
    num_jobs: int = 1,
    num_batches: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load and calculate metrics for sliding window format data using CSR-based format.

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    by : str
        The aggregation method for CSR-based sliding window format.
    norm_order : float, optional
        Order for L_p norm calculation.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include.
    batch_start_idx : int
        Starting global index for this batch.
    original_batch_size : int
        Original size of the batch.
    priors : np.ndarray, optional
        Prior probabilities for each fault. Required for CSR-based format.
    eval_windows : tuple of int, optional
        If provided, only consider windows from init_eval_window to final_eval_window.
    adj_matrix : np.ndarray, optional
        Adjacency matrix for cluster labeling. Required for committed cluster metrics.

    Returns
    -------
    inside_cluster_size_norms : np.ndarray
        Array of cluster size norm fractions.
    inside_cluster_llr_norms : np.ndarray
        Array of cluster LLR norm fractions.
    timing_info : dict
        Dictionary containing timing information.
    """
    timing_info = {
        "cluster_file_load_time": 0.0,
        "cluster_calculation_time": 0.0,
    }

    # Handle CSR-based sliding window format
    return _handle_new_csr_sliding_window_metrics(
        batch_dir_path,
        by,
        norm_order,
        sample_indices,
        batch_start_idx,
        original_batch_size,
        priors,
        eval_windows,
        adj_matrix,
        timing_info,
        num_jobs,
        num_batches,
    )


def _handle_new_csr_sliding_window_metrics(
    batch_dir_path: str,
    by: str,
    norm_order: float | None,
    sample_indices: np.ndarray | None,
    batch_start_idx: int,
    original_batch_size: int,
    priors: np.ndarray | None,
    eval_windows: Tuple[int, int] | None,
    adj_matrix: np.ndarray | None,
    timing_info: dict,
    num_jobs: int = 1,
    num_batches: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Handle new CSR-based sliding window metrics.

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    by : str
        The aggregation method for CSR-based sliding window format.
    norm_order : float, optional
        Order for L_p norm calculation.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include.
    batch_start_idx : int
        Starting global index for this batch.
    original_batch_size : int
        Original size of the batch.
    priors : np.ndarray, optional
        Prior probabilities for each fault. Required for CSR-based format.
    eval_windows : tuple of int, optional
        If provided, only consider windows from init_eval_window to final_eval_window.
    adj_matrix : np.ndarray, optional
        Adjacency matrix for cluster labeling. Required for committed cluster metrics.
    timing_info : dict
        Dictionary containing timing information.
    num_jobs : int, optional
        Number of parallel processes to use for multiprocessing. Default is 1 (sequential processing).
        Currently only supported for committed cluster norm fraction calculations.
    num_batches : int, optional
        Number of batches to split samples into for parallel processing. If None, defaults to num_jobs.
        Currently only supported for committed cluster norm fraction calculations.

    Returns
    -------
    inside_cluster_size_norms : np.ndarray
        Array of cluster size norm fractions.
    inside_cluster_llr_norms : np.ndarray
        Array of cluster LLR norm fractions.
    timing_info : dict
        Dictionary containing timing information.
    """
    if priors is None:
        raise ValueError("priors is required for new CSR-based sliding window format")

    # Parse metric type
    if "avg_window_cluster_" in by or "max_window_cluster_" in by:
        # Category 1: window-based metrics using all_clusters
        if "avg_window_cluster_" in by:
            aggregation_type = "avg"
            remaining = by.replace("avg_window_cluster_", "")
        else:
            aggregation_type = "max"
            remaining = by.replace("max_window_cluster_", "")

        # Extract value type (size or llr)
        if remaining.startswith("size_norm_frac"):
            value_type = "size"
        elif remaining.startswith("llr_norm_frac"):
            value_type = "llr"
        else:
            raise ValueError(f"Unknown value type in metric: {by}")

        # Load all_clusters.npz
        start_file_load = time.perf_counter()
        all_clusters_path = os.path.join(batch_dir_path, "all_clusters.npz")
        all_clusters_csr = sparse.load_npz(all_clusters_path)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        # Filter samples if needed
        if sample_indices is not None:
            batch_end_idx = batch_start_idx + original_batch_size
            mask = (sample_indices >= batch_start_idx) & (
                sample_indices < batch_end_idx
            )
            batch_sample_indices = sample_indices[mask]
            local_indices = batch_sample_indices - batch_start_idx
            all_clusters_csr = all_clusters_csr[local_indices, :]

        # Calculate metrics
        start_calc = time.perf_counter()
        norm_fractions = calculate_window_cluster_norm_fracs_from_csr(
            all_clusters_csr,
            priors,
            norm_order,
            value_type,
            aggregation_type,
            eval_windows,
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

        # Initialize arrays
        num_samples = len(norm_fractions)
        inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
        inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)

        # Store results based on metric type
        if value_type == "size":
            inside_cluster_size_norms = norm_fractions
        else:  # llr
            inside_cluster_llr_norms = norm_fractions

    elif "committed_cluster_" in by:
        # Category 2: committed cluster metrics using committed_clusters
        if "committed_cluster_size_norm_frac" in by:
            value_type = "size"
        elif "committed_cluster_llr_norm_frac" in by:
            value_type = "llr"
        else:
            raise ValueError(f"Unknown committed cluster metric: {by}")

        if adj_matrix is None:
            raise ValueError("adj_matrix is required for committed cluster metrics")

        # Load committed_clusters.npz
        start_file_load = time.perf_counter()
        committed_clusters_path = os.path.join(batch_dir_path, "committed_clusters.npz")
        committed_clusters_csr = sparse.load_npz(committed_clusters_path)

        # Load committed_faults.npz from parent directory (configuration level)
        sub_data_dir = os.path.dirname(batch_dir_path)
        committed_faults_path = os.path.join(sub_data_dir, "committed_faults.npz")
        committed_faults_data = np.load(committed_faults_path)
        committed_faults = [
            committed_faults_data[f"arr_{i}"]
            for i in range(len(committed_faults_data.files))
        ]

        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        # Filter samples if needed
        if sample_indices is not None:
            batch_end_idx = batch_start_idx + original_batch_size
            mask = (sample_indices >= batch_start_idx) & (
                sample_indices < batch_end_idx
            )
            batch_sample_indices = sample_indices[mask]
            local_indices = batch_sample_indices - batch_start_idx
            committed_clusters_csr = committed_clusters_csr[local_indices, :]

        # Calculate metrics
        start_calc = time.perf_counter()
        norm_fractions = calculate_committed_cluster_norm_fractions_from_csr(
            committed_clusters_csr,
            committed_faults,
            priors,
            adj_matrix,
            norm_order,
            value_type,
            eval_windows,
            num_jobs=num_jobs,
            num_batches=num_batches,
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

        # Initialize arrays
        num_samples = len(norm_fractions)
        inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
        inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)

        # Store results based on metric type
        if value_type == "size":
            inside_cluster_size_norms = norm_fractions
        else:  # llr
            inside_cluster_llr_norms = norm_fractions

    else:
        raise ValueError(f"Unsupported new CSR sliding window metric: {by}")

    return inside_cluster_size_norms, inside_cluster_llr_norms, timing_info


def _load_and_calculate_legacy_metrics(
    batch_dir_path: str,
    by: str,
    norm_order: float | None,
    local_sample_indices: np.ndarray | None,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """
    Load and calculate metrics for legacy format data.

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    by : str
        The aggregation method.
    norm_order : float, optional
        Order for L_p norm calculation.
    local_sample_indices : np.ndarray, optional
        Array of local sample indices to include.
    num_samples : int
        Number of samples in the batch.

    Returns
    -------
    inside_cluster_size_norms : np.ndarray
        Array of cluster size norms.
    inside_cluster_llr_norms : np.ndarray
        Array of cluster LLR norms.
    cluster_metrics : np.ndarray
        Array of calculated cluster metrics.
    outside_value : float
        Outside value for gap calculations.
    timing_info : dict
        Dictionary containing timing information.
    """
    # Legacy format paths
    cluster_sizes_path = os.path.join(batch_dir_path, "cluster_sizes.npy")
    cluster_llrs_path = os.path.join(batch_dir_path, "cluster_llrs.npy")
    offsets_path = os.path.join(batch_dir_path, "offsets.npy")

    timing_info = {
        "cluster_file_load_time": 0.0,
        "cluster_calculation_time": 0.0,
    }

    # Initialize arrays
    inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
    inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)
    cluster_metrics = np.full(num_samples, np.nan, dtype=float)
    outside_value = np.nan

    # Load offsets first (always needed)
    offsets = np.load(offsets_path, allow_pickle=False)[:-1]

    if by in ["cluster_size_norm", "cluster_size_norm_gap"]:
        start_file_load = time.perf_counter()
        cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        start_calc = time.perf_counter()
        inside_cluster_size_norms, outside_value = (
            calculate_cluster_norms_from_flat_data(
                flat_data=cluster_sizes_flat,
                offsets=offsets,
                norm_order=norm_order,
                sample_indices=local_sample_indices,
            )
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    elif by in ["cluster_llr_norm", "cluster_llr_norm_gap"]:
        start_file_load = time.perf_counter()
        cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        start_calc = time.perf_counter()
        inside_cluster_llr_norms, outside_value = (
            calculate_cluster_norms_from_flat_data(
                flat_data=cluster_llrs_flat,
                offsets=offsets,
                norm_order=norm_order,
                sample_indices=local_sample_indices,
            )
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    elif by in ["cluster_llr_residual_sum", "cluster_llr_residual_sum_gap"]:
        # For residual sum methods, calculate cluster_llr_norm with order=1
        start_file_load = time.perf_counter()
        cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        start_calc = time.perf_counter()
        inside_cluster_llr_norms, outside_value = (
            calculate_cluster_norms_from_flat_data(
                flat_data=cluster_llrs_flat,
                offsets=offsets,
                norm_order=1.0,  # Always use L1 norm for residual sum
                sample_indices=local_sample_indices,
            )
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    elif by == "average_cluster_size":
        # Calculate (2-norm)^2 / (1-norm) for cluster sizes
        start_file_load = time.perf_counter()
        cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        start_calc = time.perf_counter()
        cluster_metrics = _calculate_average_cluster_metrics(
            flat_data=cluster_sizes_flat,
            offsets=offsets,
            local_sample_indices=local_sample_indices,
            num_samples=num_samples,
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    elif by == "average_cluster_llr":
        # Calculate (2-norm)^2 / (1-norm) for cluster LLRs
        start_file_load = time.perf_counter()
        cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        start_calc = time.perf_counter()
        cluster_metrics = _calculate_average_cluster_metrics(
            flat_data=cluster_llrs_flat,
            offsets=offsets,
            local_sample_indices=local_sample_indices,
            num_samples=num_samples,
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    elif by == "cluster_size_norm_frac":
        # Calculate cluster_size_norm / (number of faults)
        start_file_load = time.perf_counter()
        cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        start_calc = time.perf_counter()
        # Calculate norm at requested order
        inside_cluster_size_norms, outside_value = (
            calculate_cluster_norms_from_flat_data(
                flat_data=cluster_sizes_flat,
                offsets=offsets,
                norm_order=norm_order,
                sample_indices=local_sample_indices,
            )
        )

        # Calculate norm at order=1 for denominator (number of faults)
        if norm_order == 1.0:
            inside_cluster_size_norms_1 = inside_cluster_size_norms
        else:
            inside_cluster_size_norms_1, _ = calculate_cluster_norms_from_flat_data(
                flat_data=cluster_sizes_flat,
                offsets=offsets,
                norm_order=1.0,
                sample_indices=local_sample_indices,
            )

        # Calculate fractions: norm / (outside_value + inside_norm_1)
        cluster_metrics = np.full(num_samples, np.nan, dtype=float)
        denominators = outside_value + inside_cluster_size_norms_1
        valid_mask = denominators > 0
        cluster_metrics[valid_mask] = (
            inside_cluster_size_norms[valid_mask] / denominators[valid_mask]
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    elif by == "cluster_llr_norm_frac":
        # Calculate cluster_llr_norm / (summation of LLRs)
        start_file_load = time.perf_counter()
        cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
        timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

        start_calc = time.perf_counter()
        # Calculate norm at requested order
        inside_cluster_llr_norms, outside_value = (
            calculate_cluster_norms_from_flat_data(
                flat_data=cluster_llrs_flat,
                offsets=offsets,
                norm_order=norm_order,
                sample_indices=local_sample_indices,
            )
        )

        # Calculate norm at order=1 for denominator (summation of LLRs)
        if norm_order == 1.0:
            inside_cluster_llr_norms_1 = inside_cluster_llr_norms
        else:
            inside_cluster_llr_norms_1, _ = calculate_cluster_norms_from_flat_data(
                flat_data=cluster_llrs_flat,
                offsets=offsets,
                norm_order=1.0,
                sample_indices=local_sample_indices,
            )

        # Calculate fractions: norm / (outside_value + inside_norm_1)
        cluster_metrics = np.full(num_samples, np.nan, dtype=float)
        denominators = outside_value + inside_cluster_llr_norms_1
        valid_mask = denominators > 0
        cluster_metrics[valid_mask] = (
            inside_cluster_llr_norms[valid_mask] / denominators[valid_mask]
        )
        timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    else:
        raise ValueError(f"Unsupported method ({by}) for legacy data structure.")

    return (
        inside_cluster_size_norms,
        inside_cluster_llr_norms,
        cluster_metrics,
        outside_value,
        timing_info,
    )


def _calculate_average_cluster_metrics(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    local_sample_indices: np.ndarray | None,
    num_samples: int,
) -> np.ndarray:
    """
    Calculate average cluster metrics using (2-norm)^2 / (1-norm) formula.

    Parameters
    ----------
    flat_data : np.ndarray
        Flattened cluster data (sizes or LLRs).
    offsets : np.ndarray
        Array of offsets for cluster boundaries.
    local_sample_indices : np.ndarray, optional
        Array of local sample indices to include.
    num_samples : int
        Number of samples in the batch.

    Returns
    -------
    cluster_metrics : np.ndarray
        Array of calculated average cluster metrics.
    """
    # Calculate 2-norm
    inside_cluster_2norms, _ = calculate_cluster_norms_from_flat_data(
        flat_data=flat_data,
        offsets=offsets,
        norm_order=2.0,
        sample_indices=local_sample_indices,
    )
    # Calculate 1-norm
    inside_cluster_1norms, _ = calculate_cluster_norms_from_flat_data(
        flat_data=flat_data,
        offsets=offsets,
        norm_order=1.0,
        sample_indices=local_sample_indices,
    )
    # Calculate average: (2-norm)^2 / (1-norm)
    cluster_metrics = np.full(num_samples, np.nan, dtype=float)
    valid_mask = inside_cluster_1norms > 0
    cluster_metrics[valid_mask] = (
        inside_cluster_2norms[valid_mask] ** 2
    ) / inside_cluster_1norms[valid_mask]

    return cluster_metrics


def _load_and_calculate_new_format_metrics(
    batch_dir_path: str,
    by: str,
    norm_order: float | None,
    priors: np.ndarray | None,
    sample_indices: np.ndarray | None,
    batch_start_idx: int,
    df_scalars: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """
    Load and calculate metrics for new format data (CSR matrix).

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    by : str
        The aggregation method.
    norm_order : float, optional
        Order for L_p norm calculation.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include.
    batch_start_idx : int
        Starting global index for this batch.
    df_scalars : pd.DataFrame
        DataFrame containing scalar data.

    Returns
    -------
    inside_cluster_size_norms : np.ndarray
        Array of cluster size norms.
    inside_cluster_llr_norms : np.ndarray
        Array of cluster LLR norms.
    cluster_metrics : np.ndarray
        Array of calculated cluster metrics.
    outside_values : dict
        Dictionary containing outside values for gap calculations.
    timing_info : dict
        Dictionary containing timing information.
    """
    # New format path
    clusters_path = os.path.join(batch_dir_path, "clusters.npz")

    timing_info = {
        "cluster_file_load_time": 0.0,
        "cluster_calculation_time": 0.0,
    }

    # Initialize arrays
    num_samples = len(df_scalars)
    inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
    inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)
    cluster_metrics = np.full(num_samples, np.nan, dtype=float)
    outside_values = {}

    # Load clusters CSR matrix
    start_file_load = time.perf_counter()
    clusters_csr = sparse.load_npz(clusters_path)
    timing_info["cluster_file_load_time"] = time.perf_counter() - start_file_load

    # If samples are filtered, we need to filter the CSR matrix rows
    if sample_indices is not None and not df_scalars.empty:
        scalars_path = os.path.join(batch_dir_path, "scalars.feather")
        original_df = pd.read_feather(scalars_path)
        batch_end_idx = batch_start_idx + len(original_df)
        mask = (sample_indices >= batch_start_idx) & (sample_indices < batch_end_idx)
        batch_sample_indices = sample_indices[mask]
        local_indices = batch_sample_indices - batch_start_idx

        # Filter the CSR matrix to only include selected samples
        clusters_csr = clusters_csr[local_indices, :]

    # Calculate norms directly from CSR format using Numba
    start_calc = time.perf_counter()

    if by in ["cluster_size_norm", "cluster_size_norm_gap"]:
        inside_cluster_size_norms, outside_size_values = (
            calculate_cluster_metrics_from_csr(
                clusters=clusters_csr,
                method="norm",
                priors=priors,  # Zeros for size calculations
                norm_order=norm_order,
            )
        )
        outside_values["size"] = outside_size_values

    elif by in ["cluster_llr_norm", "cluster_llr_norm_gap"]:
        inside_cluster_llr_norms, outside_llr_values = (
            calculate_cluster_metrics_from_csr(
                clusters=clusters_csr,
                method="llr_norm",
                priors=priors,
                norm_order=norm_order,
            )
        )
        outside_values["llr"] = outside_llr_values

    elif by in ["cluster_llr_residual_sum", "cluster_llr_residual_sum_gap"]:
        # For residual sum methods, calculate cluster_llr_norm with order=1
        inside_cluster_llr_norms, outside_llr_values = (
            calculate_cluster_metrics_from_csr(
                clusters=clusters_csr,
                method="llr_norm",
                priors=priors,
                norm_order=1.0,  # Always use L1 norm for residual sum
            )
        )
        outside_values["llr"] = outside_llr_values

    elif by == "average_cluster_size":
        # Calculate (2-norm)^2 / (1-norm) for cluster sizes
        # Calculate 2-norm
        inside_cluster_2norms, _ = calculate_cluster_metrics_from_csr(
            clusters=clusters_csr,
            method="norm",
            priors=None,  # priors not needed for size calculations
            norm_order=2.0,
        )
        # Calculate 1-norm
        inside_cluster_1norms, _ = calculate_cluster_metrics_from_csr(
            clusters=clusters_csr,
            method="norm",
            priors=None,  # priors not needed for size calculations
            norm_order=1.0,
        )
        # Calculate average: (2-norm)^2 / (1-norm)
        cluster_metrics = np.full(len(df_scalars), np.nan, dtype=float)
        valid_mask = inside_cluster_1norms > 0
        cluster_metrics[valid_mask] = (
            inside_cluster_2norms[valid_mask] ** 2
        ) / inside_cluster_1norms[valid_mask]

    elif by == "average_cluster_llr":
        # Calculate (2-norm)^2 / (1-norm) for cluster LLRs
        # Calculate 2-norm
        inside_cluster_2norms, _ = calculate_cluster_metrics_from_csr(
            clusters=clusters_csr,
            method="llr_norm",
            priors=priors,
            norm_order=2.0,
        )
        # Calculate 1-norm
        inside_cluster_1norms, _ = calculate_cluster_metrics_from_csr(
            clusters=clusters_csr,
            method="llr_norm",
            priors=priors,
            norm_order=1.0,
        )
        # Calculate average: (2-norm)^2 / (1-norm)
        cluster_metrics = np.full(len(df_scalars), np.nan, dtype=float)
        valid_mask = inside_cluster_1norms > 0
        cluster_metrics[valid_mask] = (
            inside_cluster_2norms[valid_mask] ** 2
        ) / inside_cluster_1norms[valid_mask]

    elif by == "cluster_size_norm_frac":
        # Calculate cluster_size_norm / (number of faults)
        inside_cluster_size_norms, _ = calculate_cluster_metrics_from_csr(
            clusters=clusters_csr,
            method="norm",
            priors=None,  # priors not needed for size calculations
            norm_order=norm_order,
        )
        # Calculate fractions: norm / total_faults
        cluster_metrics = np.full(len(df_scalars), np.nan, dtype=float)
        total_faults = clusters_csr.shape[1]  # Number of columns
        if total_faults > 0:
            cluster_metrics = inside_cluster_size_norms / total_faults

    elif by == "cluster_llr_norm_frac":
        # Calculate cluster_llr_norm / (summation of LLRs)
        if priors is None:
            raise ValueError("priors is required for cluster_llr_norm_frac calculation")

        inside_cluster_llr_norms, _ = calculate_cluster_metrics_from_csr(
            clusters=clusters_csr,
            method="llr_norm",
            priors=priors,
            norm_order=norm_order,
        )
        # Calculate fractions: norm / total_llr_sum
        cluster_metrics = np.full(len(df_scalars), np.nan, dtype=float)
        bit_llrs = np.log((1 - priors) / priors)
        total_llr_sum = np.sum(bit_llrs)
        if total_llr_sum > 0:
            cluster_metrics = inside_cluster_llr_norms / total_llr_sum

    else:
        cluster_metrics = calculate_cluster_metrics_from_csr(
            clusters=clusters_csr,
            method=by,
            priors=priors,
        )

    timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

    return (
        inside_cluster_size_norms,
        inside_cluster_llr_norms,
        cluster_metrics,
        outside_values,
        timing_info,
    )


def _create_aggregation_series(
    by: str,
    df_scalars: pd.DataFrame,
    inside_cluster_size_norms: np.ndarray,
    inside_cluster_llr_norms: np.ndarray,
    cluster_metrics: np.ndarray,
    use_sliding_window: bool,
    use_new_format: bool,
    outside_value: float = np.nan,
    outside_values: dict = None,
) -> pd.Series:
    """
    Create the final aggregation series based on the aggregation method.

    Parameters
    ----------
    by : str
        The aggregation method.
    df_scalars : pd.DataFrame
        DataFrame containing scalar data.
    inside_cluster_size_norms : np.ndarray
        Array of cluster size norms.
    inside_cluster_llr_norms : np.ndarray
        Array of cluster LLR norms.
    cluster_metrics : np.ndarray
        Array of calculated cluster metrics.
    use_sliding_window : bool
        Whether sliding window format is used.
    use_new_format : bool
        Whether new format is used.
    outside_value : float, optional
        Outside value for gap calculations (legacy format).
    outside_values : dict, optional
        Dictionary containing outside values for gap calculations (new format).

    Returns
    -------
    series_to_bin : pd.Series
        The Series containing values to be binned.
    """
    if outside_values is None:
        outside_values = {}

    # Handle sliding window metrics
    if use_sliding_window:
        # Handle CSR-based sliding window format metrics
        if (
            "avg_window_cluster_" in by
            or "max_window_cluster_" in by
            or "committed_cluster_" in by
        ) and "_norm_frac" in by:
            # CSR format: metrics are already calculated in the correct arrays
            if "size_norm_frac" in by:
                return pd.Series(inside_cluster_size_norms, index=df_scalars.index)
            elif "llr_norm_frac" in by:
                return pd.Series(inside_cluster_llr_norms, index=df_scalars.index)

    # Handle regular cluster methods
    elif by == "cluster_size_norm":
        return pd.Series(inside_cluster_size_norms, index=df_scalars.index)

    elif by == "cluster_llr_norm":
        return pd.Series(inside_cluster_llr_norms, index=df_scalars.index)

    elif by == "cluster_size_norm_gap":
        if use_new_format:
            return pd.Series(
                outside_values["size"] - inside_cluster_size_norms,
                index=df_scalars.index,
            )
        else:
            return pd.Series(
                outside_value - inside_cluster_size_norms, index=df_scalars.index
            )

    elif by == "cluster_llr_norm_gap":
        if use_new_format:
            return pd.Series(
                outside_values["llr"] - inside_cluster_llr_norms,
                index=df_scalars.index,
            )
        else:
            return pd.Series(
                outside_value - inside_cluster_llr_norms, index=df_scalars.index
            )

    elif by == "cluster_llr_residual_sum":
        # cluster_llr_norm (order=1) - pred_llr
        if "pred_llr" not in df_scalars.columns:
            raise ValueError(
                f"'pred_llr' column not found in scalars.feather. Required for cluster_llr_residual_sum."
            )
        pred_llr_values = df_scalars["pred_llr"].values
        cluster_llr_residual_sum_values = inside_cluster_llr_norms - pred_llr_values
        return pd.Series(cluster_llr_residual_sum_values, index=df_scalars.index)

    elif by == "cluster_llr_residual_sum_gap":
        # outside_llr_value - cluster_llr_residual_sum
        if "pred_llr" not in df_scalars.columns:
            raise ValueError(
                f"'pred_llr' column not found in scalars.feather. Required for cluster_llr_residual_sum_gap."
            )
        pred_llr_values = df_scalars["pred_llr"].values
        cluster_llr_residual_sum_values = inside_cluster_llr_norms - pred_llr_values
        if use_new_format:
            return pd.Series(
                outside_values["llr"] - cluster_llr_residual_sum_values,
                index=df_scalars.index,
            )
        else:
            return pd.Series(
                outside_value - cluster_llr_residual_sum_values,
                index=df_scalars.index,
            )

    elif by in [
        "average_cluster_size",
        "average_cluster_llr",
        "cluster_size_norm_frac",
        "cluster_llr_norm_frac",
    ]:
        return pd.Series(cluster_metrics, index=df_scalars.index)

    elif "cluster" in by:
        # For other cluster methods, use cluster_metrics
        return pd.Series(cluster_metrics, index=df_scalars.index)

    else:
        # 'by' is not a cluster-dependent method, try to get column directly
        if by in df_scalars.columns:
            return df_scalars[by].copy()
        else:
            raise ValueError(f"Column '{by}' not found in scalars data.")


def get_values_for_binning_from_batch(
    batch_dir_path: str,
    by: str,
    norm_order: float | None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    batch_start_idx: int = 0,
    verbose: bool = False,
    eval_windows: Tuple[int, int] | None = None,
    adj_matrix: np.ndarray | None = None,
    num_jobs: int = 1,
    num_batches: int | None = None,
) -> Tuple[pd.Series | None, pd.DataFrame | None, int, dict]:
    """
    Loads data from a single batch directory needed for a specific aggregation method ('by').

    This function loads 'scalars.feather' and, if required by the 'by' method,
    also loads cluster data. It supports legacy format (cluster_sizes.npy, cluster_llrs.npy, offsets.npy),
    new format (clusters.npz with on-the-fly calculation), and sliding window CSR format.

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    by : str
        The aggregation method.
    norm_order : float, optional
        Order for L_p norm calculation, required for norm-based 'by' methods.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit, required for cluster_llr calculations when using new format.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include. If None, all samples are included.
    batch_start_idx : int, optional
        Starting global index for this batch. Defaults to 0.
    verbose : bool, optional
        If True, prints detailed loading information.
    eval_windows : tuple of int, optional
        If provided, only consider windows from init_eval_window to final_eval_window for sliding window metrics.
    adj_matrix : np.ndarray, optional
        Adjacency matrix for cluster labeling. Required for committed cluster metrics in sliding window format.
    num_jobs : int, optional
        Number of parallel processes to use for multiprocessing. Default is 1 (sequential processing).
        Currently only supported for committed cluster norm fraction calculations in sliding window decoding.
    num_batches : int, optional
        Number of batches to split samples into for parallel processing. If None, defaults to num_jobs.
        Currently only supported for committed cluster norm fraction calculations in sliding window decoding.

    Returns
    -------
    series_to_bin : pd.Series or None
        The Series containing values to be binned. None if essential files are missing
        or data cannot be computed.
    df_scalars : pd.DataFrame or None
        The DataFrame loaded from 'scalars.feather'. None if 'scalars.feather' is missing.
    original_batch_size : int
        The original size of the batch before any filtering.
    timing_info : dict
        Dictionary containing detailed timing information for different steps.
    """
    # Initialize timing
    timing_info = {
        "file_check_time": 0.0,
        "scalars_load_time": 0.0,
        "sample_filtering_time": 0.0,
        "cluster_file_load_time": 0.0,
        "cluster_calculation_time": 0.0,
        "series_creation_time": 0.0,
    }

    start_time = time.perf_counter()

    # 1. Detect data format
    use_sliding_window_format, use_legacy_format, use_new_format = _detect_data_format(
        batch_dir_path, by, verbose
    )
    timing_info["file_check_time"] = time.perf_counter() - start_time

    # 2. Load scalars data
    start_scalars_load = time.perf_counter()
    df_scalars, original_batch_size = _load_scalars_data(
        batch_dir_path, use_sliding_window_format, verbose
    )
    timing_info["scalars_load_time"] = time.perf_counter() - start_scalars_load

    if df_scalars.empty:
        return None, df_scalars, original_batch_size, timing_info

    # 3. Filter samples by indices
    start_filtering = time.perf_counter()
    df_scalars_filtered = _filter_samples_by_indices(
        df_scalars, sample_indices, batch_start_idx, verbose
    )
    timing_info["sample_filtering_time"] = time.perf_counter() - start_filtering

    if df_scalars_filtered is None:
        return None, pd.DataFrame(), original_batch_size, timing_info

    # Update df_scalars to the filtered version
    df_scalars = df_scalars_filtered

    # 4. Calculate cluster metrics if needed
    series_to_bin: pd.Series | None = None

    if "cluster" in by:
        start_cluster_load = time.perf_counter()
        num_samples = len(df_scalars)
        inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
        inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)
        cluster_metrics = np.full(num_samples, np.nan, dtype=float)
        outside_value = np.nan
        outside_values = {}

        if use_sliding_window_format:
            inside_cluster_size_norms, inside_cluster_llr_norms, sw_timing = (
                _load_and_calculate_sliding_window_metrics(
                    batch_dir_path,
                    by,
                    norm_order,
                    sample_indices,
                    batch_start_idx,
                    original_batch_size,
                    priors,
                    eval_windows,
                    adj_matrix,
                    num_jobs,
                    num_batches,
                )
            )
            # Update timing info
            timing_info["cluster_file_load_time"] += sw_timing["cluster_file_load_time"]
            timing_info["cluster_calculation_time"] += sw_timing[
                "cluster_calculation_time"
            ]

        elif use_legacy_format:
            # Get local sample indices for this batch if filtering is needed
            local_sample_indices = None
            if sample_indices is not None and not df_scalars.empty:
                scalars_path = os.path.join(batch_dir_path, "scalars.feather")
                original_scalars = pd.read_feather(scalars_path)
                batch_end_idx = batch_start_idx + len(original_scalars)
                mask = (sample_indices >= batch_start_idx) & (
                    sample_indices < batch_end_idx
                )
                batch_sample_indices = sample_indices[mask]
                local_sample_indices = batch_sample_indices - batch_start_idx

            (
                inside_cluster_size_norms,
                inside_cluster_llr_norms,
                cluster_metrics,
                outside_value,
                legacy_timing,
            ) = _load_and_calculate_legacy_metrics(
                batch_dir_path, by, norm_order, local_sample_indices, num_samples
            )
            # Update timing info
            timing_info["cluster_file_load_time"] += legacy_timing[
                "cluster_file_load_time"
            ]
            timing_info["cluster_calculation_time"] += legacy_timing[
                "cluster_calculation_time"
            ]

        elif use_new_format:
            (
                inside_cluster_size_norms,
                inside_cluster_llr_norms,
                cluster_metrics,
                outside_values,
                new_timing,
            ) = _load_and_calculate_new_format_metrics(
                batch_dir_path,
                by,
                norm_order,
                priors,
                sample_indices,
                batch_start_idx,
                df_scalars,
            )
            # Update timing info
            timing_info["cluster_file_load_time"] += new_timing[
                "cluster_file_load_time"
            ]
            timing_info["cluster_calculation_time"] += new_timing[
                "cluster_calculation_time"
            ]

        timing_info["cluster_file_load_time"] += (
            time.perf_counter()
            - start_cluster_load
            - timing_info["cluster_calculation_time"]
        )

        # 5. Create aggregation series
        start_series_creation = time.perf_counter()
        series_to_bin = _create_aggregation_series(
            by,
            df_scalars,
            inside_cluster_size_norms,
            inside_cluster_llr_norms,
            cluster_metrics,
            use_sliding_window_format,
            use_new_format,
            outside_value,
            outside_values,
        )
        timing_info["series_creation_time"] = (
            time.perf_counter() - start_series_creation
        )

    else:
        # 'by' is not a cluster-dependent method, try to get column directly
        start_series_creation = time.perf_counter()
        if by in df_scalars.columns:
            series_to_bin = df_scalars[by].copy()
        else:
            raise ValueError(
                f"Column '{by}' not found in scalars.feather for {batch_dir_path}."
            )
        timing_info["series_creation_time"] = (
            time.perf_counter() - start_series_creation
        )

    return series_to_bin, df_scalars, original_batch_size, timing_info
