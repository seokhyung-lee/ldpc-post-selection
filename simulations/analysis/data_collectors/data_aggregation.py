import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import time
from scipy import sparse
from tqdm import tqdm

from ...utils.simulation_utils import get_existing_shots
from ..numpy_utils import (
    _calculate_cluster_norms_from_flat_data_numba,
    calculate_cluster_metrics_from_csr,
    _calculate_histograms_bplsd_numba,
    _calculate_histograms_matching_numba,
)


def _get_values_for_binning_from_batch(
    batch_dir_path: str,
    by: str,
    norm_order: float | None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    batch_start_idx: int = 0,
    verbose: bool = False,
) -> Tuple[pd.Series | None, pd.DataFrame | None]:
    """
    Loads data from a single batch directory needed for a specific aggregation method ('by').

    This function loads 'scalars.feather' and, if required by the 'by' method,
    also loads cluster data. It supports both legacy format (cluster_sizes.npy, cluster_llrs.npy, offsets.npy)
    and new format (clusters.npz with on-the-fly calculation).

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

    Returns
    -------
    series_to_bin : pd.Series or None
        The Series containing values to be binned. None if essential files are missing
        or data cannot be computed.
    df_scalars : pd.DataFrame or None
        The DataFrame loaded from 'scalars.feather'. None if 'scalars.feather' is missing.
    """
    scalars_path = os.path.join(batch_dir_path, "scalars.feather")
    cluster_sizes_path = os.path.join(batch_dir_path, "cluster_sizes.npy")
    cluster_llrs_path = os.path.join(batch_dir_path, "cluster_llrs.npy")
    offsets_path = os.path.join(batch_dir_path, "offsets.npy")
    clusters_path = os.path.join(batch_dir_path, "clusters.npz")

    # Check for scalars.feather first, as it's always needed.
    if not os.path.isfile(scalars_path):
        if verbose:
            print(
                f"  Error: scalars.feather not found in {batch_dir_path}. Cannot process this batch."
            )
        # Raise FileNotFoundError directly as per new requirement
        raise FileNotFoundError(
            f"scalars.feather not found in {batch_dir_path}. Cannot process this batch."
        )

    # Check for cluster data availability
    use_legacy_format = False
    use_new_format = False

    if "cluster" in by:
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

    df_scalars = pd.read_feather(scalars_path)

    if df_scalars.empty:
        if verbose:
            print(f"  Warning: scalars.feather in {batch_dir_path} is empty.")
        # Still return the empty df_scalars as it exists, the caller can decide to skip.
        # The series_to_bin will likely be empty or None.

    # Filter samples based on sample_indices if provided
    if sample_indices is not None and not df_scalars.empty:
        batch_end_idx = batch_start_idx + len(df_scalars)

        # Find which global indices fall within this batch's range
        mask = (sample_indices >= batch_start_idx) & (sample_indices < batch_end_idx)
        batch_sample_indices = sample_indices[mask]

        if len(batch_sample_indices) == 0:
            # No samples from this batch are in the requested indices
            if verbose:
                print(
                    f"  No requested samples found in batch {batch_dir_path} (range: {batch_start_idx}-{batch_end_idx-1})"
                )
            return None, pd.DataFrame()

        # Convert global indices to local batch indices
        local_indices = batch_sample_indices - batch_start_idx

        # Filter the dataframe to only include requested samples
        df_scalars = df_scalars.iloc[local_indices].reset_index(drop=True)

        if verbose:
            print(
                f"  Filtered to {len(df_scalars)} samples from batch {batch_dir_path}"
            )

    series_to_bin: pd.Series | None = None

    if "cluster" in by:
        num_samples = len(df_scalars)
        inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
        inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)
        cluster_metrics = np.full(num_samples, np.nan, dtype=float)

        if use_legacy_format:
            # Legacy format: load flat arrays and offsets
            offsets = np.load(offsets_path, allow_pickle=False)[:-1]

            # Get local sample indices for this batch if filtering is needed
            local_sample_indices = None
            if sample_indices is not None and not df_scalars.empty:
                batch_end_idx = batch_start_idx + len(pd.read_feather(scalars_path))
                mask = (sample_indices >= batch_start_idx) & (
                    sample_indices < batch_end_idx
                )
                batch_sample_indices = sample_indices[mask]
                local_sample_indices = batch_sample_indices - batch_start_idx

            if by in ["cluster_size_norm", "cluster_size_norm_gap"]:
                cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
                inside_cluster_size_norms, outside_value = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_sizes_flat,
                        offsets=offsets,
                        norm_order=norm_order,
                        sample_indices=local_sample_indices,
                    )
                )
            elif by in ["cluster_llr_norm", "cluster_llr_norm_gap"]:
                cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
                inside_cluster_llr_norms, outside_value = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_llrs_flat,
                        offsets=offsets,
                        norm_order=norm_order,
                        sample_indices=local_sample_indices,
                    )
                )
            elif by in ["cluster_llr_residual_sum", "cluster_llr_residual_sum_gap"]:
                # For residual sum methods, calculate cluster_llr_norm with order=1
                cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
                inside_cluster_llr_norms, outside_value = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_llrs_flat,
                        offsets=offsets,
                        norm_order=1.0,  # Always use L1 norm for residual sum
                        sample_indices=local_sample_indices,
                    )
                )
            elif by == "average_cluster_size":
                # Calculate (2-norm)^2 / (1-norm) for cluster sizes
                cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
                # Calculate 2-norm
                inside_cluster_2norms, _ = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_sizes_flat,
                        offsets=offsets,
                        norm_order=2.0,
                        sample_indices=local_sample_indices,
                    )
                )
                # Calculate 1-norm
                inside_cluster_1norms, _ = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_sizes_flat,
                        offsets=offsets,
                        norm_order=1.0,
                        sample_indices=local_sample_indices,
                    )
                )
                # Calculate average: (2-norm)^2 / (1-norm)
                cluster_metrics = np.full(num_samples, np.nan, dtype=float)
                valid_mask = inside_cluster_1norms > 0
                cluster_metrics[valid_mask] = (
                    inside_cluster_2norms[valid_mask] ** 2
                ) / inside_cluster_1norms[valid_mask]
            elif by == "average_cluster_llr":
                # Calculate (2-norm)^2 / (1-norm) for cluster LLRs
                cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
                # Calculate 2-norm
                inside_cluster_2norms, _ = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_llrs_flat,
                        offsets=offsets,
                        norm_order=2.0,
                        sample_indices=local_sample_indices,
                    )
                )
                # Calculate 1-norm
                inside_cluster_1norms, _ = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_llrs_flat,
                        offsets=offsets,
                        norm_order=1.0,
                        sample_indices=local_sample_indices,
                    )
                )
                # Calculate average: (2-norm)^2 / (1-norm)
                cluster_metrics = np.full(num_samples, np.nan, dtype=float)
                valid_mask = inside_cluster_1norms > 0
                cluster_metrics[valid_mask] = (
                    inside_cluster_2norms[valid_mask] ** 2
                ) / inside_cluster_1norms[valid_mask]
            else:
                raise ValueError(
                    f"Unsupported method ({by}) for legacy data structure."
                )

        elif use_new_format:
            # New format: load clusters and calculate directly from CSR format
            clusters_csr = sparse.load_npz(clusters_path)

            # If samples are filtered, we need to filter the CSR matrix rows
            if sample_indices is not None and not df_scalars.empty:
                batch_end_idx = batch_start_idx + len(pd.read_feather(scalars_path))
                mask = (sample_indices >= batch_start_idx) & (
                    sample_indices < batch_end_idx
                )
                batch_sample_indices = sample_indices[mask]
                local_indices = batch_sample_indices - batch_start_idx

                # Filter the CSR matrix to only include selected samples
                clusters_csr = clusters_csr[local_indices, :]

            # Calculate norms directly from CSR format using Numba
            if by in ["cluster_size_norm", "cluster_size_norm_gap"]:
                inside_cluster_size_norms, outside_size_values = (
                    calculate_cluster_metrics_from_csr(
                        clusters=clusters_csr,
                        method="norm",
                        priors=priors,  # Zeros for size calculations
                        norm_order=norm_order,
                    )
                )
            elif by in ["cluster_llr_norm", "cluster_llr_norm_gap"]:
                inside_cluster_llr_norms, outside_llr_values = (
                    calculate_cluster_metrics_from_csr(
                        clusters=clusters_csr,
                        method="llr_norm",
                        priors=priors,
                        norm_order=norm_order,
                    )
                )
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
            else:
                cluster_metrics = calculate_cluster_metrics_from_csr(
                    clusters=clusters_csr,
                    method=by,
                    priors=priors,
                )

        if by == "cluster_size_norm":
            series_to_bin = pd.Series(inside_cluster_size_norms, index=df_scalars.index)
        elif by == "cluster_llr_norm":
            series_to_bin = pd.Series(inside_cluster_llr_norms, index=df_scalars.index)
        elif by == "cluster_size_norm_gap":
            if use_new_format:
                series_to_bin = pd.Series(
                    outside_size_values - inside_cluster_size_norms,
                    index=df_scalars.index,
                )
            else:
                series_to_bin = pd.Series(
                    outside_value - inside_cluster_size_norms, index=df_scalars.index
                )
        elif by == "cluster_llr_norm_gap":
            if use_new_format:
                series_to_bin = pd.Series(
                    outside_llr_values - inside_cluster_llr_norms,
                    index=df_scalars.index,
                )
            else:
                series_to_bin = pd.Series(
                    outside_value - inside_cluster_llr_norms, index=df_scalars.index
                )
        elif by == "cluster_llr_residual_sum":
            # cluster_llr_norm (order=1) - pred_llr
            if "pred_llr" not in df_scalars.columns:
                raise ValueError(
                    f"'pred_llr' column not found in scalars.feather for {batch_dir_path}. Required for cluster_llr_residual_sum."
                )
            pred_llr_values = df_scalars["pred_llr"].values
            cluster_llr_residual_sum_values = inside_cluster_llr_norms - pred_llr_values
            series_to_bin = pd.Series(
                cluster_llr_residual_sum_values, index=df_scalars.index
            )
        elif by == "cluster_llr_residual_sum_gap":
            # outside_llr_value - cluster_llr_residual_sum
            if "pred_llr" not in df_scalars.columns:
                raise ValueError(
                    f"'pred_llr' column not found in scalars.feather for {batch_dir_path}. Required for cluster_llr_residual_sum_gap."
                )
            pred_llr_values = df_scalars["pred_llr"].values
            cluster_llr_residual_sum_values = inside_cluster_llr_norms - pred_llr_values
            if use_new_format:
                series_to_bin = pd.Series(
                    outside_llr_values - cluster_llr_residual_sum_values,
                    index=df_scalars.index,
                )
            else:
                series_to_bin = pd.Series(
                    outside_value - cluster_llr_residual_sum_values,
                    index=df_scalars.index,
                )
        elif by in ["average_cluster_size", "average_cluster_llr"]:
            series_to_bin = pd.Series(cluster_metrics, index=df_scalars.index)
        else:
            series_to_bin = pd.Series(cluster_metrics, index=df_scalars.index)
    else:  # 'by' is not an npy_dependent_method, try to get column directly
        if by in df_scalars.columns:
            series_to_bin = df_scalars[by].copy()
        else:
            raise ValueError(
                f"  Error: Column '{by}' not found in scalars.feather for {batch_dir_path}."
            )

    return series_to_bin, df_scalars


def _create_bin_edges(
    actual_min_val: float,
    actual_max_val: float,
    num_hist_bins: int,
    by: str,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Create histogram bin edges.

    Parameters
    ----------
    actual_min_val : float
        Minimum value for binning.
    actual_max_val : float
        Maximum value for binning.
    num_hist_bins : int
        Number of histogram bins.
    by : str
        Aggregation method name (for error messages).
    verbose : bool, optional
        Whether to print warnings.

    Returns
    -------
    bin_edges : np.ndarray
        Array of bin edges.
    num_hist_bins : int
        Adjusted number of bins (may be modified if range is too small).
    """
    if not isinstance(num_hist_bins, int) or num_hist_bins < 1:
        raise ValueError("num_hist_bins must be a positive integer.")

    if actual_max_val < actual_min_val:
        if verbose:
            print(
                f"Warning: Final max_value ({actual_max_val}) < min_value ({actual_min_val}). Setting max_value = min_value."
            )
        actual_max_val = actual_min_val

    if actual_min_val == actual_max_val and num_hist_bins > 1:
        if verbose:
            print(
                f"Warning: Cannot create {num_hist_bins} distinct bins for {by} as min_value ({actual_min_val}) == max_value ({actual_max_val}). Adjusting to 1 bin."
            )
        num_hist_bins = 1

    bin_edges = np.linspace(actual_min_val, actual_max_val, num_hist_bins + 1)

    # Validate bin edges
    if not np.all(np.diff(bin_edges) >= 0) and len(bin_edges) > 1:
        if num_hist_bins == 1:
            bin_edges = np.array([actual_min_val, actual_max_val])
            if bin_edges[0] > bin_edges[1]:
                bin_edges[1] = bin_edges[0]
        else:
            raise ValueError(
                f"Could not create monotonic bins for num_hist_bins={num_hist_bins} with range [{actual_min_val}, {actual_max_val}]. Edges: {bin_edges}"
            )

    return bin_edges, num_hist_bins


def calculate_df_agg_for_combination(
    data_dir: str,
    num_hist_bins: int = 1000,
    min_value_override: float | None = None,
    max_value_override: float | None = None,
    ascending_confidence: bool = True,
    by: str = "pred_llr",
    norm_order: float | None = None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Calculate the post-selection DataFrame (df_agg) for batch directories in a single parameter combination directory.
    Reads data from batch directories, each containing 'scalars.feather',
    'cluster_sizes.npy', 'cluster_llrs.npy', and 'offsets.npy'.
    Uses Numba JIT for histogram calculation.
    The range for the aggregation value histogram can be auto-detected or user-specified.

    Parameters
    ----------
    data_dir : str
        Directory path containing batch subdirectories for a single parameter combination.
    num_hist_bins : int, optional
        Number of bins to use for the histogram. Defaults to 1000.
    min_value_override : float, optional
        User-specified minimum value for the histogram range.
        If None, it's auto-detected.
    max_value_override : float, optional
        User-specified maximum value for the histogram range.
        If None, it's auto-detected.
    ascending_confidence : bool, optional
        Indicates the relationship between the aggregated value and decoding confidence.
        If True (default), a higher value implies higher confidence. The reported
        value in `df_agg` for a bin will be its lower edge.
        If False, a higher value implies lower confidence. The reported
        value in `df_agg` for a bin will be its upper edge.
    by : str, optional
        Column or method to aggregate by. Defaults to "pred_llr".
        Supported values:
        - "pred_llr": Reads from the 'pred_llr' column in 'scalars.feather'.
        - "detector_density": Reads from the 'detector_density' column in 'scalars.feather'.
        - "cluster_size_norm": Calculates norm of "inside" cluster sizes per sample.
                               Requires `norm_order`.
        - "cluster_llr_norm": Calculates norm of "inside" cluster LLRs per sample.
                              Requires `norm_order`.
        - "cluster_size_norm_gap": outside_cluster_size - norm_of_inside_cluster_sizes.
                                   Requires `norm_order`.
        - "cluster_llr_norm_gap": outside_cluster_llr - norm_of_inside_cluster_llrs.
                                  Requires `norm_order`.
        - "cluster_llr_residual_sum": cluster_llr_norm (order=1) - pred_llr.
                                      Requires `priors`.
        - "cluster_llr_residual_sum_gap": outside_cluster_llr - cluster_llr_residual_sum.
                                          Requires `priors`.
        - "cluster_inv_entropy": Sum of entropies for all inside clusters per sample.
                            Each bit's inv_entropy is calculated as -p * log(p) - (1-p) * log(1-p).
                            Requires `priors`.
        - "average_cluster_size": Average cluster size calculated as (2-norm)^2 / (1-norm) for cluster sizes.
        - "average_cluster_llr": Average cluster LLR calculated as (2-norm)^2 / (1-norm) for cluster LLRs.
                                 Requires `priors`.
    norm_order : float, optional
        The order for L_p norm calculation when `by` is one of the norm-based methods.
        Must be a positive float (can be np.inf).
        Required if `by` is one of the norm or norm-gap methods.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit, required for cluster_llr calculations when using new format.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include in the aggregation. If None, all samples are included.
    verbose : bool, optional
        Whether to print progress and benchmarking information. Defaults to False.

    Returns
    -------
    df_agg : pd.DataFrame
        DataFrame with columns [<by_column_name>, 'count', 'num_fails', 'num_converged', 'num_converged_fails'].
        Empty if processing fails or no data is found.
    total_rows_processed : int
        Total number of simulation samples processed (typically from scalars.feather).
    """
    # Find batch directories directly in data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Get all batch directories (assuming they start with "batch_")
    batch_dir_paths = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith("batch_"):
            batch_dir_paths.append(item_path)

    if not batch_dir_paths:
        raise FileNotFoundError(f"No batch directories found in {data_dir}")

    batch_dir_paths.sort()  # For consistent ordering

    if verbose:
        print(f"Found {len(batch_dir_paths)} batch directories in {data_dir}")

    # Determine value range for binning
    actual_min_val, actual_max_val = _determine_value_range_single(
        batch_dir_paths,
        by,
        norm_order,
        min_value_override,
        max_value_override,
        priors,
        sample_indices,
        verbose,
    )

    # Create bin edges
    bin_edges, adjusted_num_hist_bins = _create_bin_edges(
        actual_min_val, actual_max_val, num_hist_bins, by, verbose
    )

    # Process histograms from all batches
    (
        total_counts_hist,
        fail_counts_hist,
        converge_counts_hist,
        fail_converge_counts_hist,
        total_rows_processed,
        total_samples_considered,
        total_read_time,
        total_calc_value_time,
        total_hist_time,
    ) = _process_histograms_from_batches_single(
        batch_dir_paths,
        by,
        norm_order,
        bin_edges,
        adjusted_num_hist_bins,
        min_value_override,
        max_value_override,
        priors,
        sample_indices,
        verbose,
    )

    # Print benchmarking results if verbose
    if verbose:
        print("--- Benchmarking Results ---")
        print(
            f"Total samples considered from scalars.feather: {total_samples_considered}"
        )
        print(f"Total valid entries binned: {total_rows_processed}")
        print(
            f"Total time reading & initial processing per batch: {total_read_time:.4f} seconds"
        )
        print(
            f"Total time for downstream calculations on series (e.g., dropna): {total_calc_value_time:.4f} seconds"
        )
        print(f"Total time calculating histograms: {total_hist_time:.4f} seconds")
        print("----------------------------")

    # Check if any data was processed
    if total_rows_processed == 0:
        print(
            f"Warning: Processed 0 valid entries to bin for aggregation method {by}. Output df_agg will be empty."
        )

    # Determine if we have BPLSD columns based on whether any converge data was found
    has_bplsd_data = np.any(converge_counts_hist > 0) or np.any(
        fail_converge_counts_hist > 0
    )

    # Build and return result DataFrame
    df_agg = _build_result_dataframe_single(
        total_counts_hist,
        fail_counts_hist,
        converge_counts_hist,
        fail_converge_counts_hist,
        bin_edges,
        adjusted_num_hist_bins,
        by,
        ascending_confidence,
        has_bplsd_data,
    )

    if verbose:
        print(
            f"  -> Generated df_agg with {len(df_agg)} rows from {total_rows_processed} total valid binned entries ({total_samples_considered} samples initially considered), using method '{by}'."
        )

    return df_agg, total_rows_processed


def _determine_value_range_single(
    batch_dir_paths: List[str],
    by: str,
    norm_order: float | None,
    min_value_override: float | None,
    max_value_override: float | None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Determine the value range for histogram binning for a single parameter combination.

    Parameters
    ----------
    batch_dir_paths : List[str]
        List of batch directory paths to process.
    by : str
        Column or method to aggregate by.
    norm_order : float, optional
        The order for L_p norm calculation.
    min_value_override : float, optional
        User-specified minimum value override.
    max_value_override : float, optional
        User-specified maximum value override.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include. If None, all samples are included.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    actual_min_val : float
        The minimum value for the histogram range.
    actual_max_val : float
        The maximum value for the histogram range.
    """
    if min_value_override is not None and max_value_override is not None:
        if min_value_override >= max_value_override:
            raise ValueError("min_value_override must be less than max_value_override.")
        if verbose:
            print(
                f"  Using user-specified value range for {by}: [{min_value_override}, {max_value_override}]"
            )
        return min_value_override, max_value_override

    if verbose:
        print(
            f"  Auto-detecting value range for {by} (first pass over batch directories)..."
        )

    current_min_val = np.inf
    current_max_val = -np.inf
    found_any_valid_value_for_range = False

    desc_text = f"Range detection for {by}"
    current_batch_start_idx = 0

    for batch_dir_pass1 in tqdm(batch_dir_paths, desc=desc_text):
        temp_series_to_bin, _ = _get_values_for_binning_from_batch(
            batch_dir_path=batch_dir_pass1,
            by=by,
            norm_order=norm_order,
            priors=priors,
            sample_indices=sample_indices,
            batch_start_idx=current_batch_start_idx,
            verbose=verbose > 1,
        )

        # Update batch start index for next iteration
        # Read the original scalars file to get the actual batch size
        scalars_path = os.path.join(batch_dir_pass1, "scalars.feather")
        if os.path.isfile(scalars_path):
            batch_df = pd.read_feather(scalars_path)
            current_batch_start_idx += len(batch_df)

        if temp_series_to_bin is not None and not temp_series_to_bin.empty:
            temp_series_to_bin_cleaned = temp_series_to_bin.dropna()
            if not temp_series_to_bin_cleaned.empty:
                found_any_valid_value_for_range = True
                current_min_val = min(current_min_val, temp_series_to_bin_cleaned.min())
                current_max_val = max(current_max_val, temp_series_to_bin_cleaned.max())
            del temp_series_to_bin_cleaned
        if temp_series_to_bin is not None:
            del temp_series_to_bin

    if not found_any_valid_value_for_range:
        raise ValueError(f"No valid data found for {by} during range detection.")

    actual_min_val = current_min_val
    actual_max_val = current_max_val

    # Handle edge case where min ≈ max
    if np.isclose(actual_min_val, actual_max_val):
        adjustment = max(1.0, abs(actual_min_val * 0.1)) if actual_min_val != 0 else 1.0
        actual_max_val = actual_min_val + adjustment
        actual_min_val = actual_min_val - adjustment
        if verbose:
            print(
                f"  Detected min_value ~ max_value. Adjusted range for {by}: [{actual_min_val}, {actual_max_val}]"
            )
    elif actual_max_val < actual_min_val:
        actual_max_val = actual_min_val

    if verbose:
        print(
            f"  Auto-detected value range for {by}: [{actual_min_val}, {actual_max_val}]"
        )

    return actual_min_val, actual_max_val


def _process_histograms_from_batches_single(
    batch_dir_paths: List[str],
    by: str,
    norm_order: float | None,
    bin_edges: np.ndarray,
    num_hist_bins: int,
    min_value_override: float | None,
    max_value_override: float | None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    verbose: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, float, float
]:
    """
    Process histogram data from all batch directories for a single parameter combination.

    Parameters
    ----------
    batch_dir_paths : List[str]
        List of batch directory paths.
    by : str
        Aggregation method.
    norm_order : float, optional
        Norm order for norm-based methods.
    bin_edges : np.ndarray
        Histogram bin edges.
    num_hist_bins : int
        Number of histogram bins.
    min_value_override : float, optional
        Minimum value override.
    max_value_override : float, optional
        Maximum value override.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include. If None, all samples are included.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    total_counts_hist : np.ndarray
        Total counts histogram.
    fail_counts_hist : np.ndarray
        Failure counts histogram.
    converge_counts_hist : np.ndarray
        Convergence counts histogram.
    fail_converge_counts_hist : np.ndarray
        Failure convergence counts histogram.
    total_rows_processed : int
        Total number of rows processed.
    total_samples_considered : int
        Total number of samples considered.
    total_read_time : float
        Total time spent reading data.
    total_calc_value_time : float
        Total time spent calculating values.
    total_hist_time : float
        Total time spent creating histograms.
    """
    total_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    fail_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    converge_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    fail_converge_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    total_rows_processed = 0
    total_samples_considered = 0

    total_read_time = 0.0
    total_calc_value_time = 0.0
    total_hist_time = 0.0

    if verbose:
        print(
            f"Processing {len(batch_dir_paths)} batch directories iteratively (Numba histograms)..."
        )

    desc_text = f"Aggregation for {by}"
    current_batch_start_idx = 0

    for batch_dir in tqdm(batch_dir_paths, desc=desc_text):
        start_time_read = time.perf_counter()

        series_to_bin, df_scalars = _get_values_for_binning_from_batch(
            batch_dir_path=batch_dir,
            by=by,
            norm_order=norm_order,
            priors=priors,
            sample_indices=sample_indices,
            batch_start_idx=current_batch_start_idx,
            verbose=verbose > 1,
        )
        read_and_initial_calc_time = time.perf_counter() - start_time_read
        total_read_time += read_and_initial_calc_time

        if series_to_bin is None or df_scalars is None or df_scalars.empty:
            if verbose:
                print(
                    f"  Skipping batch {os.path.basename(batch_dir)} due to issues from _get_values_for_binning (series or scalars missing/empty)."
                )
            continue

        total_samples_considered += len(df_scalars)

        # Check for required columns
        required_scalar_cols = ["fail", "converge", "fail_bp"]
        has_bplsd_cols = all(col in df_scalars.columns for col in required_scalar_cols)
        has_fail_col = "fail" in df_scalars.columns

        if not has_fail_col:
            if verbose:
                print(
                    f"  Skipping {os.path.basename(batch_dir)} due to missing 'fail' column in scalars.feather."
                )
            continue

        start_time_calc_downstream = time.perf_counter()
        series_to_bin_cleaned = series_to_bin.dropna()
        calc_value_time_batch_downstream = (
            time.perf_counter() - start_time_calc_downstream
        )
        total_calc_value_time += calc_value_time_batch_downstream

        if series_to_bin_cleaned.empty:
            if verbose:
                print(
                    f"  No valid (non-NaN) data to bin for 'by={by}' in {os.path.basename(batch_dir)} after dropna. Skipping."
                )
            continue

        total_rows_processed += len(series_to_bin_cleaned)

        # Validate value range if overrides are specified
        if min_value_override is not None and max_value_override is not None:
            values_np_check = series_to_bin_cleaned.to_numpy()
            if np.any(values_np_check < min_value_override) or np.any(
                values_np_check > max_value_override
            ):
                raise ValueError(
                    f"Data found outside user-specified value range "
                    f"[{min_value_override}, {max_value_override}]. "
                    f"Aggregation method: {by}, Batch: {os.path.basename(batch_dir)}"
                )

        start_time_hist = time.perf_counter()
        values_np = series_to_bin_cleaned.to_numpy()
        fail_mask = df_scalars.loc[series_to_bin_cleaned.index, "fail"].to_numpy(
            dtype=bool
        )

        if has_bplsd_cols:
            converge_mask = df_scalars.loc[
                series_to_bin_cleaned.index, "converge"
            ].to_numpy(dtype=bool)
            fail_bp_mask = df_scalars.loc[
                series_to_bin_cleaned.index, "fail_bp"
            ].to_numpy(dtype=bool)

            (
                total_counts_hist,
                fail_counts_hist,
                converge_counts_hist,
                fail_converge_counts_hist,
            ) = _calculate_histograms_bplsd_numba(
                values_np,
                fail_mask,
                bin_edges,
                total_counts_hist,
                fail_counts_hist,
                converge_mask,
                converge_counts_hist,
                fail_converge_counts_hist,
                fail_bp_mask,
            )
        else:  # Only 'fail' column is guaranteed to be present
            (
                total_counts_hist,
                fail_counts_hist,
            ) = _calculate_histograms_matching_numba(
                values_np,
                fail_mask,
                bin_edges,
                total_counts_hist,
                fail_counts_hist,
            )

        hist_time_batch = time.perf_counter() - start_time_hist
        total_hist_time += hist_time_batch

        # Update batch start index for next iteration
        # Read the original scalars file to get the actual batch size
        scalars_path = os.path.join(batch_dir, "scalars.feather")
        if os.path.isfile(scalars_path):
            batch_df = pd.read_feather(scalars_path)
            current_batch_start_idx += len(batch_df)

    return (
        total_counts_hist,
        fail_counts_hist,
        converge_counts_hist,
        fail_converge_counts_hist,
        total_rows_processed,
        total_samples_considered,
        total_read_time,
        total_calc_value_time,
        total_hist_time,
    )


def _build_result_dataframe_single(
    total_counts_hist: np.ndarray,
    fail_counts_hist: np.ndarray,
    converge_counts_hist: np.ndarray,
    fail_converge_counts_hist: np.ndarray,
    bin_edges: np.ndarray,
    num_hist_bins: int,
    by: str,
    ascending_confidence: bool,
    has_bplsd_data: bool,
) -> pd.DataFrame:
    """
    Build the result DataFrame from histogram data for a single parameter combination.

    Parameters
    ----------
    total_counts_hist : np.ndarray
        Total counts histogram.
    fail_counts_hist : np.ndarray
        Failure counts histogram.
    converge_counts_hist : np.ndarray
        Convergence counts histogram.
    fail_converge_counts_hist : np.ndarray
        Failure convergence counts histogram.
    bin_edges : np.ndarray
        Histogram bin edges.
    num_hist_bins : int
        Number of histogram bins.
    by : str
        Aggregation method name.
    ascending_confidence : bool
        Whether higher values mean higher confidence.
    has_bplsd_data : bool
        Whether BPLSD convergence data is available.

    Returns
    -------
    df_agg : pd.DataFrame
        Aggregated result DataFrame.
    """
    binned_value_column_name = by

    if ascending_confidence:
        binned_values_for_df = bin_edges[:-1]
    else:
        binned_values_for_df = bin_edges[1:]

    if len(binned_values_for_df) != num_hist_bins:
        if num_hist_bins == 1 and len(total_counts_hist) == 1:
            if ascending_confidence:
                binned_values_for_df = np.array([bin_edges[0]])
            else:
                binned_values_for_df = np.array([bin_edges[len(bin_edges) - 1]])
        else:
            raise ValueError(
                f"Mismatch between binned_values_for_df (len {len(binned_values_for_df)}) and num_hist_bins ({num_hist_bins}). Cannot construct df_agg."
            )

    df_agg_data = {
        binned_value_column_name: binned_values_for_df,
        "count": total_counts_hist,
        "num_fails": fail_counts_hist,
    }

    # Only add convergence columns if they have data
    if has_bplsd_data:
        df_agg_data["num_converged"] = converge_counts_hist
        df_agg_data["num_converged_fails"] = fail_converge_counts_hist

    df_agg = pd.DataFrame(df_agg_data)

    # Filter out empty bins and sort
    df_agg = df_agg[df_agg["count"] > 0].copy()
    if not df_agg.empty:
        df_agg.sort_values(binned_value_column_name, ascending=True, inplace=True)
        df_agg.reset_index(drop=True, inplace=True)
        # Set the by column as index
        df_agg.set_index(binned_value_column_name, inplace=True)

    return df_agg


def extract_sample_metric_values(
    data_dir: str,
    *,
    by: str,
    norm_order: float | None = None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
    verbose: bool = False,
) -> np.ndarray:
    """
    Extract metric values for all samples from batch directories without binning.

    Parameters
    ----------
    data_dir : str
        Directory path containing batch subdirectories for a single parameter combination.
    by : str
        Column or method to extract values from.
        Supported values:
        - "cluster_size_norm": Calculates norm of "inside" cluster sizes per sample.
                               Requires `norm_order`.
        - "cluster_llr_norm": Calculates norm of "inside" cluster LLRs per sample.
                              Requires `norm_order`.
        - "cluster_size_norm_gap": outside_cluster_size - norm_of_inside_cluster_sizes.
                                   Requires `norm_order`.
        - "cluster_llr_norm_gap": outside_cluster_llr - norm_of_inside_cluster_llrs.
                                  Requires `norm_order`.
        - "cluster_llr_residual_sum": cluster_llr_norm (order=1) - pred_llr.
                                      Requires `priors`.
        - "cluster_llr_residual_sum_gap": outside_cluster_llr - cluster_llr_residual_sum.
                                          Requires `priors`.
        - "cluster_inv_entropy": Sum of entropies for all inside clusters per sample.
                            Each bit's inv_entropy is calculated as -p * log(p) - (1-p) * log(1-p).
                            Requires `priors`.
        - "average_cluster_size": Average cluster size calculated as (2-norm)^2 / (1-norm) for cluster sizes.
        - "average_cluster_llr": Average cluster LLR calculated as (2-norm)^2 / (1-norm) for cluster LLRs.
                                 Requires `priors`.
        - All the other values are read from the 'scalars.feather' file.
    norm_order : float, optional
        The order for L_p norm calculation when `by` is one of the norm or norm-gap methods.
        Must be a positive float. Required if `by` is one of the norm-based methods.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit, required for cluster_llr calculations when using new format.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include. If None, all samples are included.
        The indices should be global across all batches (e.g., if batch 0 has 1000 samples and batch 1 has 1000 samples,
        then sample index 1500 would refer to the 500th sample in batch 1).
    dtype : np.dtype, optional
        Data type for the output array. Defaults to np.float32 for memory efficiency.
        Common choices: np.float32, np.float64, np.int32, np.int64.
    verbose : bool, optional
        Whether to print progress and informational messages. Defaults to False.

    Returns
    -------
    values : np.ndarray
        1D numpy array containing the metric values for all samples with specified dtype.
        NaN values are removed from the output.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Error: Data directory not found: {data_dir}")

    # Find batch directories directly in data_dir
    batch_dir_paths = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith("batch_"):
            batch_dir_paths.append(item_path)

    if not batch_dir_paths:
        raise FileNotFoundError(f"No batch directories found in {data_dir}")

    batch_dir_paths.sort()  # For consistent ordering

    if verbose:
        print(f"Found {len(batch_dir_paths)} batch directories in {data_dir}")
        print(f"Extracting metric values for: {by}")
        if norm_order is not None:
            print(f"Norm order: {norm_order}")
        if sample_indices is not None:
            print(
                f"Using {len(sample_indices)} specific sample indices (range: {sample_indices.min()}-{sample_indices.max()})"
            )

    all_values = []
    current_batch_start_idx = 0

    desc_text = f"Extracting {by} values"

    for batch_dir in tqdm(batch_dir_paths, desc=desc_text):
        try:
            series_to_extract, df_scalars = _get_values_for_binning_from_batch(
                batch_dir_path=batch_dir,
                by=by,
                norm_order=norm_order,
                priors=priors,
                sample_indices=sample_indices,
                batch_start_idx=current_batch_start_idx,
                verbose=verbose > 1,
            )

            # Update batch start index for next iteration
            # Read the original scalars file to get the actual batch size
            scalars_path = os.path.join(batch_dir, "scalars.feather")
            if os.path.isfile(scalars_path):
                batch_df = pd.read_feather(scalars_path)
                current_batch_start_idx += len(batch_df)

            if series_to_extract is not None and not series_to_extract.empty:
                # Convert to numpy with specified dtype and append to list
                batch_values = series_to_extract.to_numpy().astype(dtype)
                all_values.append(batch_values)

                if verbose:
                    valid_count = np.sum(~np.isnan(batch_values))
                    print(
                        f"  Extracted {valid_count} valid values from {os.path.basename(batch_dir)} (dtype: {dtype})"
                    )
            else:
                if verbose:
                    print(f"  No values extracted from {os.path.basename(batch_dir)}")

        except Exception as e:
            raise ValueError(f"  Error processing {os.path.basename(batch_dir)}: {e}")

    if not all_values:
        if verbose:
            print("No values were extracted from any batch directory.")
        return np.array([], dtype=dtype)

    # Concatenate all values
    concatenated_values = np.concatenate(all_values)

    # Remove NaN values and ensure final dtype
    valid_values = concatenated_values[~np.isnan(concatenated_values)]

    if verbose:
        print(
            f"Extracted {len(valid_values)} valid values (removed {len(concatenated_values) - len(valid_values)} NaN values)"
        )
        if len(valid_values) > 0:
            print(f"Value range: [{valid_values.min():.6f}, {valid_values.max():.6f}]")
        print(f"Final array dtype: {valid_values.dtype}")

    return valid_values


def aggregate_data(
    data_dir: str,
    *,
    by: str,
    num_hist_bins: int = 1000,
    value_range: tuple[float, float] | None = None,
    ascending_confidence: bool = True,
    norm_order: float | None = None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    df_existing: pd.DataFrame | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Aggregate simulation data based on specified metrics from batch directories in a single parameter combination directory.

    Parameters
    ----------
    data_dir : str
        Directory path containing batch subdirectories for a single parameter combination.
    by : str
        Column or method to aggregate by.
        Supported values:
        - "cluster_size_norm": Calculates norm of "inside" cluster sizes per sample.
                               Requires `norm_order`.
        - "cluster_llr_norm": Calculates norm of "inside" cluster LLRs per sample.
                              Requires `norm_order`.
        - "cluster_size_norm_gap": outside_cluster_size - norm_of_inside_cluster_sizes.
                                   Requires `norm_order`.
        - "cluster_llr_norm_gap": outside_cluster_llr - norm_of_inside_cluster_llrs.
                                  Requires `norm_order`.
        - "cluster_llr_residual_sum": cluster_llr_norm (order=1) - pred_llr.
                                      Requires `priors`.
        - "cluster_llr_residual_sum_gap": outside_cluster_llr - cluster_llr_residual_sum.
                                          Requires `priors`.
        - "cluster_inv_entropy": Sum of entropies for all inside clusters per sample.
                            Each bit's inv_entropy is calculated as -p * log(p) - (1-p) * log(1-p).
                            Requires `priors`.
        - "average_cluster_size": Average cluster size calculated as (2-norm)^2 / (1-norm) for cluster sizes.
        - "average_cluster_llr": Average cluster LLR calculated as (2-norm)^2 / (1-norm) for cluster LLRs.
                                 Requires `priors`.
        - All the other values are read from the 'scalars.feather' file.
    num_hist_bins : int, optional
        Number of bins to use for the histogram. Defaults to 1000.
    value_range : tuple[float, float], optional
        User-specified minimum and maximum values for the histogram range ([min, max]).
        If None, it's auto-detected across all batch data for the chosen `by` method.
    ascending_confidence : bool, optional
        Indicates the relationship between the aggregated value and decoding confidence.
        Defaults to True.
    norm_order : float, optional
        The order for L_p norm calculation when `by` is one of the norm or norm-gap methods.
        Must be a positive float. Required if `by` is one of the norm-based methods.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit, required for cluster_llr calculations when using new format.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include in the aggregation. If None, all samples are included.
        The indices should be global across all batches (e.g., if batch 0 has 1000 samples and batch 1 has 1000 samples,
        then sample index 1500 would refer to the 500th sample in batch 1).
    df_existing : pd.DataFrame, optional
        Existing aggregation data with the same format as the return value of this function.
        If provided and the total shots from data files match the total shots in df_existing,
        the existing data will be returned directly without reprocessing.
    verbose : bool, optional
        Whether to print progress and informational messages. Defaults to False.

    Returns
    -------
    df_agg : pd.DataFrame
        DataFrame with index corresponding to the `by` column and columns containing
        post-selection statistics. Returns an empty DataFrame if no valid data is found or processed.
    reused_existing : bool
        True if existing data was reused, False if new calculations were performed.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Error: Data directory not found: {data_dir}")

    # Parse value_range
    min_val_override, max_val_override = None, None
    if value_range:
        if (
            len(value_range) == 2
            and isinstance(value_range[0], (int, float))
            and isinstance(value_range[1], (int, float))
        ):
            min_val_override, max_val_override = float(value_range[0]), float(
                value_range[1]
            )
            if min_val_override >= max_val_override:
                print(
                    "Warning: value_range min must be less than max. Ignoring provided value_range."
                )
                min_val_override, max_val_override = None, None
        else:
            print(
                "Warning: value_range must be a tuple of two numbers (min, max). Ignoring provided value_range."
            )

    # Check if we can reuse existing data
    # Note: If sample_indices is provided, we should not reuse existing data
    # as it likely contains different samples
    if df_existing is not None and not df_existing.empty and sample_indices is None:
        # Validate existing data format
        if by not in df_existing.index.names:
            if by in df_existing.columns:
                # Try to set the by column as index if it's currently a column
                df_existing_copy = df_existing.set_index(by)
            else:
                raise ValueError(
                    f"df_existing does not have '{by}' as index or column."
                )
        else:
            df_existing_copy = df_existing.copy()

        if "count" not in df_existing_copy.columns:
            raise ValueError("df_existing does not have a 'count' column.")

        try:
            existing_total_shots = df_existing_copy["count"].sum()
            file_total_shots, _ = get_existing_shots(data_dir)

            if existing_total_shots == file_total_shots:
                if verbose:
                    print(
                        f"Existing shots ({existing_total_shots}) match file shots ({file_total_shots}). Returning existing data."
                    )
                return df_existing_copy, True
            else:
                if verbose:
                    print(
                        f"Existing shots ({existing_total_shots}) != file shots ({file_total_shots}). Reprocessing."
                    )
        except (FileNotFoundError, ValueError) as e:
            if verbose:
                print(f"Could not check file shots: {e}. Reprocessing.")
    elif sample_indices is not None and verbose:
        print("Sample indices provided. Skipping existing data reuse and reprocessing.")

    # Process the data
    if verbose:
        print(f"Processing data directory: {data_dir}")
        print(f"Aggregation method: {by}")
        if norm_order is not None:
            print(f"Norm order: {norm_order}")
        if sample_indices is not None:
            print(
                f"Using {len(sample_indices)} specific sample indices (range: {sample_indices.min()}-{sample_indices.max()})"
            )

    df_agg, total_rows_processed = calculate_df_agg_for_combination(
        data_dir=data_dir,
        num_hist_bins=num_hist_bins,
        min_value_override=min_val_override,
        max_value_override=max_val_override,
        ascending_confidence=ascending_confidence,
        by=by,
        norm_order=norm_order,
        priors=priors,
        sample_indices=sample_indices,
        verbose=verbose,
    )

    if verbose:
        if df_agg.empty:
            print("No data was processed successfully.")
        else:
            print(
                f"Successfully aggregated {total_rows_processed} samples into {len(df_agg)} bins."
            )

    return df_agg, False
