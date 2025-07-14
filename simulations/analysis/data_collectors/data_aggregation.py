import os
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
import time
from scipy import sparse
from tqdm import tqdm
from joblib import Parallel, delayed

from ...utils.simulation_utils import get_existing_shots
from ..numpy_utils import (
    _calculate_cluster_norms_from_flat_data_numba,
    calculate_cluster_metrics_from_csr,
    _calculate_histograms_bplsd_numba,
    _calculate_histograms_matching_numba,
)


def _calculate_sliding_window_norm_fractions(
    cluster_data_list: List[List[np.ndarray]],
    norm_order: float,
    aggregation_type: str,
) -> np.ndarray:
    """
    Calculate norm fractions for sliding window cluster data.

    Parameters
    ----------
    cluster_data_list : list of list of numpy arrays
        List where each element is a shot's window data (list of arrays per window).
    norm_order : float
        Order for L_p norm calculation.
    aggregation_type : str
        Type of aggregation: "mean", "max", or "committed".

    Returns
    -------
    norm_fractions : 1D numpy array of float
        Norm fractions for each shot.
    """
    num_shots = len(cluster_data_list)
    norm_fractions = np.zeros(num_shots, dtype=float)

    for shot_idx, shot_windows in enumerate(cluster_data_list):
        if not shot_windows:  # Empty list
            norm_fractions[shot_idx] = 0.0
            continue

        window_norm_fracs = []

        if aggregation_type == "committed":
            # Keep only the last window for this aggregation type
            shot_windows = [shot_windows[-1]]

        for window_data in shot_windows:
            if len(window_data) == 0:
                window_norm_fracs.append(0.0)
                continue

            # Calculate norm fraction for this window
            # Assuming window_data[0] is outside region, window_data[1:] are inside clusters
            total_sum = np.sum(window_data)
            if total_sum == 0:
                window_norm_fracs.append(0.0)
                continue

            inside_values = window_data[1:] if len(window_data) > 1 else np.array([])

            if len(inside_values) == 0:
                window_norm_fracs.append(0.0)
            elif norm_order == np.inf:
                norm_frac = (
                    np.max(inside_values) / total_sum if len(inside_values) > 0 else 0.0
                )
            elif norm_order == 1:
                norm_frac = np.sum(inside_values) / total_sum
            elif norm_order == 2:
                norm_frac = np.sqrt(np.sum(inside_values**2)) / total_sum
            else:
                norm_frac = (
                    np.sum(inside_values**norm_order) ** (1 / norm_order) / total_sum
                )

            window_norm_fracs.append(float(norm_frac))

        # Aggregate across windows based on aggregation_type
        if not window_norm_fracs:
            norm_fractions[shot_idx] = 0.0
        elif aggregation_type == "mean":
            norm_fractions[shot_idx] = np.mean(window_norm_fracs)
        elif aggregation_type == "max":
            norm_fractions[shot_idx] = np.max(window_norm_fracs)
        elif aggregation_type == "committed":
            # Use the last window's norm fraction
            norm_fractions[shot_idx] = window_norm_fracs[-1]
        else:
            raise ValueError(f"Unknown aggregation_type: {aggregation_type}")

    return norm_fractions


def _get_values_for_binning_from_batch(
    batch_dir_path: str,
    by: str,
    norm_order: float | None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    batch_start_idx: int = 0,
    verbose: bool = False,
) -> Tuple[pd.Series | None, pd.DataFrame | None, int, dict]:
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
    original_batch_size : int
        The original size of the batch before any filtering.
    timing_info : dict
        Dictionary containing detailed timing information for different steps.
    """
    timing_info = {
        "file_check_time": 0.0,
        "scalars_load_time": 0.0,
        "sample_filtering_time": 0.0,
        "cluster_file_load_time": 0.0,
        "cluster_calculation_time": 0.0,
        "series_creation_time": 0.0,
    }

    start_time = time.perf_counter()

    scalars_path = os.path.join(batch_dir_path, "scalars.feather")
    cluster_sizes_path = os.path.join(batch_dir_path, "cluster_sizes.npy")
    cluster_llrs_path = os.path.join(batch_dir_path, "cluster_llrs.npy")
    offsets_path = os.path.join(batch_dir_path, "offsets.npy")
    clusters_path = os.path.join(batch_dir_path, "clusters.npz")

    # Sliding window format paths
    fails_path = os.path.join(batch_dir_path, "fails.npy")
    cluster_sizes_pkl_path = os.path.join(batch_dir_path, "cluster_sizes.pkl")
    cluster_llrs_pkl_path = os.path.join(batch_dir_path, "cluster_llrs.pkl")
    committed_cluster_sizes_pkl_path = os.path.join(
        batch_dir_path, "committed_cluster_sizes.pkl"
    )
    committed_cluster_llrs_pkl_path = os.path.join(
        batch_dir_path, "committed_cluster_llrs.pkl"
    )

    # Check for scalars.feather first, as it's always needed for regular format
    # For sliding window format, we'll use fails.npy instead
    use_sliding_window_format = False

    if not os.path.isfile(scalars_path):
        # Check if this is sliding window format
        if os.path.isfile(fails_path) and os.path.isfile(cluster_sizes_pkl_path):
            use_sliding_window_format = True
            if verbose:
                print(f"  Using sliding window format for {batch_dir_path}")
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
    use_legacy_format = False
    use_new_format = False

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

    timing_info["file_check_time"] = time.perf_counter() - start_time

    # Load data based on format
    start_scalars_load = time.perf_counter()

    if use_sliding_window_format:
        # Load fails.npy for sliding window format
        fails = np.load(fails_path)
        original_batch_size = len(fails)

        # Create minimal df_scalars with fail column
        df_scalars = pd.DataFrame({"fail": fails})

        if verbose:
            print(f"  Loaded {len(fails)} samples from sliding window format")
    else:
        # Load regular scalars.feather
        df_scalars = pd.read_feather(scalars_path)
        original_batch_size = len(
            df_scalars
        )  # Store original size before any filtering

        if df_scalars.empty:
            if verbose:
                print(f"  Warning: scalars.feather in {batch_dir_path} is empty.")
            # Still return the empty df_scalars as it exists, the caller can decide to skip.
            # The series_to_bin will likely be empty or None.

    timing_info["scalars_load_time"] = time.perf_counter() - start_scalars_load

    # Filter samples based on sample_indices if provided
    start_filtering = time.perf_counter()
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
            return None, pd.DataFrame(), original_batch_size, timing_info

        # Convert global indices to local batch indices
        local_indices = batch_sample_indices - batch_start_idx

        # Filter the dataframe to only include requested samples
        df_scalars = df_scalars.iloc[local_indices].reset_index(drop=True)

        if verbose:
            print(
                f"  Filtered to {len(df_scalars)} samples from batch {batch_dir_path}"
            )
    timing_info["sample_filtering_time"] = time.perf_counter() - start_filtering

    series_to_bin: pd.Series | None = None

    if "cluster" in by:
        start_cluster_load = time.perf_counter()
        num_samples = len(df_scalars)
        inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
        inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)
        cluster_metrics = np.full(num_samples, np.nan, dtype=float)

        if use_sliding_window_format:
            # Sliding window format: load pickled lists
            start_file_load = time.perf_counter()

            # Parse the aggregation type from the 'by' parameter
            # Expected format: {aggregation_type}_cluster_{size/llr}_norm_frac_{order}
            if "_cluster_" in by and "_norm_frac_" in by:
                parts = by.split("_cluster_")
                aggregation_type = parts[0]  # mean, max, or committed

                remaining = parts[1].split("_norm_frac_")
                value_type = remaining[0]  # size or llr

                # Load appropriate cluster data
                if value_type == "size":
                    if aggregation_type == "committed":
                        with open(committed_cluster_sizes_pkl_path, "rb") as f:
                            cluster_data_list = pickle.load(f)
                    else:
                        with open(cluster_sizes_pkl_path, "rb") as f:
                            cluster_data_list = pickle.load(f)
                elif value_type == "llr":
                    if aggregation_type == "committed":
                        with open(committed_cluster_llrs_pkl_path, "rb") as f:
                            cluster_data_list = pickle.load(f)
                    else:
                        with open(cluster_llrs_pkl_path, "rb") as f:
                            cluster_data_list = pickle.load(f)
                else:
                    raise ValueError(
                        f"Unknown value type in sliding window metric: {value_type}"
                    )

                cluster_files_load_time = time.perf_counter() - start_file_load
                timing_info["cluster_file_load_time"] = cluster_files_load_time

                # Filter samples if needed
                if sample_indices is not None:
                    filtered_cluster_data = []
                    batch_end_idx = batch_start_idx + original_batch_size
                    mask = (sample_indices >= batch_start_idx) & (
                        sample_indices < batch_end_idx
                    )
                    batch_sample_indices = sample_indices[mask]
                    local_indices = batch_sample_indices - batch_start_idx

                    for idx in local_indices:
                        if idx < len(cluster_data_list):
                            filtered_cluster_data.append(cluster_data_list[idx])

                    cluster_data_list = filtered_cluster_data

                # Calculate norm fractions
                start_calc = time.perf_counter()
                norm_fractions = _calculate_sliding_window_norm_fractions(
                    cluster_data_list=cluster_data_list,
                    norm_order=norm_order,
                    aggregation_type=aggregation_type,
                )
                timing_info["cluster_calculation_time"] = (
                    time.perf_counter() - start_calc
                )

                # Store results based on metric type
                if value_type == "size":
                    inside_cluster_size_norms = norm_fractions
                else:
                    inside_cluster_llr_norms = norm_fractions
            else:
                raise ValueError(f"Unsupported sliding window metric format: {by}")

        elif use_legacy_format:
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

            cluster_files_load_time = 0.0
            if by in ["cluster_size_norm", "cluster_size_norm_gap"]:
                start_file_load = time.perf_counter()
                cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
                cluster_files_load_time = time.perf_counter() - start_file_load

                start_calc = time.perf_counter()
                inside_cluster_size_norms, outside_value = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_sizes_flat,
                        offsets=offsets,
                        norm_order=norm_order,
                        sample_indices=local_sample_indices,
                    )
                )
                timing_info["cluster_calculation_time"] = (
                    time.perf_counter() - start_calc
                )
            elif by in ["cluster_llr_norm", "cluster_llr_norm_gap"]:
                start_file_load = time.perf_counter()
                cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
                cluster_files_load_time = time.perf_counter() - start_file_load

                start_calc = time.perf_counter()
                inside_cluster_llr_norms, outside_value = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_llrs_flat,
                        offsets=offsets,
                        norm_order=norm_order,
                        sample_indices=local_sample_indices,
                    )
                )
                timing_info["cluster_calculation_time"] = (
                    time.perf_counter() - start_calc
                )
            elif by in ["cluster_llr_residual_sum", "cluster_llr_residual_sum_gap"]:
                # For residual sum methods, calculate cluster_llr_norm with order=1
                start_file_load = time.perf_counter()
                cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
                cluster_files_load_time = time.perf_counter() - start_file_load

                start_calc = time.perf_counter()
                inside_cluster_llr_norms, outside_value = (
                    _calculate_cluster_norms_from_flat_data_numba(
                        flat_data=cluster_llrs_flat,
                        offsets=offsets,
                        norm_order=1.0,  # Always use L1 norm for residual sum
                        sample_indices=local_sample_indices,
                    )
                )
                timing_info["cluster_calculation_time"] = (
                    time.perf_counter() - start_calc
                )
            elif by == "average_cluster_size":
                # Calculate (2-norm)^2 / (1-norm) for cluster sizes
                start_file_load = time.perf_counter()
                cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
                cluster_files_load_time = time.perf_counter() - start_file_load

                start_calc = time.perf_counter()
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
                timing_info["cluster_calculation_time"] = (
                    time.perf_counter() - start_calc
                )
            elif by == "average_cluster_llr":
                # Calculate (2-norm)^2 / (1-norm) for cluster LLRs
                start_file_load = time.perf_counter()
                cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
                cluster_files_load_time = time.perf_counter() - start_file_load

                start_calc = time.perf_counter()
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
                timing_info["cluster_calculation_time"] = (
                    time.perf_counter() - start_calc
                )
            else:
                raise ValueError(
                    f"Unsupported method ({by}) for legacy data structure."
                )

            timing_info["cluster_file_load_time"] = cluster_files_load_time

        elif use_new_format:
            # New format: load clusters and calculate directly from CSR format
            start_file_load = time.perf_counter()
            clusters_csr = sparse.load_npz(clusters_path)
            timing_info["cluster_file_load_time"] = (
                time.perf_counter() - start_file_load
            )

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
            timing_info["cluster_calculation_time"] = time.perf_counter() - start_calc

        timing_info["cluster_file_load_time"] += (
            time.perf_counter()
            - start_cluster_load
            - timing_info["cluster_calculation_time"]
        )

        start_series_creation = time.perf_counter()

        # Handle sliding window metrics
        if use_sliding_window_format and "_cluster_" in by and "_norm_frac_" in by:
            # For sliding window format, we've already calculated the norm fractions
            parts = by.split("_cluster_")
            remaining = parts[1].split("_norm_frac_")
            value_type = remaining[0]  # size or llr

            if value_type == "size":
                series_to_bin = pd.Series(
                    inside_cluster_size_norms, index=df_scalars.index
                )
            else:  # llr
                series_to_bin = pd.Series(
                    inside_cluster_llr_norms, index=df_scalars.index
                )
        elif by == "cluster_size_norm":
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
        timing_info["series_creation_time"] = (
            time.perf_counter() - start_series_creation
        )
    else:  # 'by' is not an npy_dependent_method, try to get column directly
        start_series_creation = time.perf_counter()
        if by in df_scalars.columns:
            series_to_bin = df_scalars[by].copy()
        else:
            raise ValueError(
                f"  Error: Column '{by}' not found in scalars.feather for {batch_dir_path}."
            )
        timing_info["series_creation_time"] = (
            time.perf_counter() - start_series_creation
        )

    return series_to_bin, df_scalars, original_batch_size, timing_info


def calculate_df_agg_for_combination(
    data_dir: str,
    decimals: int = 2,
    ascending_confidence: bool = True,
    by: str = "pred_llr",
    norm_order: float | None = None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    verbose: bool = False,
    disable_tqdm: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Calculate the post-selection DataFrame (df_agg) for batch directories in a single parameter combination directory.
    Reads data from batch directories, each containing 'scalars.feather',
    'cluster_sizes.npy', 'cluster_llrs.npy', and 'offsets.npy'.
    Uses simple rounding and counting instead of histogram binning for efficiency.

    Parameters
    ----------
    data_dir : str
        Directory path containing batch subdirectories for a single parameter combination.
    decimals : int, optional
        Number of decimal places to round to. Can be negative for rounding to tens, hundreds, etc.
        Defaults to 2.
    ascending_confidence : bool, optional
        Indicates the relationship between the aggregated value and decoding confidence.
        If True (default), a higher value implies higher confidence. Values are rounded down (floor).
        If False, a higher value implies lower confidence. Values are rounded up (ceil).
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
    disable_tqdm : bool, optional
        Whether to disable tqdm progress bars. Defaults to False.

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
        print(
            f"Using decimal places: {decimals}, ascending_confidence: {ascending_confidence}"
        )

    (
        df_agg,
        total_rows_processed,
        total_samples_considered,
        detailed_timing,
    ) = _process_and_aggregate_batches_single_pass(
        batch_dir_paths,
        by,
        decimals,
        ascending_confidence,
        norm_order,
        priors,
        sample_indices,
        verbose,
        disable_tqdm,
    )

    # Print detailed benchmarking results if verbose
    if verbose:
        print("=== DETAILED BENCHMARKING RESULTS ===")
        print(
            f"Total samples considered from scalars.feather: {total_samples_considered}"
        )
        print(f"Total valid entries aggregated: {total_rows_processed}")
        print("--- Per-step timing breakdown ---")
        overall_total_time = (
            sum(detailed_timing["overall"].values())
            if detailed_timing["overall"]
            else 0
        )
        for step, timing in detailed_timing["overall"].items():
            if isinstance(timing, dict):
                print(f"{step}:")
                for substep, time_val in timing.items():
                    percentage = (
                        (time_val / overall_total_time) * 100
                        if overall_total_time > 0
                        else 0
                    )
                    print(f"  {substep}: {time_val:.4f}s ({percentage:.1f}%)")
            else:
                percentage = (
                    (timing / overall_total_time) * 100 if overall_total_time > 0 else 0
                )
                print(f"{step}: {timing:.4f}s ({percentage:.1f}%)")

                # Show batch_aggregation_time breakdown
                if (
                    step == "batch_aggregation_time"
                    and detailed_timing["batch_aggregation_breakdown"]
                ):
                    for substep, time_val in detailed_timing[
                        "batch_aggregation_breakdown"
                    ].items():
                        sub_percentage = (time_val / timing) * 100 if timing > 0 else 0
                        overall_percentage = (
                            (time_val / overall_total_time) * 100
                            if overall_total_time > 0
                            else 0
                        )
                        print(
                            f"  {substep}: {time_val:.4f}s ({sub_percentage:.1f}% of batch_agg, {overall_percentage:.1f}% overall)"
                        )
        print("--- Batch processing statistics ---")
        if detailed_timing["batch_stats"]:
            batch_times = detailed_timing["batch_stats"]
            print(f"Average time per batch: {np.mean(batch_times):.4f}s")
            print(f"Median time per batch: {np.median(batch_times):.4f}s")
            print(
                f"Min/Max time per batch: {np.min(batch_times):.4f}s / {np.max(batch_times):.4f}s"
            )
            print(f"Std deviation: {np.std(batch_times):.4f}s")
        print("=====================================")

    # Check if any data was processed
    if total_rows_processed == 0:
        print(
            f"Warning: Processed 0 valid entries for aggregation method {by}. Output df_agg will be empty."
        )

    if verbose:
        print(
            f"  -> Generated df_agg with {len(df_agg)} rows from {total_rows_processed} total valid aggregated entries ({total_samples_considered} samples initially considered), using method '{by}'."
        )

    return df_agg, total_rows_processed


def _process_and_aggregate_batches_single_pass(
    batch_dir_paths: List[str],
    by: str,
    decimals: int,
    ascending_confidence: bool,
    norm_order: float | None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    verbose: bool = False,
    disable_tqdm: bool = False,
) -> Tuple[pd.DataFrame, int, int, dict]:
    """
    Process all batch directories in a single pass and aggregate data using rounding and counting.

    Parameters
    ----------
    batch_dir_paths : List[str]
        List of batch directory paths.
    by : str
        Aggregation method.
    decimals : int
        Number of decimal places to round to.
    ascending_confidence : bool
        Whether higher values mean higher confidence (determines rounding direction).
    norm_order : float, optional
        Norm order for norm-based methods.
    priors : np.ndarray, optional
        1D array of prior probabilities for each bit.
    sample_indices : np.ndarray, optional
        Array of global sample indices to include. If None, all samples are included.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    df_agg : pd.DataFrame
        Aggregated result DataFrame.
    total_rows_processed : int
        Total number of rows processed.
    total_samples_considered : int
        Total number of samples considered.
    detailed_timing : dict
        Detailed timing information for all steps.
    """
    # Initialize aggregation containers
    batch_dataframes = []  # List to store DataFrames from each batch
    total_rows_processed = 0
    total_samples_considered = 0
    has_bplsd_data = False

    # Initialize detailed timing tracking
    detailed_timing = {
        "overall": {
            "file_check_time": 0.0,
            "scalars_load_time": 0.0,
            "sample_filtering_time": 0.0,
            "cluster_file_load_time": 0.0,
            "cluster_calculation_time": 0.0,
            "series_creation_time": 0.0,
            "dropna_time": 0.0,
            "batch_aggregation_time": 0.0,
            "dataframe_concat_time": 0.0,
            "final_aggregation_time": 0.0,
        },
        "batch_stats": [],  # Time for each individual batch
        "batch_aggregation_breakdown": {
            "rounding_time": 0.0,
            "dataframe_creation_time": 0.0,
            "numba_operations_time": 0.0,
            "groupby_time": 0.0,
        },
    }

    if verbose:
        print(f"Processing {len(batch_dir_paths)} batch directories in single pass...")

    desc_text = f"Aggregation for {by}"
    current_batch_start_idx = 0

    for batch_idx, batch_dir in enumerate(
        tqdm(batch_dir_paths, desc=desc_text, disable=disable_tqdm)
    ):
        batch_start_time = time.perf_counter()

        series_to_aggregate, df_scalars, original_batch_size, batch_timing = (
            _get_values_for_binning_from_batch(
                batch_dir_path=batch_dir,
                by=by,
                norm_order=norm_order,
                priors=priors,
                sample_indices=sample_indices,
                batch_start_idx=current_batch_start_idx,
                verbose=verbose > 1,
            )
        )

        # Accumulate batch timing info
        for key, value in batch_timing.items():
            if key in detailed_timing["overall"]:
                detailed_timing["overall"][key] += value

        if series_to_aggregate is None or df_scalars is None or df_scalars.empty:
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

        # Update has_bplsd_data flag
        if has_bplsd_cols:
            has_bplsd_data = True

        start_dropna = time.perf_counter()
        series_to_aggregate_cleaned = series_to_aggregate.dropna()
        detailed_timing["overall"]["dropna_time"] += time.perf_counter() - start_dropna

        if series_to_aggregate_cleaned.empty:
            if verbose:
                print(
                    f"  No valid (non-NaN) data to aggregate for 'by={by}' in {os.path.basename(batch_dir)} after dropna. Skipping."
                )
            continue

        total_rows_processed += len(series_to_aggregate_cleaned)

        start_batch_agg = time.perf_counter()
        batch_agg_df = _round_and_aggregate_batch_data(
            series_to_aggregate_cleaned,
            df_scalars,
            decimals,
            ascending_confidence,
            has_bplsd_cols,
            detailed_timing["batch_aggregation_breakdown"],
        )

        # Add the aggregated DataFrame to our list
        batch_dataframes.append(batch_agg_df)
        batch_agg_time = time.perf_counter() - start_batch_agg
        detailed_timing["overall"]["batch_aggregation_time"] += batch_agg_time

        # Track individual batch timing
        batch_total_time = time.perf_counter() - batch_start_time
        detailed_timing["batch_stats"].append(batch_total_time)

        # Update batch start index for next iteration using the original batch size
        current_batch_start_idx += original_batch_size

        if verbose > 1:
            print(
                f"  Batch {batch_idx+1}/{len(batch_dir_paths)} ({os.path.basename(batch_dir)}): {batch_total_time:.3f}s, {len(series_to_aggregate_cleaned)} samples"
            )

    # Combine all batch DataFrames and perform final aggregation
    start_concat = time.perf_counter()
    if not batch_dataframes:
        # Return empty DataFrame with correct structure
        columns = ["count", "num_fails"]
        if has_bplsd_data:
            columns.extend(["num_converged", "num_converged_fails"])
        df_agg = pd.DataFrame(columns=columns).set_index(pd.Index([], name=by))
    else:
        # Concatenate all batch DataFrames
        combined_df = pd.concat(batch_dataframes, axis=0)
        detailed_timing["overall"]["dataframe_concat_time"] = (
            time.perf_counter() - start_concat
        )

        # Perform final aggregation by summing across all batches for each rounded_value
        start_final_agg = time.perf_counter()
        df_agg = combined_df.groupby(combined_df.index).sum()
        detailed_timing["overall"]["final_aggregation_time"] = (
            time.perf_counter() - start_final_agg
        )

        # Set the index name to match the aggregation column
        df_agg.index.name = by

    return (
        df_agg,
        total_rows_processed,
        total_samples_considered,
        detailed_timing,
    )


def _round_and_aggregate_batch_data(
    series_to_aggregate_cleaned: pd.Series,
    df_scalars: pd.DataFrame,
    decimals: int,
    ascending_confidence: bool,
    has_bplsd_cols: bool,
    aggregation_timing: dict = None,
) -> pd.DataFrame:
    """
    Round values and aggregate batch data into a DataFrame.

    Parameters
    ----------
    series_to_aggregate_cleaned : pd.Series
        Series of values to aggregate after removing NaN values.
    df_scalars : pd.DataFrame
        DataFrame containing scalar metrics for each sample.
    decimals : int
        Number of decimal places to round to.
    ascending_confidence : bool
        Whether higher values mean higher confidence (determines rounding direction).
    has_bplsd_cols : bool
        Whether BPLSD columns are available in df_scalars.
    aggregation_timing : dict, optional
        Dictionary to store timing information for aggregation steps.

    Returns
    -------
    batch_agg : pd.DataFrame
        Aggregated DataFrame with rounded values as index and count statistics as columns.
    """
    if aggregation_timing is None:
        aggregation_timing = {}

    # Round values based on decimals and ascending_confidence
    start_rounding = time.perf_counter()
    multiplier = 10**decimals
    if ascending_confidence:
        # Round down (floor) - keep as integers
        rounded_values_int = np.floor(
            series_to_aggregate_cleaned.values * multiplier
        ).astype(np.int32)
    else:
        # Round up (ceil) - keep as integers
        rounded_values_int = np.ceil(
            series_to_aggregate_cleaned.values * multiplier
        ).astype(np.int32)
    rounding_time = time.perf_counter() - start_rounding
    if "rounding_time" in aggregation_timing:
        aggregation_timing["rounding_time"] += rounding_time

    # Create DataFrame for aggregation using integer rounded values
    start_dataframe_creation = time.perf_counter()

    # Check if series_to_aggregate_cleaned.index contains all indices from df_scalars
    # If so, use entire columns directly instead of masking for better performance
    use_full_columns = len(series_to_aggregate_cleaned.index) == len(
        df_scalars
    ) and series_to_aggregate_cleaned.index.equals(df_scalars.index)

    if use_full_columns:
        # Use entire columns directly - much faster
        agg_df = pd.DataFrame(
            {
                "rounded_value": rounded_values_int,
                "fail": df_scalars["fail"].values,
            }
        )

        if has_bplsd_cols:
            agg_df["converge"] = df_scalars["converge"].values
            agg_df["fail_bp"] = df_scalars["fail_bp"].values
    else:
        # Use index-based selection for subset
        agg_df = pd.DataFrame(
            {
                "rounded_value": rounded_values_int,
                "fail": df_scalars.loc[
                    series_to_aggregate_cleaned.index, "fail"
                ].values,
            }
        )

        if has_bplsd_cols:
            agg_df["converge"] = df_scalars.loc[
                series_to_aggregate_cleaned.index, "converge"
            ].values
            agg_df["fail_bp"] = df_scalars.loc[
                series_to_aggregate_cleaned.index, "fail_bp"
            ].values

    # Create additional column for converged_fails if needed
    if has_bplsd_cols:
        agg_df["converged_fails"] = agg_df["converge"] & agg_df["fail_bp"]

    dataframe_creation_time = time.perf_counter() - start_dataframe_creation
    if "dataframe_creation_time" in aggregation_timing:
        aggregation_timing["dataframe_creation_time"] += dataframe_creation_time

    # Convert bool columns to uint8 for numba compatibility
    # numba engine cannot handle bool dtype, so we need to convert to uint8
    start_numba_prep = time.perf_counter()
    agg_df_numba = agg_df.copy()
    agg_df_numba["fail"] = agg_df_numba["fail"].astype(np.uint8)
    if has_bplsd_cols:
        agg_df_numba["converge"] = agg_df_numba["converge"].astype(np.uint8)
        agg_df_numba["converged_fails"] = agg_df_numba["converged_fails"].astype(
            np.uint8
        )
    numba_prep_time = time.perf_counter() - start_numba_prep
    if "numba_operations_time" in aggregation_timing:
        aggregation_timing["numba_operations_time"] += numba_prep_time

    # Define aggregation functions for vectorized operations
    # Note: "size" doesn't support engine="numba", so we need to handle size and sum separately

    # First, perform size aggregation (count) without numba engine
    start_groupby = time.perf_counter()
    batch_agg_size = agg_df.groupby("rounded_value").size()

    # Then, perform sum aggregations with numba engine for better performance
    sum_agg_funcs = {"fail": "sum"}  # sum gives num_fails
    if has_bplsd_cols:
        sum_agg_funcs["converge"] = "sum"
        sum_agg_funcs["converged_fails"] = "sum"

    batch_agg_sum = agg_df_numba.groupby("rounded_value").agg(
        sum_agg_funcs, engine="numba"
    )

    # Combine size and sum results
    batch_agg = pd.DataFrame(index=batch_agg_size.index)
    batch_agg["count"] = batch_agg_size
    batch_agg["num_fails"] = batch_agg_sum["fail"]

    if has_bplsd_cols:
        batch_agg["num_converged"] = batch_agg_sum["converge"]
        batch_agg["num_converged_fails"] = batch_agg_sum["converged_fails"]

    # Convert index from integers back to floats by dividing by multiplier and ensure float64 dtype
    batch_agg.index = (batch_agg.index / multiplier).astype(np.float64).round(decimals)

    groupby_time = time.perf_counter() - start_groupby
    if "groupby_time" in aggregation_timing:
        aggregation_timing["groupby_time"] += groupby_time

    return batch_agg


def extract_sample_metric_values(
    data_dir: str,
    *,
    by: str,
    norm_order: float | None = None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
    verbose: bool = False,
    disable_tqdm: bool = False,
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

    for batch_dir in tqdm(batch_dir_paths, desc=desc_text, disable=disable_tqdm):
        try:
            series_to_extract, df_scalars, original_batch_size, _ = (
                _get_values_for_binning_from_batch(
                    batch_dir_path=batch_dir,
                    by=by,
                    norm_order=norm_order,
                    priors=priors,
                    sample_indices=sample_indices,
                    batch_start_idx=current_batch_start_idx,
                    verbose=verbose > 1,
                )
            )

            # Update batch start index for next iteration using the original batch size
            current_batch_start_idx += original_batch_size

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
    decimals: int = 2,
    ascending_confidence: bool = True,
    norm_order: float | None = None,
    priors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    df_existing: pd.DataFrame | None = None,
    verbose: bool = False,
    disable_tqdm: bool = False,
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
    decimals : int, optional
        Number of decimal places to round to. Can be negative for rounding to tens, hundreds, etc.
        Defaults to 2.
    ascending_confidence : bool, optional
        Indicates the relationship between the aggregated value and decoding confidence.
        If True (default), values are rounded down (floor).
        If False, values are rounded up (ceil).
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
    disable_tqdm : bool, optional
        Whether to disable tqdm progress bars. Defaults to False.

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

        # Check if decimals parameter matches the precision used in df_existing
        def _detect_decimals_from_index(index_values: pd.Index) -> int:
            """Detect the number of decimal places used in the index values."""
            if len(index_values) == 0:
                return decimals  # Default to current decimals if empty

            # Check only first 10 values for efficiency
            sample_values = index_values[:10]
            
            # Convert to string and find maximum decimal places
            max_decimals = 0
            for value in sample_values:
                if pd.isna(value):
                    continue
                str_value = f"{float(value):.15f}".rstrip("0").rstrip(".")
                if "." in str_value:
                    decimal_places = len(str_value.split(".")[1])
                    max_decimals = max(max_decimals, decimal_places)
            return max_decimals

        existing_decimals = _detect_decimals_from_index(df_existing_copy.index)

        if existing_decimals != decimals:
            if verbose:
                print(
                    f"Existing decimals ({existing_decimals}) != current decimals ({decimals}). Reprocessing."
                )
        else:
            try:
                existing_total_shots = df_existing_copy["count"].sum()
                file_total_shots, _ = get_existing_shots(data_dir)

                if existing_total_shots == file_total_shots:
                    if verbose:
                        print(
                            f"Existing shots ({existing_total_shots}) match file shots ({file_total_shots}) and decimals match ({decimals}). Returning existing data."
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
        print(f"Decimal places: {decimals}")
        if norm_order is not None:
            print(f"Norm order: {norm_order}")
        if sample_indices is not None:
            print(
                f"Using {len(sample_indices)} specific sample indices (range: {sample_indices.min()}-{sample_indices.max()})"
            )

    df_agg, total_rows_processed = calculate_df_agg_for_combination(
        data_dir=data_dir,
        decimals=decimals,
        ascending_confidence=ascending_confidence,
        by=by,
        norm_order=norm_order,
        priors=priors,
        sample_indices=sample_indices,
        verbose=verbose,
        disable_tqdm=disable_tqdm,
    )

    if verbose:
        if df_agg.empty:
            print("No data was processed successfully.")
        else:
            print(
                f"Successfully aggregated {total_rows_processed} samples into {len(df_agg)} groups."
            )

    return df_agg, False
