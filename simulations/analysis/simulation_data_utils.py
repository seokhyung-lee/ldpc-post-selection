import gc
import os
import re
from typing import Tuple

import numba
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


def natural_sort_key(s: str) -> int:
    """
    Extracts the numerical part from a filename like 'data_123.feather'
    for natural sorting.

    Parameters
    ----------
    s : str
        The filename string.

    Returns
    -------
    int
        The extracted number, or -1 if no number is found.
    """
    # Extract the number after 'data_' and before '.feather'
    match = re.search(r"data_(\d+)\.feather", os.path.basename(s))
    return int(match.group(1)) if match else -1


@numba.njit(fastmath=True, cache=True)
def _calculate_histograms_numba(
    values_np: np.ndarray,  # Expect float array
    fail_mask_np: np.ndarray,  # Expect boolean array
    bin_edges: np.ndarray,  # Expect float array
    total_counts_hist: np.ndarray,  # Expect int64 array
    fail_counts_hist: np.ndarray,  # Expect int64 array
    converge_mask_np: np.ndarray,  # Expect boolean array
    converge_counts_hist: np.ndarray,  # Expect int64 array
    fail_converge_counts_hist: np.ndarray,  # Expect int64 array
    fail_bp_mask_np: np.ndarray,  # Expect boolean array
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate total, fail, converge, and fail_converge histograms using Numba with an optimized single loop.

    Parameters
    ----------
    values_np : 1D numpy array of float
        Values to be histogrammed (e.g., cluster fractions or other column data).
        NaNs should be removed before calling.
    fail_mask_np : 1D numpy array of bool
        Indicates failures, aligned with values_np.
    bin_edges : 1D numpy array of float
        Defines the histogram bin edges, must be sorted.
    total_counts_hist : 1D numpy array of int64
        Histogram counts for all samples (updated in-place).
    fail_counts_hist : 1D numpy array of int64
        Histogram counts for failed samples (updated in-place).
    converge_mask_np : 1D numpy array of bool
        Indicates convergence, aligned with values_np.
    converge_counts_hist : 1D numpy array of int64
        Histogram counts for converged samples (updated in-place).
    fail_converge_counts_hist : 1D numpy array of int64
        Histogram counts for samples where both fail_bp and converge are true (updated in-place).
    fail_bp_mask_np : 1D numpy array of bool
        Indicates failures based on BP decoder, aligned with values_np.

    Returns
    -------
    total_counts_hist : 1D numpy array of int64
        The updated histogram counts for all samples.
    fail_counts_hist : 1D numpy array of int64
        The updated histogram counts for failed samples.
    converge_counts_hist : 1D numpy array of int64
        The updated histogram counts for converged samples.
    fail_converge_counts_hist : 1D numpy array of int64
        The updated histogram counts for fail_converged samples.

    Notes
    -----
    This implementation manually calculates the histogram bins in a single pass
    for potentially better performance within Numba compared to calling
    np.histogram multiple times. It assumes NaNs have been filtered from
    values_np before calling. It replicates np.histogram's behavior
    regarding bin edges: bins are [left, right), except the last bin which
    is [left, right].
    """
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:  # Handle empty or invalid bin_edges
        return (
            total_counts_hist,
            fail_counts_hist,
            converge_counts_hist,
            fail_converge_counts_hist,
        )

    for i in range(len(values_np)):
        val = values_np[i]

        # Check if value is within the histogram range
        if val < bin_edges[0] or val > bin_edges[-1]:
            continue

        # Determine the bin index using searchsorted
        bin_idx = np.searchsorted(bin_edges[:-1], val, side="right")

        if bin_idx == n_bins:
            if val == bin_edges[-1]:
                bin_idx = n_bins - 1
            else:
                continue
        else:
            bin_idx = bin_idx - 1

        if 0 <= bin_idx < n_bins:
            total_counts_hist[bin_idx] += 1
            if fail_mask_np[i]:
                fail_counts_hist[bin_idx] += 1
            if converge_mask_np[i]:
                converge_counts_hist[bin_idx] += 1
            if fail_bp_mask_np[i] and converge_mask_np[i]:
                fail_converge_counts_hist[bin_idx] += 1

    return (
        total_counts_hist,
        fail_counts_hist,
        converge_counts_hist,
        fail_converge_counts_hist,
    )


def calculate_confidence_interval(
    n: int, k: int, alpha: float = 0.05, method: str = "wilson"
) -> Tuple[float, float]:
    """
    Calculate the proportion and confidence interval width.

    Parameters
    ----------
    n : int
        Total number of trials.
    k : int
        Number of successes.
    alpha : float, optional
        Significance level (default is 0.05 for 95% confidence).
    method : str, optional
        Method for confidence interval calculation (default is "wilson").

    Returns
    -------
    p : float
        Estimated proportion (midpoint of the confidence interval).
    delta_p : float
        Half-width of the confidence interval.
    """
    p_low, p_upp = proportion_confint(k, n, alpha=alpha, method=method)
    p = (p_low + p_upp) / 2
    delta_p = p_upp - p
    return p, delta_p


def get_df_ps(df_agg: pd.DataFrame, ascending_confidence: bool = True) -> pd.DataFrame:
    """
    Calculate post-selection probabilities and statistics from aggregated data.

    Parameters
    ----------
    df_agg : pd.DataFrame
        Aggregated data with columns like 'count', 'num_fails', 'num_converged', 'num_converged_fails'.
        The index should represent the binned post-selection variable.
    ascending_confidence : bool, optional
        If True (default), data is processed assuming higher index values mean higher confidence
        (data is reversed internally for cumulative sums). If False, data is processed as is.

    Returns
    -------
    df_ps : pd.DataFrame
        DataFrame containing post-selection probabilities (p_fail, p_abort) and their confidence
        intervals (delta_p_fail, delta_p_abort), both considering and ignoring convergence.
        Also includes cumulative counts. The index matches the input df_agg index.
    """
    if ascending_confidence:
        df_agg = df_agg.iloc[::-1]

    shots = df_agg["count"].sum()

    # --- Ignoring convergence ---
    counts = df_agg["count"].cumsum()
    num_fails = df_agg["num_fails"].cumsum()

    pfail, delta_pfail = calculate_confidence_interval(counts, num_fails)
    # Calculate acceptance probability: (total shots - aborted shots) / total shots
    # Aborted shots = total shots - counts (accepted shots)
    pacc, delta_pacc = calculate_confidence_interval(shots, counts)

    # --- Treating convergence = confident ---
    # Calculate counts considering convergence: Start with total converged, add non-converged counts bin-wise
    counts_conv = df_agg["count"] - df_agg["num_converged"]
    if not counts_conv.empty:
        counts_conv.iloc[0] += df_agg["num_converged"].sum()
    counts_conv = counts_conv.cumsum()

    # Ensure total count matches shots
    if not counts_conv.empty:
        assert np.isclose(
            counts_conv.iloc[-1], shots
        ), f"counts_conv final ({counts_conv.iloc[-1]}) != shots ({shots})"

    # Calculate failures considering convergence: Start with total converged fails, add non-converged fails bin-wise
    num_fails_conv = df_agg["num_fails"] - df_agg["num_converged_fails"]
    if not num_fails_conv.empty:
        num_fails_conv.iloc[0] += df_agg["num_converged_fails"].sum()
    num_fails_conv = num_fails_conv.cumsum()

    pfail_conv, delta_pfail_conv = calculate_confidence_interval(
        counts_conv, num_fails_conv
    )
    pacc_conv, delta_pacc_conv = calculate_confidence_interval(shots, counts_conv)

    # --- Assemble Results ---
    # Create the DataFrame with the same index as the input (or reversed input)
    df_ps = pd.DataFrame(index=df_agg.index)

    df_ps["p_fail"] = pfail
    df_ps["delta_p_fail"] = delta_pfail
    df_ps["p_abort"] = 1.0 - pacc  # Abort probability = 1 - acceptance probability
    df_ps["delta_p_abort"] = delta_pacc  # Width is the same for p and 1-p

    df_ps["p_fail_conv"] = pfail_conv
    df_ps["delta_p_fail_conv"] = delta_pfail_conv
    df_ps["p_abort_conv"] = 1.0 - pacc_conv
    df_ps["delta_p_abort_conv"] = delta_pacc_conv

    # Add cumulative counts for reference
    df_ps["count"] = counts
    df_ps["num_fails"] = num_fails
    df_ps["count_conv"] = counts_conv
    df_ps["num_fails_conv"] = num_fails_conv

    # If the input was reversed, reverse the output back to match the original df_agg index order
    if ascending_confidence:
        df_ps = df_ps.iloc[::-1]

    return df_ps


@numba.njit(parallel=True, fastmath=True, cache=True)
def _calculate_norms_for_samples(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    norm_order: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the L_p norm for each sample from flattened cluster data,
    excluding the last element of each sample's segment (treated as outside_cluster_value).
    The norm is (sum(value^order))^(1/order) or max(abs(value)) for L-infinity.

    Parameters
    ----------
    flat_data : 1D numpy array
        Flattened cluster values. For each sample, the last value in its segment
        is considered the outside_cluster_value.
    offsets : 1D numpy array
        Starting indices in flat_data for each sample.
    norm_order : float
        The order p for the L_p norm. Must be positive (can be np.inf).

    Returns
    -------
    sample_norms : 1D numpy array
        The calculated norm for the "inside cluster" values of each sample.
        Norm is 0 if no "inside cluster" values exist or if the sample data is empty.
    outside_cluster_values : 1D numpy array
        The last element of each sample's segment in flat_data.
        np.nan if the sample's segment is empty.
    """
    n = offsets.size
    # Pass the sentinel "end" index instead of building it with np.concatenate
    end_of_data = flat_data.size

    # pre-allocate output (NumPy scalars are fine – Numba understands them)
    norms = np.zeros(n, dtype=np.float32)
    outside = np.full(n, np.nan, dtype=np.float32)

    # Constant-fold tests that do not depend on i
    use_inf = np.isinf(norm_order)

    for i in numba.prange(n):  # ← prange enables multi-threaded loop
        # Get raw start and stop from offsets (can be float)
        _raw_start = offsets[i]
        _raw_stop = offsets[i + 1] if i < n - 1 else end_of_data

        # Convert to int for indexing and range()
        start = int(_raw_start)
        stop = int(_raw_stop)

        if stop <= start:  # empty slice – nothing to do
            continue

        # last element belongs to "outside"
        last_val = flat_data[stop - 1]
        outside[i] = last_val

        # all but the last one are inside; bail if length-0
        if stop - start == 1:
            continue

        # NB: working on the original array avoids an extra allocation
        if use_inf:  # L∞
            max_abs = 0.0
            for j in range(start, stop - 1):
                v = flat_data[j]
                if v < 0:
                    v = -v
                if v > max_abs:
                    max_abs = v
            norms[i] = max_abs
        else:
            # special-case the common p = 1, 2 to avoid slow np.power
            if norm_order == 1.0:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    s += v if v >= 0 else -v
                norms[i] = s
            elif norm_order == 2.0:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    s += v * v
                norms[i] = np.sqrt(s)
            elif norm_order == 0.5:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    s += np.sqrt(v)
                norms[i] = s * s
            else:
                s = 0.0
                for j in range(start, stop - 1):
                    v = flat_data[j]
                    # fastmath allows powerf; fall back to np.power otherwise
                    s += v**norm_order
                norms[i] = s ** (1.0 / norm_order)

    return norms, outside


def _get_values_for_binning_from_batch(
    batch_dir_path: str, by: str, norm_order: float | None, verbose: bool = False
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Load data from a single batch directory and extract/calculate the series
    to be used for binning, along with the scalars DataFrame.
    Handles different 'by' methods including direct column reads and norm calculations.

    Returns
    -------
    series_to_bin : pd.Series
        The data series to be binned.
    df_scalars : pd.DataFrame
        The DataFrame loaded from 'scalars.feather'.

    Raises
    ------
    FileNotFoundError
        If 'scalars.feather' or other required .npy files are not found.
    IOError
        If there's an issue loading 'scalars.feather' (other than not found).
    ValueError
        If 'scalars.feather' is empty, 'by' column is invalid, norm_order is missing/invalid,
        or array length mismatches occur.
    KeyError
        If the specified 'by' column is not in 'scalars.feather'.
    RuntimeError
        For internal logic errors or unexpected issues during norm calculation.
    """
    scalars_path = os.path.join(batch_dir_path, "scalars.feather")

    df_scalars = pd.read_feather(scalars_path)

    if df_scalars.empty:
        # This handles the case where read_feather succeeds but returns an empty DataFrame
        raise ValueError(
            f"scalars.feather at {scalars_path} loaded as an empty DataFrame."
        )

    series_to_bin: pd.Series | None = None  # Initialize to satisfy type checker

    if by in ["pred_llr", "detector_density"]:
        if by not in df_scalars.columns:
            raise KeyError(
                f"Column '{by}' not found in scalars.feather for {batch_dir_path}. Available columns: {df_scalars.columns.tolist()}"
            )
        series_to_bin = df_scalars[by]

    elif by in [
        "cluster_size_norm",
        "cluster_llr_norm",
        "cluster_size_norm_gap",
        "cluster_llr_norm_gap",
    ]:
        if norm_order is None:
            raise ValueError(
                f"'norm_order' is required for 'by={by}' but not provided for batch: {os.path.basename(batch_dir_path)}"
            )

        cluster_sizes_path = os.path.join(batch_dir_path, "cluster_sizes.npy")
        cluster_llrs_path = os.path.join(batch_dir_path, "cluster_llrs.npy")
        offsets_path = os.path.join(batch_dir_path, "offsets.npy")

        cluster_sizes_all_samples = np.load(cluster_sizes_path, allow_pickle=True)
        cluster_llrs_all_samples = np.load(cluster_llrs_path, allow_pickle=True)
        offsets = np.load(offsets_path)
        offsets = offsets[
            :-1
        ]  # Remove the last element as it's one greater than num_samples

        data_for_norm = None
        # Determine data_for_norm based on 'by'
        if by.startswith("cluster_size_norm"):
            data_for_norm = cluster_sizes_all_samples
        elif by.startswith("cluster_llr_norm"):
            data_for_norm = cluster_llrs_all_samples

        if data_for_norm is None:  # Should not happen if by is one of the norm methods
            raise RuntimeError(
                f"Internal error: data_for_norm is None for by='{by}' in batch {os.path.basename(batch_dir_path)}. This path should be unreachable."
            )

        (
            calculated_norms,
            outside_values_from_func,
        ) = _calculate_norms_for_samples(
            data_for_norm, offsets, norm_order  # type: ignore
        )

        if len(calculated_norms) != len(df_scalars.index):
            raise ValueError(
                f"Mismatch in array lengths from norm calculation ({len(calculated_norms)}) and scalar samples ({len(df_scalars.index)}) for by='{by}' in {os.path.basename(batch_dir_path)}."
            )

        # Create a Series for the norms, aligned with df_scalars' index
        norm_series = pd.Series(calculated_norms, index=df_scalars.index, dtype=float)

        if by.endswith("_gap"):
            # For "gap" methods, series_to_bin = outside_value - norm_value
            outside_series = pd.Series(
                outside_values_from_func, index=df_scalars.index, dtype=float
            )
            series_to_bin = outside_series - norm_series
        else:
            # For non-"gap" norm methods, series_to_bin = norm_value
            series_to_bin = norm_series

    else:
        raise ValueError(
            f"Unsupported or unhandled 'by' method: '{by}' provided to _get_values_for_binning_from_batch for {os.path.basename(batch_dir_path)}"
        )

    # Ensure the series returned has the same index as df_scalars
    if not series_to_bin.index.equals(df_scalars.index):
        if verbose:
            print(
                f"  Warning (simulation_data_utils): series_to_bin index does not match df_scalars index for {by} in {os.path.basename(batch_dir_path)}. Reindexing."
            )
        # This reindex is crucial. If series_to_bin was formed from subset of indices (e.g. only processed_sample_indices),
        # it needs to be expanded to match df_scalars for consistent downstream processing.
        # Fill_value=np.nan ensures that samples that couldn't be processed for norms (if any) are marked as NaN.
        series_to_bin = series_to_bin.reindex(df_scalars.index, fill_value=np.nan)

    return series_to_bin, df_scalars
