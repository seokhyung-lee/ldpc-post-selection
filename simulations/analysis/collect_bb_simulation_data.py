import gc  # Add garbage collector import
import glob
import os
import re
import time  # Import time module
from typing import List, Set, Tuple

import numba  # Import Numba
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from src.ldpc_post_selection.build_circuit import build_BB_circuit


def find_bb_simulation_files(
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",  # Base directory
    verbose: bool = False,
) -> List[str]:
    """
    Find BB code circuit simulation data feather files (`data_*.feather`)
    within the specific subdirectory for an (n, T, p) tuple.

    Parameters
    ----------
    n : int
        Number of qubits in the BB code
    T : int
        Number of rounds
    p : float
        Physical error rate
    data_dir : str
        Base directory path containing the `n{n}_T{T}_p{p}` subdirectories.
    verbose : bool, optional
        Whether to print the number of found files. Defaults to False.

    Returns
    -------
    file_paths : list of str
        List of file paths matching `data_*.feather` in the subdirectory.

    Raises
    ------
    FileNotFoundError
        If the base data directory or the specific n{n}_T{T}_p{p} subdirectory does not exist.
    ValueError
        If no `data_*.feather` files are found within the subdirectory.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Base data directory not found: {data_dir}")

    # Construct the specific subdirectory path
    p_str = f"{p:.{len(str(p).split('.')[-1])}f}".rstrip("0").rstrip(
        "."
    )  # Format p for directory name
    sub_dirname = f"n{n}_T{T}_p{p_str}"
    sub_dir_path = os.path.join(data_dir, sub_dirname)

    if not os.path.isdir(sub_dir_path):
        raise FileNotFoundError(
            f"Subdirectory not found for n={n}, T={T}, p={p}: {sub_dir_path}"
        )

    # Search for data_*.feather files within the subdirectory
    file_pattern = os.path.join(sub_dir_path, "data_*.feather")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        raise ValueError(f"No data files (data_*.feather) found in {sub_dir_path}.")

    # Sort files naturally (e.g., data_1, data_2, ..., data_10)
    def natural_sort_key(s):
        # Extract the number after 'data_' and before '.feather'
        match = re.search(r"data_(\d+)\.feather", os.path.basename(s))
        return int(match.group(1)) if match else -1

    file_paths.sort(key=natural_sort_key)

    if verbose:
        print(f"Found {len(file_paths)} data files in {sub_dirname}.")

    return file_paths


# --- Numba JIT function for histogram calculation ---
@numba.jit(nopython=True, fastmath=True)  # Use nopython=True for best performance
def _calculate_histograms_numba(
    values_np: np.ndarray,  # Expect float array, renamed from cluster_frac_np
    fail_mask_np: np.ndarray,  # Expect boolean array
    bin_edges: np.ndarray,  # Expect float array
    total_counts_hist: np.ndarray,  # Expect int64 array
    fail_counts_hist: np.ndarray,  # Expect int64 array
    converge_mask_np: np.ndarray,  # Expect boolean array
    converge_counts_hist: np.ndarray,  # Expect int64 array
    fail_converge_counts_hist: np.ndarray,  # Expect int64 array
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate total, fail, converge, and fail_converge histograms using Numba with an optimized single loop.

    Parameters
    ----------
    values_np : 1D numpy array of float
        Values to be histogrammed (e.g., cluster fractions or other column data).
        NaNs should be removed before calling. Renamed from cluster_frac_np.
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
        Histogram counts for samples where both fail and converge are true (updated in-place).

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
        # Note: np.histogram includes the right edge only for the last bin.
        if val < bin_edges[0] or val > bin_edges[-1]:
            continue

        # Determine the bin index using searchsorted
        # np.searchsorted finds the index where val would be inserted to maintain order.
        # We search in all edges except the last one initially.
        bin_idx = np.searchsorted(bin_edges[:-1], val, side="right")

        # Special case: If val equals the last edge, it belongs to the last bin.
        # searchsorted on bin_edges[:-1] would return n_bins in this case.
        if bin_idx == n_bins:
            # Check if it's exactly the last edge value
            if val == bin_edges[-1]:
                bin_idx = n_bins - 1  # Assign to the last bin
            else:
                # This case implies val > bin_edges[-1], but we already checked range.
                # However, for robustness, skip if something unexpected happens.
                continue
        else:
            # General case: bin index is the position found minus 1.
            # e.g., val=1.5, edges=[1,2,3]. searchsorted -> 1. bin_idx=1-1=0. (Bin [1,2))
            # e.g., val=2.0, edges=[1,2,3]. searchsorted -> 2. bin_idx=2-1=1. (Bin [2,3))
            bin_idx = bin_idx - 1

        # Ensure the calculated bin index is valid before incrementing
        if 0 <= bin_idx < n_bins:
            total_counts_hist[bin_idx] += 1
            if fail_mask_np[i]:
                fail_counts_hist[bin_idx] += 1
            if converge_mask_np[i]:
                converge_counts_hist[bin_idx] += 1
            if (
                fail_mask_np[i] and converge_mask_np[i]
            ):  # Check for both fail and converge
                fail_converge_counts_hist[bin_idx] += 1

    return (
        total_counts_hist,
        fail_counts_hist,
        converge_counts_hist,
        fail_converge_counts_hist,
    )


def calculate_df_agg_for_combination(
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
    num_hist_bins: int = 1000,
    min_value_override: float | None = None,  # Renamed from min_cluster_frac_override
    max_value_override: float | None = None,  # Renamed from max_cluster_frac_override
    ascending_confidence: bool = True,
    by: str = "cluster_frac",  # Renamed from aggregation_column_name, default changed
    verbose: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Calculate the post-selection DataFrame (df_agg) for a single (n, T, p) combination.
    Uses Numba JIT for histogram calculation.
    The range for the aggregation value histogram can be auto-detected or user-specified.

    Parameters
    ----------
    n : int
        Number of qubits in the BB code.
    T : int
        Number of rounds.
    p : float
        Physical error rate.
    data_dir : str
        Directory path containing the simulation data files.
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
        Column to aggregate by. Defaults to "cluster_frac".
        If "cluster_frac", values from "cluster_size_sum" are read and
        normalized by `num_errors`.
        If "cluster_frac_bp_llr", the ratio cluster_bp_llr_sum / (cluster_bp_llr_sum + outside_cluster_bp_llr) is used.
        If "cluster_frac_llr", the ratio cluster_llr_sum / (cluster_llr_sum + outside_cluster_llr) is used.
        If "cluster_gap_bp_llr", the difference outside_cluster_bp_llr - cluster_bp_llr_sum is used.
        If "cluster_gap_llr", the difference outside_cluster_llr - cluster_llr_sum is used.
        Otherwise, `by` is treated as the direct column name to read and use raw values from.
    verbose : bool, optional
        Whether to print progress and benchmarking information. Defaults to False.

    Returns
    -------
    df_agg : pd.DataFrame
        DataFrame with columns ['c', 'num_accs', 'num_fails']. Empty if fails.
    total_rows_processed : int
        Total number of simulation rows processed.
    """
    try:
        file_paths = find_bb_simulation_files(
            n=n, T=T, p=p, data_dir=data_dir, verbose=verbose
        )
        if not file_paths:
            print(f"Warning: No files found for n={n}, T={T}, p={p}. Skipping.")
            return pd.DataFrame(), 0

        # --- Get num_errors if needed ---
        num_errors: int | None = None
        perform_normalization_with_num_errors = by == "cluster_frac"

        actual_columns_to_read: list[str]
        if by == "cluster_frac":
            actual_columns_to_read = ["cluster_size_sum"]
        elif by == "cluster_frac_bp_llr":
            actual_columns_to_read = ["cluster_bp_llr_sum", "outside_cluster_bp_llr"]
        elif by == "cluster_frac_llr":
            actual_columns_to_read = ["cluster_llr_sum", "outside_cluster_llr"]
        elif by == "cluster_gap_bp_llr":
            actual_columns_to_read = ["outside_cluster_bp_llr", "cluster_bp_llr_sum"]
        elif by == "cluster_gap_llr":
            actual_columns_to_read = ["outside_cluster_llr", "cluster_llr_sum"]
        else:
            actual_columns_to_read = [by]  # Direct column name

        if perform_normalization_with_num_errors:
            try:
                circuit = build_BB_circuit(n=n, T=T, p=p)
                detector_error_model = circuit.detector_error_model()
                if detector_error_model is None:
                    raise ValueError("Could not get detector_error_model.")
                num_errors = detector_error_model.num_errors
                if (
                    num_errors is None or num_errors < 0
                ):  # Allow num_errors == 0 for empty cluster_frac later
                    raise ValueError(f"Invalid num_errors ({num_errors}).")
            except Exception as e:
                print(
                    f"Warning: Error getting num_errors for n={n}, T={T}, p={p} (needed for normalization): {e}. Skipping."
                )
                return pd.DataFrame(), 0
        # --------------------------------

        # --- Determine value range (min_val, max_val) ---
        actual_min_val: float
        actual_max_val: float

        if min_value_override is not None and max_value_override is not None:
            if min_value_override >= max_value_override:
                raise ValueError(
                    "min_value_override must be less than max_value_override."
                )
            actual_min_val = min_value_override
            actual_max_val = max_value_override
            if verbose:
                print(
                    f"  Using user-specified value range for {by}: [{actual_min_val}, {actual_max_val}]"
                )
        else:
            # Auto-detect range: First pass over files
            if verbose:
                print(f"  Auto-detecting value range for {by} (first pass)...")
            current_min_val = np.inf
            current_max_val = -np.inf
            found_any_valid_value = False
            # Use actual_columns_to_read for the range detection pass
            range_detection_cols = actual_columns_to_read

            for file_path_pass1 in tqdm(file_paths, desc="Range detection"):
                try:
                    df_temp = pd.read_feather(
                        file_path_pass1, columns=range_detection_cols
                    )
                    # Ensure all required columns for the current 'by' mode are present
                    if not df_temp.empty and all(
                        col in df_temp.columns for col in actual_columns_to_read
                    ):
                        if by == "cluster_frac":
                            if (
                                num_errors == 0
                            ):  # Should be caught if num_errors is critical
                                temp_series_to_bin = pd.Series(dtype=float)
                            else:
                                temp_series_to_bin = (
                                    df_temp[actual_columns_to_read[0]] / num_errors
                                )
                        elif by == "cluster_frac_bp_llr":
                            numerator = df_temp["cluster_bp_llr_sum"]
                            denominator = (
                                df_temp["cluster_bp_llr_sum"]
                                + df_temp["outside_cluster_bp_llr"]
                            )
                            temp_series_to_bin = numerator / denominator.replace(
                                0, np.nan
                            )  # Avoid division by zero
                        elif by == "cluster_frac_llr":
                            numerator = df_temp["cluster_llr_sum"]
                            denominator = (
                                df_temp["cluster_llr_sum"]
                                + df_temp["outside_cluster_llr"]
                            )
                            temp_series_to_bin = numerator / denominator.replace(
                                0, np.nan
                            )  # Avoid division by zero
                        elif by == "cluster_gap_bp_llr":
                            temp_series_to_bin = (
                                df_temp["outside_cluster_bp_llr"]
                                - df_temp["cluster_bp_llr_sum"]
                            )
                        elif by == "cluster_gap_llr":
                            temp_series_to_bin = (
                                df_temp["outside_cluster_llr"]
                                - df_temp["cluster_llr_sum"]
                            )
                        else:  # Direct column name
                            temp_series_to_bin = df_temp[actual_columns_to_read[0]]

                        temp_series_to_bin = temp_series_to_bin.dropna()
                        if not temp_series_to_bin.empty:
                            found_any_valid_value = True
                            current_min_val = min(
                                current_min_val, temp_series_to_bin.min()
                            )
                            current_max_val = max(
                                current_max_val, temp_series_to_bin.max()
                            )
                    del df_temp
                except Exception as e_pass1:
                    print(
                        f"Warning: Error during range detection for file {file_path_pass1}: {e_pass1}"
                    )

            if not found_any_valid_value:
                print(
                    f"Warning: No valid data found for {by} for n={n}, T={T}, p={p} during range detection. Skipping."
                )
                return pd.DataFrame(), 0

            actual_min_val = current_min_val
            actual_max_val = current_max_val

            if np.isclose(actual_min_val, actual_max_val):
                if verbose:
                    print(
                        f"  Detected min_value ({actual_min_val}) approx equals max_value ({actual_max_val})."
                    )
                if actual_max_val < actual_min_val:
                    actual_max_val = actual_min_val
            if verbose:
                print(
                    f"  Auto-detected value range for {by}: [{actual_min_val}, {actual_max_val}]"
                )

        # --- Initialize Histogram approach ---
        if not isinstance(num_hist_bins, int) or num_hist_bins < 1:
            raise ValueError("num_hist_bins must be a positive integer.")

        if actual_max_val < actual_min_val:
            print(
                f"Warning: max_value ({actual_max_val}) < min_value ({actual_min_val}). Setting max_value = min_value."
            )
            actual_max_val = actual_min_val

        bin_edges = np.linspace(actual_min_val, actual_max_val, num_hist_bins + 1)
        total_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
        fail_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
        converge_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
        fail_converge_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
        total_rows_processed = 0

        # --- Pre-compile Numba function with dummy data ---
        if verbose:
            print("Pre-compiling Numba histogram function...")
        try:
            dummy_values = np.array([0.5], dtype=np.float64)
            dummy_fail_mask = np.array([True], dtype=bool)
            dummy_converge_mask = np.array([True], dtype=bool)
            _calculate_histograms_numba(
                dummy_values,
                dummy_fail_mask,
                bin_edges,
                total_counts_hist.copy(),
                fail_counts_hist.copy(),
                dummy_converge_mask,
                converge_counts_hist.copy(),
                fail_converge_counts_hist.copy(),
            )
            if verbose:
                print("Numba function pre-compiled.")
        except Exception as e:
            print(
                f"Warning: Numba pre-compilation failed: {e}. Proceeding without pre-compilation."
            )
        # -------------------------------------------------

        # --- Benchmarking variables ---
        total_read_time = 0.0
        total_calc_value_time = 0.0  # Renamed from total_calc_frac_time
        total_hist_time = 0.0
        # ----------------------------

        # --- Process files iteratively (Optimized Read, Numba Histograms) ---
        if verbose:
            print(
                f"Processing {len(file_paths)} files iteratively (Numba histograms)..."
            )
        # Dynamically determine required columns for main processing pass
        # Start with columns needed for the 'by' logic, then add common ones.
        required_columns_for_read = list(actual_columns_to_read)  # Copy the list
        required_columns_for_read.extend(["fail", "converge"])
        required_columns_for_read = sorted(list(set(required_columns_for_read)))

        for i, file_path in tqdm(list(enumerate(file_paths))):
            try:
                start_time = time.perf_counter()
                df_single = pd.read_feather(
                    file_path, columns=required_columns_for_read
                )
                read_time = time.perf_counter() - start_time
                total_read_time += read_time

                if df_single.empty:
                    continue

                if not all(
                    col in df_single.columns for col in required_columns_for_read
                ):
                    continue

                current_rows = len(df_single)
                total_rows_processed += current_rows

                start_time = time.perf_counter()
                if by == "cluster_frac":
                    if num_errors == 0:  # num_errors is guaranteed to be non-None here
                        series_to_bin = pd.Series(dtype=float)
                    else:
                        series_to_bin = (
                            df_single[actual_columns_to_read[0]] / num_errors
                        )
                elif by == "cluster_frac_bp_llr":
                    numerator = df_single["cluster_bp_llr_sum"]
                    denominator = (
                        df_single["cluster_bp_llr_sum"]
                        + df_single["outside_cluster_bp_llr"]
                    )
                    series_to_bin = numerator / denominator.replace(
                        0, np.nan
                    )  # Avoid division by zero
                elif by == "cluster_frac_llr":
                    numerator = df_single["cluster_llr_sum"]
                    denominator = (
                        df_single["cluster_llr_sum"] + df_single["outside_cluster_llr"]
                    )
                    series_to_bin = numerator / denominator.replace(
                        0, np.nan
                    )  # Avoid division by zero
                elif by == "cluster_gap_bp_llr":
                    series_to_bin = (
                        df_single["outside_cluster_bp_llr"]
                        - df_single["cluster_bp_llr_sum"]
                    )
                elif by == "cluster_gap_llr":
                    series_to_bin = (
                        df_single["outside_cluster_llr"] - df_single["cluster_llr_sum"]
                    )
                else:  # Direct column name
                    series_to_bin = df_single[actual_columns_to_read[0]]

                series_to_bin_cleaned = series_to_bin.dropna()

                calc_value_time = time.perf_counter() - start_time  # Renamed
                total_calc_value_time += calc_value_time  # Renamed

                if (
                    min_value_override is not None
                    and max_value_override is not None
                    and not series_to_bin_cleaned.empty
                ):
                    values_np_check = series_to_bin_cleaned.to_numpy()
                    if np.any(values_np_check < min_value_override) or np.any(
                        values_np_check > max_value_override
                    ):
                        raise ValueError(
                            f"Data found outside user-specified value range "
                            f"[{min_value_override}, {max_value_override}]. "
                            f"Aggregation method: {by}, File: {os.path.basename(file_path)}"
                        )

                start_time = time.perf_counter()
                if not series_to_bin_cleaned.empty:
                    values_np = series_to_bin_cleaned.to_numpy()
                    fail_mask = (
                        df_single["fail"]
                        .reindex(series_to_bin_cleaned.index)
                        .to_numpy()
                    )
                    converge_mask = (
                        df_single["converge"]
                        .reindex(series_to_bin_cleaned.index)
                        .to_numpy()
                    )

                    (
                        total_counts_hist,
                        fail_counts_hist,
                        converge_counts_hist,
                        fail_converge_counts_hist,
                    ) = _calculate_histograms_numba(
                        values_np,
                        fail_mask,
                        bin_edges,
                        total_counts_hist,
                        fail_counts_hist,
                        converge_mask,
                        converge_counts_hist,
                        fail_converge_counts_hist,
                    )

                hist_time = time.perf_counter() - start_time
                total_hist_time += hist_time

                del df_single, series_to_bin
                if "series_to_bin_cleaned" in locals():
                    del series_to_bin_cleaned
                if "values_np" in locals():
                    del values_np
                if "fail_mask" in locals():
                    del fail_mask
                if "converge_mask" in locals():
                    del converge_mask

            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}")

        gc.collect()

        # print("--- Benchmarking Results ---")
        # print(f"Total time reading files: {total_read_time:.4f} seconds")
        # print(f"Total time calculating values: {total_calc_value_time:.4f} seconds") # Renamed
        # print(f"Total time calculating histograms: {total_hist_time:.4f} seconds")
        # print("----------------------------")

        if total_rows_processed == 0:
            print(
                f"Warning: Processed 0 valid rows for n={n}, T={T}, p={p} using aggregation method {by}. Skipping."
            )
            return pd.DataFrame(), 0

        # --- Construct df_agg with raw bin counts ---
        binned_value_column_name = (
            by  # The output column is named after the 'by' method
        )

        if ascending_confidence:
            binned_values_for_df = bin_edges[:-1]
        else:
            binned_values_for_df = bin_edges[1:]

        df_agg = pd.DataFrame(
            {
                binned_value_column_name: binned_values_for_df,
                "count": total_counts_hist,
                "num_fails": fail_counts_hist,
                "num_converged": converge_counts_hist,
                "num_converged_fails": fail_converge_counts_hist,
            }
        )

        df_agg = df_agg[df_agg["count"] > 0]
        df_agg = df_agg.sort_values(
            binned_value_column_name, ascending=True
        ).reset_index(drop=True)

        if verbose:
            print(
                f"  -> Generated df_agg with {len(df_agg)} rows from {total_rows_processed} total samples, using aggregation method '{by}'."
            )
        return df_agg, total_rows_processed

    except (FileNotFoundError, ValueError) as e:
        print(
            f"Info: Skipping n={n}, T={T}, p={p} due to data finding/validation issue: {e}"
        )
        return pd.DataFrame(), 0
    except Exception as e:
        print(f"Error processing n={n}, T={T}, p={p}: {e}. Skipping this combination.")
        return pd.DataFrame(), 0


def aggregate_data(
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
    by: str = "cluster_frac",
    n: int | None = None,
    T: int | None = None,
    p: float | None = None,
    num_hist_bins: int = 1000,
    value_range: tuple[float, float] | None = None,
    ascending_confidence: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Aggregate BB simulation data based on a specified column or cluster fraction.

    Parameters
    ----------
    data_dir : str
        Base directory path containing the `n{n}_T{T}_p{p}` subdirectories.
    by : str, optional
        Column to aggregate by. Defaults to "cluster_frac".
        If "cluster_frac", values from "cluster_size_sum" are read and
        normalized by `num_errors` by the underlying aggregation function.
        If "cluster_frac_bp_llr", the ratio cluster_bp_llr_sum / (cluster_bp_llr_sum + outside_cluster_bp_llr) is used.
        If "cluster_frac_llr", the ratio cluster_llr_sum / (cluster_llr_sum + outside_cluster_llr) is used.
        If "cluster_gap_bp_llr", the difference outside_cluster_bp_llr - cluster_bp_llr_sum is used.
        If "cluster_gap_llr", the difference outside_cluster_llr - cluster_llr_sum is used.
        Otherwise, `by` is treated as the direct column name to read and use raw values from.
    n : int, optional
        Specific number of qubits for the BB code. If None, scan for all n.
    T : int, optional
        Specific number of rounds. If None, scan for all T.
    p : float, optional
        Specific physical error rate. If None, scan for all p.
    num_hist_bins : int, optional
        Number of bins to use for the histogram. Defaults to 1000.
    value_range : tuple[float, float], optional
        User-specified minimum and maximum values for the histogram range.
        If None, it's auto-detected.
    ascending_confidence : bool, optional
        Indicates the relationship between the aggregated value and decoding confidence.
        Defaults to True.
    verbose : bool, optional
        Whether to print progress and informational messages. Defaults to False.

    Returns
    -------
    all_df_agg : pd.DataFrame
        Combined DataFrame with multi-index (n, T, p, <binned_value_column>) containing
        post-selection statistics and probabilities. Returns an empty DataFrame
        if no valid data is found or processed for the specified or scanned combinations.
    """
    if not os.path.isdir(data_dir):
        print(f"Error: Base data directory not found: {data_dir}")
        return pd.DataFrame()

    target_combinations: List[Tuple[int, int, float]] = []

    # --- Determine target combinations ---
    if n is not None and T is not None and p is not None:
        # Specific combination provided
        if verbose:
            print(f"Checking for specified combination: n={n}, T={T}, p={p}")
        try:
            # Check if files exist within the subdirectory for this specific combination
            find_bb_simulation_files(n=n, T=T, p=p, data_dir=data_dir, verbose=verbose)
            target_combinations = [(n, T, p)]
            if verbose:
                print(f"  -> Subdirectory and data files found.")
        except (FileNotFoundError, ValueError) as e:
            print(
                f"Info: Cannot proceed with specified combination n={n}, T={T}, p={p}: {e}"
            )
            return pd.DataFrame()
    else:
        # Scan directory for all relevant subdirectories
        if verbose:
            print(f"Scanning {data_dir} for `n*_T*_p*` subdirectories...")
        # Pattern to find potential subdirectories
        subdir_pattern = os.path.join(data_dir, "n*_T*_p*")
        potential_subdirs = glob.glob(subdir_pattern)

        if not potential_subdirs:
            print(f"Error: No subdirectories matching 'n*_T*_p*' found in {data_dir}")
            return pd.DataFrame()

        # Regex to parse n, T, p from directory names
        param_pattern = re.compile(r"n(\d+)_T(\d+)_p([\d\.]+)")
        unique_combinations: Set[Tuple[int, int, float]] = set()

        for path in potential_subdirs:
            if os.path.isdir(path):
                dirname = os.path.basename(path)
                match = param_pattern.match(dirname)  # Use match for start of string
                if match:
                    try:
                        scan_n = int(match.group(1))
                        scan_T = int(match.group(2))
                        scan_p = float(match.group(3))

                        # Filter based on any provided parameters (n, T, p)
                        if (
                            (n is None or scan_n == n)
                            and (T is None or scan_T == T)
                            and (p is None or np.isclose(scan_p, p))
                        ):
                            # Check if the directory actually contains data files
                            try:
                                find_bb_simulation_files(
                                    n=scan_n,
                                    T=scan_T,
                                    p=scan_p,
                                    data_dir=data_dir,
                                    verbose=False,
                                )
                                unique_combinations.add((scan_n, scan_T, scan_p))
                            except (FileNotFoundError, ValueError):
                                if verbose:
                                    print(
                                        f"  Skipping directory {dirname}: Does not contain valid data files or is empty."
                                    )
                                continue  # Skip if no files inside
                    except (ValueError, TypeError) as e:
                        print(
                            f"Warning: Could not parse parameters from directory name {dirname}: {e}. Skipping this directory."
                        )
                    except Exception as e:
                        print(
                            f"Warning: Unexpected error processing directory {dirname}: {e}. Skipping this directory."
                        )

        if not unique_combinations:
            print(
                "Error: No valid subdirectories with data found matching the criteria."
            )
            return pd.DataFrame()

        target_combinations = sorted(list(unique_combinations))
        if verbose:
            print(
                f"Found {len(target_combinations)} matching unique (n, T, p) combinations with data."
            )

    all_ps_dfs: List[pd.DataFrame] = []
    processed_combinations_count = 0
    min_val_override, max_val_override = (None, None)  # Default to None
    if value_range:
        if len(value_range) == 2:
            min_val_override, max_val_override = value_range
        else:
            print(
                "Warning: value_range must be a tuple of two floats (min, max). Ignoring provided value_range."
            )

    for comb_n, comb_T, comb_p in target_combinations:
        if verbose:
            print(f"\nProcessing combination: n={comb_n}, T={comb_T}, p={comb_p}")
        df_agg_single, total_samples = calculate_df_agg_for_combination(
            n=comb_n,
            T=comb_T,
            p=comb_p,
            data_dir=data_dir,
            num_hist_bins=num_hist_bins,
            min_value_override=min_val_override,
            max_value_override=max_val_override,
            ascending_confidence=ascending_confidence,
            by=by,
            verbose=verbose,
        )

        if not df_agg_single.empty and total_samples > 0:
            df_agg_single["n"] = comb_n
            df_agg_single["T"] = comb_T
            df_agg_single["p"] = comb_p
            all_ps_dfs.append(df_agg_single)
            processed_combinations_count += 1
            if verbose:
                print(
                    f"  -> Successfully processed combination, added {len(df_agg_single)} rows to results."
                )
        else:
            if verbose:
                print(
                    f"  -> No df_agg data generated or an error occurred for n={comb_n}, T={comb_T}, p={comb_p}."
                )

    if not all_ps_dfs:
        print(
            "\nError: No df_agg dataframes were successfully generated for any matching combination."
        )
        return pd.DataFrame()

    if verbose:
        print(
            f"\nConcatenating results for {processed_combinations_count} successful combinations..."
        )
    try:
        all_df_agg = pd.concat(all_ps_dfs, ignore_index=True)
    except Exception as e:
        print(f"Error during final concatenation: {e}")
        return pd.DataFrame()

    if verbose:
        print("Setting multi-index (n, T, p, <binned_value_column>)...")
    try:
        # Determine the name of the column used for binning for the index
        binned_value_idx_col_name = "cluster_frac" if by == "cluster_frac" else by

        required_cols = ["n", "T", "p", binned_value_idx_col_name]
        if not all(col in all_df_agg.columns for col in required_cols):
            missing = [col for col in required_cols if col not in all_df_agg.columns]
            print(f"Error: Cannot set multi-index. Missing columns: {missing}")
            print("Columns available:", all_df_agg.columns.tolist())
            print("Returning concatenated DataFrame without multi-index.")
            return all_df_agg

        all_df_agg = all_df_agg.set_index(required_cols)
        if verbose:
            print("Sorting index...")
        all_df_agg = all_df_agg.sort_index()
        if verbose:
            print("Concatenation and indexing complete.")
    except Exception as e:
        print(f"An unexpected error occurred during final indexing/sorting: {e}")
        print("Returning concatenated DataFrame without multi-index.")
        return all_df_agg

    return all_df_agg


def calculate_confidence_interval(n, k, alpha=0.05, method="wilson"):
    p_low, p_upp = proportion_confint(k, n, alpha=alpha, method=method)
    p = (p_low + p_upp) / 2
    delta_p = p_upp - p
    return p, delta_p


def get_df_ps(df_agg, ascending_confidence=True):
    if ascending_confidence:
        df_agg = df_agg.iloc[::-1]

    shots = df_agg["count"].sum()

    # Ignoring convergence

    counts, num_fails = (
        df_agg["count"].cumsum(),
        df_agg["num_fails"].cumsum(),
    )

    pfail, delta_pfail = calculate_confidence_interval(counts, num_fails)
    pacc, delta_pacc = calculate_confidence_interval(shots, counts)

    # Treating convergence = confident

    counts_conv = df_agg["count"] - df_agg["num_converged"]
    counts_conv.iloc[0] += df_agg["num_converged"].sum()
    counts_conv = counts_conv.cumsum()

    assert counts_conv.iloc[-1] == shots

    num_fails_conv = df_agg["num_fails"] - df_agg["num_converged_fails"]
    num_fails_conv.iloc[0] += df_agg["num_converged_fails"].sum()
    num_fails_conv = num_fails_conv.cumsum()

    pfail_conv, delta_pfail_conv = calculate_confidence_interval(
        counts_conv, num_fails_conv
    )
    pacc_conv, delta_pacc_conv = calculate_confidence_interval(shots, counts_conv)

    df_ps = pd.DataFrame(index=df_agg.index)

    df_ps["p_fail"] = pfail
    df_ps["delta_p_fail"] = delta_pfail
    df_ps["p_abort"] = 1 - pacc
    df_ps["delta_p_abort"] = delta_pacc

    df_ps["p_fail_conv"] = pfail_conv
    df_ps["delta_p_fail_conv"] = delta_pfail_conv
    df_ps["p_abort_conv"] = 1 - pacc_conv
    df_ps["delta_p_abort_conv"] = delta_pacc_conv

    df_ps["count"] = counts
    df_ps["num_fails"] = num_fails
    df_ps["count_conv"] = counts_conv
    df_ps["num_fails_conv"] = num_fails_conv

    return df_ps
