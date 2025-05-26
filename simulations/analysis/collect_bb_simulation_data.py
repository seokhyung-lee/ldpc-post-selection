import gc  # Add garbage collector import
import glob
import os
import re
import time  # Import time module
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.ldpc_post_selection.build_circuit import build_BB_circuit

# Import from the new utility file
from .simulation_data_utils import (
    _calculate_histograms_bplsd_numba,
    _get_values_for_binning_from_batch,
    natural_sort_key,
)


def find_bb_simulation_files(
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",  # Base directory
    verbose: bool = False,
) -> List[str]:
    """
    Find BB code simulation batch directories (`batch_{idx}`)
    within the specific subdirectory for an (n, T, p) tuple.
    Each valid batch directory must contain "scalars.feather", "cluster_sizes.npy",
    "cluster_llrs.npy", and "offsets.npy".

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
        Whether to print the number of found valid batch directories. Defaults to False.

    Returns
    -------
    batch_dir_paths : list of str
        List of paths to valid batch directories.

    Raises
    ------
    FileNotFoundError
        If the base data directory or the specific n{n}_T{T}_p{p} subdirectory does not exist.
    ValueError
        If no valid `batch_{idx}` directories (containing all required files)
        are found within the subdirectory.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Base data directory not found: {data_dir}")

    # Construct the specific subdirectory path
    sub_dirname = f"n{n}_T{T}_p{p}"
    sub_dir_path = os.path.join(data_dir, sub_dirname)

    if not os.path.isdir(sub_dir_path):
        raise FileNotFoundError(
            f"Subdirectory not found for n={n}, T={T}, p={p}: {sub_dir_path}"
        )

    # Search for batch_* subdirectories within the n_T_p subdirectory
    batch_dir_pattern = os.path.join(sub_dir_path, "batch_*")
    potential_batch_dirs = glob.glob(batch_dir_pattern)
    potential_batch_dirs = [d for d in potential_batch_dirs if os.path.isdir(d)]

    if not potential_batch_dirs:
        raise ValueError(f"No batch directories (batch_*) found in {sub_dir_path}.")

    valid_batch_dir_paths = []
    required_files = [
        "scalars.feather",
        "cluster_sizes.npy",
        "cluster_llrs.npy",
        "offsets.npy",
    ]

    for batch_dir in potential_batch_dirs:
        is_complete = True
        for req_file in required_files:
            if not os.path.isfile(os.path.join(batch_dir, req_file)):
                is_complete = False
                if verbose:
                    print(
                        f"  Skipping batch directory {os.path.basename(batch_dir)}: missing {req_file}"
                    )
                break
        if is_complete:
            valid_batch_dir_paths.append(batch_dir)

    if not valid_batch_dir_paths:
        raise ValueError(
            f"No valid batch directories (containing all required files) found in {sub_dir_path}."
        )

    # Sort batch directories naturally using the utility function
    valid_batch_dir_paths.sort(key=natural_sort_key)

    if verbose:
        print(
            f"Found {len(valid_batch_dir_paths)} valid batch directories in {sub_dirname}."
        )

    return valid_batch_dir_paths


def calculate_df_agg_for_combination(
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
    num_hist_bins: int = 1000,
    min_value_override: float | None = None,
    max_value_override: float | None = None,
    ascending_confidence: bool = True,
    by: str = "pred_llr",
    norm_order: float | None = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Calculate the post-selection DataFrame (df_agg) for a single (n, T, p) combination.
    Reads data from batch directories, each containing 'scalars.feather',
    'cluster_sizes.npy', 'cluster_llrs.npy', and 'offsets.npy'.
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
        Directory path containing the simulation data (n_T_p subdirectories).
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
    norm_order : float, optional
        The order for L_p norm calculation when `by` is one of the norm-based methods.
        Must be a positive float (can be np.inf).
        Required if `by` is one of the norm or norm-gap methods.
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
    supported_by_methods = [
        "pred_llr",
        "detector_density",
        "cluster_size_norm",
        "cluster_llr_norm",
        "cluster_size_norm_gap",
        "cluster_llr_norm_gap",
    ]
    if by not in supported_by_methods:
        raise ValueError(
            f"Unsupported 'by' method: {by}. Supported methods are: {supported_by_methods}"
        )

    norm_based_methods = [
        "cluster_size_norm",
        "cluster_llr_norm",
        "cluster_size_norm_gap",
        "cluster_llr_norm_gap",
    ]
    if by in norm_based_methods and (norm_order is None or norm_order <= 0):
        raise ValueError(
            f"'norm_order' must be a positive float when 'by' is '{by}'. Got: {norm_order}"
        )

    batch_dir_paths = find_bb_simulation_files(
        n=n, T=T, p=p, data_dir=data_dir, verbose=verbose
    )
    # find_bb_simulation_files raises error if no dirs, so this check is mostly a safeguard.
    if not batch_dir_paths:
        print(
            f"Warning: No valid batch directories found for n={n}, T={T}, p={p} by find_bb_simulation_files. Skipping."
        )
        return pd.DataFrame(), 0

    # --- Determine value range (min_val, max_val) --- (First Pass)
    actual_min_val: float
    actual_max_val: float

    if min_value_override is not None and max_value_override is not None:
        if min_value_override >= max_value_override:
            raise ValueError("min_value_override must be less than max_value_override.")
        actual_min_val = min_value_override
        actual_max_val = max_value_override
        if verbose:
            print(
                f"  Using user-specified value range for {by}: [{actual_min_val}, {actual_max_val}]"
            )
    else:
        if verbose:
            print(
                f"  Auto-detecting value range for {by} (first pass over batch directories)..."
            )
        current_min_val = np.inf
        current_max_val = -np.inf
        found_any_valid_value_for_range = False

        for batch_dir_pass1 in tqdm(batch_dir_paths, desc="Range detection"):
            # For range detection, we don't need to keep df_scalars_loaded
            temp_series_to_bin, _ = _get_values_for_binning_from_batch(
                batch_dir_path=batch_dir_pass1,
                by=by,
                norm_order=norm_order,
                verbose=verbose
                > 1,  # More verbose logging inside helper for pass 1 if main verbose is high
            )

            if temp_series_to_bin is not None and not temp_series_to_bin.empty:
                temp_series_to_bin_cleaned = temp_series_to_bin.dropna()
                if not temp_series_to_bin_cleaned.empty:
                    found_any_valid_value_for_range = True
                    current_min_val = min(
                        current_min_val, temp_series_to_bin_cleaned.min()
                    )
                    current_max_val = max(
                        current_max_val, temp_series_to_bin_cleaned.max()
                    )
                del temp_series_to_bin_cleaned
            if temp_series_to_bin is not None:
                del temp_series_to_bin

        if not found_any_valid_value_for_range:
            print(
                f"Warning: No valid data found for {by} for n={n}, T={T}, p={p} during range detection. Skipping."
            )
            return pd.DataFrame(), 0

        actual_min_val = current_min_val
        actual_max_val = current_max_val

        if np.isclose(actual_min_val, actual_max_val):
            adjustment = (
                max(1.0, abs(actual_min_val * 0.1)) if actual_min_val != 0 else 1.0
            )
            actual_max_val = actual_min_val + adjustment
            actual_min_val = actual_min_val - adjustment
            if actual_max_val == actual_min_val and num_hist_bins > 1:
                actual_max_val = actual_min_val + (num_hist_bins * np.finfo(float).eps)
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
    # --- End Determine value range ---

    if not isinstance(num_hist_bins, int) or num_hist_bins < 1:
        raise ValueError("num_hist_bins must be a positive integer.")

    if actual_max_val < actual_min_val:
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
    if not np.all(np.diff(bin_edges) >= 0) and len(bin_edges) > 1:
        if num_hist_bins == 1:  # If only one bin, [val, val] is fine for edges
            bin_edges = np.array([actual_min_val, actual_max_val])
            if bin_edges[0] > bin_edges[1]:
                bin_edges[1] = bin_edges[0]  # ensure min <= max
        else:
            raise ValueError(
                f"  Error: Could not create monotonic bins for num_hist_bins={num_hist_bins} with range [{actual_min_val}, {actual_max_val}]. Edges: {bin_edges}. Returning empty."
            )

    total_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    fail_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    converge_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    fail_converge_counts_hist = np.zeros(num_hist_bins, dtype=np.int64)
    total_rows_processed = 0  # Counts actual binned entries
    total_samples_considered = (
        0  # Counts samples from scalars.feather that were processed
    )

    total_read_time = 0.0
    total_calc_value_time = 0.0
    total_hist_time = 0.0

    if verbose:
        print(
            f"Processing {len(batch_dir_paths)} batch directories iteratively (Numba histograms)..."
        )

    for batch_dir in tqdm(batch_dir_paths, desc="Aggregation"):
        start_time_read = (
            time.perf_counter()
        )  # Time only for this specific batch read + initial processing

        # In main pass, df_scalars is loaded by/returned from the helper
        series_to_bin, df_scalars = _get_values_for_binning_from_batch(
            batch_dir_path=batch_dir,
            by=by,
            norm_order=norm_order,
            verbose=verbose > 1,
        )
        read_and_initial_calc_time = time.perf_counter() - start_time_read
        total_read_time += read_and_initial_calc_time  # This now includes some of the calc time from helper
        # For simplicity, not splitting out precisely here.

        if series_to_bin is None or df_scalars is None or df_scalars.empty:
            if verbose:
                print(
                    f"  Skipping batch {os.path.basename(batch_dir)} due to issues from _get_values_for_binning (series or scalars missing/empty)."
                )
            continue

        total_samples_considered += len(df_scalars)

        required_scalar_cols = ["fail", "converge", "fail_bp"]
        if not all(col in df_scalars.columns for col in required_scalar_cols):
            if verbose:
                print(
                    f"  Skipping {os.path.basename(batch_dir)} due to missing fail/converge/fail_bp columns in scalars.feather."
                )
            continue

        start_time_calc_downstream = time.perf_counter()
        series_to_bin_cleaned = series_to_bin.dropna()
        calc_value_time_batch_downstream = (
            time.perf_counter() - start_time_calc_downstream
        )
        total_calc_value_time += calc_value_time_batch_downstream  # Time for dropna and any subsequent ops on series

        if series_to_bin_cleaned.empty:
            if verbose:
                print(
                    f"  No valid (non-NaN) data to bin for 'by={by}' in {os.path.basename(batch_dir)} after dropna. Skipping."
                )
            continue

        total_rows_processed += len(
            series_to_bin_cleaned
        )  # Count actual entries that will be binned

        if min_value_override is not None and max_value_override is not None:
            values_np_check = series_to_bin_cleaned.to_numpy()
            if np.any(values_np_check < min_value_override) or np.any(
                values_np_check > max_value_override
            ):
                # Error if values are outside user-defined hard boundaries
                raise ValueError(
                    f"Data found outside user-specified value range "
                    f"[{min_value_override}, {max_value_override}]. "
                    f"Aggregation method: {by}, Batch: {os.path.basename(batch_dir)}"
                )

        start_time_hist = time.perf_counter()
        values_np = series_to_bin_cleaned.to_numpy()
        # Align masks with the cleaned series index
        fail_mask = df_scalars.loc[series_to_bin_cleaned.index, "fail"].to_numpy(
            dtype=bool
        )
        converge_mask = df_scalars.loc[
            series_to_bin_cleaned.index, "converge"
        ].to_numpy(dtype=bool)
        fail_bp_mask = df_scalars.loc[series_to_bin_cleaned.index, "fail_bp"].to_numpy(
            dtype=bool
        )

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

        hist_time_batch = time.perf_counter() - start_time_hist
        total_hist_time += hist_time_batch

        del (
            series_to_bin,
            df_scalars,
            series_to_bin_cleaned,
            values_np,
            fail_mask,
            converge_mask,
            fail_bp_mask,
        )

        # Optional: if a batch error occurs, decide if you want to continue or stop
        # if batch_processing_error: continue # or break, or raise

    gc.collect()

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

    if total_rows_processed == 0:
        # This means no data points were actually binned across all batches.
        print(
            f"Warning: Processed 0 valid entries to bin for n={n}, T={T}, p={p} using aggregation method {by}. Output df_agg will be empty."
        )
        # Do not return early if total_samples_considered > 0 but total_rows_processed == 0.
        # An empty df_agg is a valid result if no data fell into bins or all was NaN.
        # Only return early if find_bb_simulation_files found nothing or range detection failed critically.

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
                binned_values_for_df = np.array(
                    [bin_edges[len(bin_edges) - 1]]
                )  # Use actual last edge for 1 bin
        else:
            print(
                f"Fatal Error: Mismatch between binned_values_for_df (len {len(binned_values_for_df)}) and num_hist_bins ({num_hist_bins}). Cannot construct df_agg."
            )
            return pd.DataFrame(), total_rows_processed

    df_agg = pd.DataFrame(
        {
            binned_value_column_name: binned_values_for_df,
            "count": total_counts_hist,
            "num_fails": fail_counts_hist,
            "num_converged": converge_counts_hist,
            "num_converged_fails": fail_converge_counts_hist,
        }
    )

    df_agg = df_agg[df_agg["count"] > 0].copy()
    if not df_agg.empty:
        df_agg.sort_values(binned_value_column_name, ascending=True, inplace=True)
        df_agg.reset_index(drop=True, inplace=True)

    if verbose:
        print(
            f"  -> Generated df_agg with {len(df_agg)} rows from {total_rows_processed} total valid binned entries ({total_samples_considered} samples initially considered), using method '{by}'."
        )
    return df_agg, total_rows_processed


def aggregate_data(
    by: str,
    *,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
    n: int | None = None,
    T: int | None = None,
    p: float | None = None,
    num_hist_bins: int = 1000,
    value_range: tuple[float, float] | None = None,
    ascending_confidence: bool = True,
    norm_order: float | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Aggregate BB simulation data based on specified metrics from batch directories.

    Parameters
    ----------
    by : str
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
    data_dir : str
        Base directory path containing the `n{n}_T{T}_p{p}` subdirectories.
    n : int, optional
        Specific number of qubits for the BB code. If None, scan for all n.
    T : int, optional
        Specific number of rounds. If None, scan for all T.
    p : float, optional
        Specific physical error rate. If None, scan for all p.
    num_hist_bins : int, optional
        Number of bins to use for the histogram. Defaults to 1000.
    value_range : tuple[float, float], optional
        User-specified minimum and maximum values for the histogram range ([min, max]).
        If None, it's auto-detected across all relevant data for the chosen `by` method.
    ascending_confidence : bool, optional
        Indicates the relationship between the aggregated value and decoding confidence.
        Defaults to True.
    norm_order : float, optional
        The order for L_p norm calculation when `by` is one of the norm or norm-gap methods.
        Must be a positive float. Required if `by` is one of the norm-based methods.
    verbose : bool, optional
        Whether to print progress and informational messages. Defaults to False.

    Returns
    -------
    all_df_agg : pd.DataFrame
        Combined DataFrame with multi-index (n, T, p, <by_column_name>) containing
        post-selection statistics. Returns an empty DataFrame if no valid data is
        found or processed for the specified or scanned combinations.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Error: Base data directory not found: {data_dir}")

    supported_by_methods = [
        "pred_llr",
        "detector_density",
        "cluster_size_norm",
        "cluster_llr_norm",
        "cluster_size_norm_gap",
        "cluster_llr_norm_gap",
    ]
    if by not in supported_by_methods:
        raise ValueError(
            f"Error: Unsupported 'by' method: {by}. Supported methods are: {supported_by_methods}"
        )

    norm_based_methods = [
        "cluster_size_norm",
        "cluster_llr_norm",
        "cluster_size_norm_gap",
        "cluster_llr_norm_gap",
    ]
    if by in norm_based_methods and (norm_order is None or norm_order <= 0):
        raise ValueError(
            f"Error: 'norm_order' must be a positive float when 'by' is '{by}'. Got: {norm_order}"
        )

    target_combinations: List[Tuple[int, int, float]] = []

    # --- Determine target combinations ---
    if n is not None and T is not None and p is not None:
        if verbose:
            print(f"Checking for specified combination: n={n}, T={T}, p={p}")
        try:
            find_bb_simulation_files(
                n=n, T=T, p=p, data_dir=data_dir, verbose=False
            )  # Just check existence
            target_combinations = [(n, T, p)]
            if verbose:
                print(
                    f"  -> Directory structure for specified combination seems to exist."
                )
        except (FileNotFoundError, ValueError) as e:
            print(
                f"Info: Cannot proceed with specified combination n={n}, T={T}, p={p}: {e}"
            )
            return pd.DataFrame()
    else:
        if verbose:
            print(
                f"Scanning {data_dir} for `n*_T*_p*` subdirectories with valid batch data..."
            )
        subdir_pattern = os.path.join(data_dir, "n*_T*_p*")
        potential_subdirs = glob.glob(subdir_pattern)

        if not potential_subdirs:
            raise FileNotFoundError(
                f"Error: No subdirectories matching 'n*_T*_p*' found in {data_dir}"
            )

        param_pattern = re.compile(r"n(\d+)_T(\d+)_p([\d\.]+)")
        unique_combinations: Set[Tuple[int, int, float]] = set()

        for path in potential_subdirs:
            if os.path.isdir(path):
                dirname = os.path.basename(path)
                match = param_pattern.match(dirname)
                if match:
                    try:
                        scan_n = int(match.group(1))
                        scan_T = int(match.group(2))
                        scan_p = float(match.group(3))

                        if (
                            (n is None or scan_n == n)
                            and (T is None or scan_T == T)
                            and (p is None or np.isclose(scan_p, p))
                        ):
                            try:
                                # Check if the directory actually contains valid batch subdirectories
                                find_bb_simulation_files(
                                    n=scan_n,
                                    T=scan_T,
                                    p=scan_p,
                                    data_dir=data_dir,
                                    verbose=False,  # Keep this concise for scanning phase
                                )
                                unique_combinations.add((scan_n, scan_T, scan_p))
                            except (FileNotFoundError, ValueError):
                                if verbose > 1:  # More detailed verbosity for this skip
                                    print(
                                        f"  Skipping directory {dirname}: Does not contain valid batch subdirectories or required files."
                                    )
                                continue
                    except (ValueError, TypeError) as e_parse:
                        if verbose:
                            print(
                                f"Warning: Could not parse parameters from directory name {dirname}: {e_parse}. Skipping."
                            )
                    except Exception as e_scan_dir:
                        if verbose:
                            print(
                                f"Warning: Unexpected error processing directory {dirname}: {e_scan_dir}. Skipping."
                            )

        if not unique_combinations:
            raise ValueError(
                "Error: No valid subdirectories with data found matching the criteria."
            )

        target_combinations = sorted(list(unique_combinations))
        if verbose:
            print(
                f"Found {len(target_combinations)} matching unique (n, T, p) combinations with data structures."
            )
    # --- End Determine target combinations ---

    all_ps_dfs: List[pd.DataFrame] = []
    processed_combinations_count = 0
    min_val_override, max_val_override = (None, None)
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
            min_val_override, max_val_override = None, None

    for comb_n, comb_T, comb_p in target_combinations:
        if verbose:
            print(
                f"\nProcessing combination: n={comb_n}, T={comb_T}, p={comb_p}, by='{by}', norm_order={norm_order if by in ['cluster_size_norm', 'cluster_llr_norm'] else 'N/A'}"
            )
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
            norm_order=norm_order,
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
                    f"  -> No df_agg data generated or an error occurred for n={comb_n}, T={comb_T}, p={comb_p}, by='{by}'."
                )

    if not all_ps_dfs:
        print(
            "\nInfo: No df_agg dataframes were successfully generated for any matching combination."
        )
        return pd.DataFrame()

    if verbose:
        print(
            f"\nConcatenating results for {processed_combinations_count} successful combinations..."
        )

    all_df_agg = pd.concat(all_ps_dfs, ignore_index=True)

    if all_df_agg.empty:
        if verbose:
            print("Concatenated DataFrame is empty. No data to index.")
        return pd.DataFrame()  # Return empty if concatenation somehow results in empty

    if verbose:
        print(f"Setting multi-index (n, T, p, {by})...")

    # The binned value column is now directly named by the 'by' parameter.
    binned_value_idx_col_name = by

    required_cols = ["n", "T", "p", binned_value_idx_col_name]
    if not all(col in all_df_agg.columns for col in required_cols):
        missing = [col for col in required_cols if col not in all_df_agg.columns]
        raise ValueError(
            f"Error: Cannot set multi-index. Missing columns: {missing}. Available columns: {all_df_agg.columns.tolist()}"
        )

    all_df_agg = all_df_agg.set_index(required_cols).sort_index()
    if verbose:
        print("Concatenation and indexing complete.")

    return all_df_agg


if __name__ == "__main__":
    ascending_confidences = {
        #     "pred_llr": False,
        #     "detector_density": False,
        "cluster_size_norm": False,
        #     "cluster_size_norm_gap": True,
        #     "cluster_llr_norm": False,
        #     "cluster_llr_norm_gap": True,
    }

    # orders = [0.5, 1, 2, np.inf]
    orders = [2]

    n = 72
    p = 0.001

    df_agg_dict = {}
    for by, ascending_confidence in ascending_confidences.items():
        print(
            f"\nAggregating data for {by} with ascending_confidence={ascending_confidence}..."
        )

        norm_orders = orders if "norm" in by else [None]

        for order in norm_orders:
            if order is not None:
                print(f"norm_order = {order}")
                key = f"{by}_{order}"
            else:
                key = by

            df_agg = aggregate_data(
                by=by,
                n=n,
                p=p,
                norm_order=order,
                num_hist_bins=10000,
                ascending_confidence=ascending_confidence,
                data_dir="../data/bb_minsum_iter30_lsd0",
                verbose=False,
            )
            df_agg_dict[key] = (df_agg, ascending_confidence)

        print("=============")
