import gc  # Add garbage collector import
import glob
import os
import re
import time  # Import time module
from typing import List, Set, Tuple

import numba  # Import Numba
import numpy as np
import pandas as pd
from numba import prange
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from src.ldpc_post_selection.build_circuit import build_BB_circuit, get_BB_distance


def find_bb_simulation_files(  # Rename function
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
) -> List[str]:  # Return List[str] instead of pd.DataFrame
    """
    Find BB code circuit simulation data feather files for a specific (n, T, p) tuple.

    Parameters
    ----------
    n : int
        Number of qubits in the BB code
    T : int
        Number of rounds
    p : float
        Physical error rate
    data_dir : str
        Directory path containing the simulation data files.

    Returns
    -------
    file_paths : list of str
        List of file paths matching the specified (n, T, p)

    Raise
    ------
    FileNotFoundError
        If the data directory does not exist.
    ValueError
        If no data files arg found for the specified (n, T, p).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_paths = []
    processed_filenames = set()

    # --- Try pattern with T ---
    p_str = f"{p:.{len(str(p).split('.')[-1])}f}".rstrip("0").rstrip(
        "."
    )  # Adjust precision as needed
    pattern_with_T = f"n{n}_T{T}_p{p_str}_*.feather"
    found_files = glob.glob(os.path.join(data_dir, pattern_with_T))
    for f in found_files:
        fname = os.path.basename(f)
        if fname not in processed_filenames:
            file_paths.append(f)
            processed_filenames.add(fname)

    # --- Fallback: less precise glob with T and filter ---
    if not file_paths:
        fallback_pattern_with_T = f"n{n}_T{T}_p*.feather"
        potential_files = glob.glob(os.path.join(data_dir, fallback_pattern_with_T))
        p_pattern_regex = re.compile(rf"n{n}_T{T}_p([\d\.]+)_.*\.feather")
        for fpath in potential_files:
            fname = os.path.basename(fpath)
            if fname in processed_filenames:
                continue
            match = p_pattern_regex.search(fname)
            if match:
                try:
                    file_p = float(match.group(1))
                    if np.isclose(file_p, p):
                        file_paths.append(fpath)
                        processed_filenames.add(fname)
                except ValueError:
                    continue

    # --- Fallback: Try pattern without T (assuming T=get_BB_distance(n)) ---
    if not file_paths:
        expected_T = get_BB_distance(n)
        if expected_T == T:  # Only proceed if the calculated T matches the input T
            pattern_no_T = f"n{n}_p{p_str}_*.feather"
            found_files_no_T = glob.glob(os.path.join(data_dir, pattern_no_T))
            for f in found_files_no_T:
                fname = os.path.basename(f)
                if fname not in processed_filenames:
                    file_paths.append(f)
                    processed_filenames.add(fname)

            # --- Fallback: less precise glob without T and filter ---
            if not file_paths:
                fallback_pattern_no_T = f"n{n}_p*.feather"
                potential_files_no_T = glob.glob(
                    os.path.join(data_dir, fallback_pattern_no_T)
                )
                p_pattern_regex_no_T = re.compile(rf"n{n}_p([\d\.]+)_.*\.feather")
                for fpath in potential_files_no_T:
                    fname = os.path.basename(fpath)
                    if fname in processed_filenames:
                        continue
                    match = p_pattern_regex_no_T.search(fname)
                    if match:
                        try:
                            file_p = float(match.group(1))
                            if np.isclose(file_p, p):
                                file_paths.append(fpath)
                                processed_filenames.add(fname)
                        except ValueError:
                            continue

    if not file_paths:
        raise ValueError(f"No data files found for n={n}, T={T}, p={p} in {data_dir}.")

    print(
        f"Found {len(file_paths)} data files for n={n}, T={T}, p={p}."
    )  # Modify print message

    return file_paths  # Return the list of paths


# --- Numba JIT function for histogram calculation ---
@numba.jit(nopython=True, fastmath=True)  # Use nopython=True for best performance
def _calculate_histograms_numba(
    cluster_frac_np: np.ndarray,  # Expect float array
    fail_mask_np: np.ndarray,  # Expect boolean array
    bin_edges: np.ndarray,  # Expect float array
    total_counts_hist: np.ndarray,  # Expect int64 array
    fail_counts_hist: np.ndarray,  # Expect int64 array
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate total and fail histograms using Numba with an optimized single loop.

    Parameters
    ----------
    cluster_frac_np : 1D numpy array of float
        Valid cluster fractions (NaNs should be removed before calling).
    fail_mask_np : 1D numpy array of bool
        Indicates failures, aligned with cluster_frac_np.
    bin_edges : 1D numpy array of float
        Defines the histogram bin edges, must be sorted.
    total_counts_hist : 1D numpy array of int64
        Histogram counts for all samples (updated in-place).
    fail_counts_hist : 1D numpy array of int64
        Histogram counts for failed samples (updated in-place).

    Returns
    -------
    total_counts_hist : 1D numpy array of int64
        The updated histogram counts for all samples.
    fail_counts_hist : 1D numpy array of int64
        The updated histogram counts for failed samples.

    Notes
    -----
    This implementation manually calculates the histogram bins in a single pass
    for potentially better performance within Numba compared to calling
    np.histogram multiple times. It assumes NaNs have been filtered from
    cluster_frac_np before calling. It replicates np.histogram's behavior
    regarding bin edges: bins are [left, right), except the last bin which
    is [left, right].
    """
    n_bins = len(bin_edges) - 1
    if n_bins <= 0:  # Handle empty or invalid bin_edges
        return total_counts_hist, fail_counts_hist

    for i in range(len(cluster_frac_np)):
        val = cluster_frac_np[i]

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

    return total_counts_hist, fail_counts_hist


def calculate_df_ps_for_combination(
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
    cluster_frac_precision: int = 3,
) -> Tuple[pd.DataFrame, int]:
    """
    Calculate the post-selection DataFrame (df_ps) for a single (n, T, p) combination.
    Uses Numba JIT for histogram calculation.
    Includes timing for benchmarking loop operations.

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
    cluster_frac_precision : int, optional
        Number of decimal places to consider for the cluster fraction histogram bins.
        Defaults to 3.

    Returns
    -------
    df_ps : pd.DataFrame
        DataFrame with columns ['c', 'num_accs', 'num_fails']. Empty if fails.
    total_rows_processed : int
        Total number of simulation rows processed.
    """
    try:
        file_paths = find_bb_simulation_files(n=n, T=T, p=p, data_dir=data_dir)
        if not file_paths:
            print(f"Warning: No files found for n={n}, T={T}, p={p}. Skipping.")
            return pd.DataFrame(), 0

        # --- Get num_errors once ---
        try:
            circuit = build_BB_circuit(n=n, T=T, p=p)
            detector_error_model = circuit.detector_error_model()
            if detector_error_model is None:
                raise ValueError("Could not get detector_error_model.")
            num_errors = detector_error_model.num_errors
            if num_errors is None or num_errors <= 0:
                raise ValueError(f"Invalid num_errors ({num_errors}).")
        except Exception as e:
            print(
                f"Warning: Error getting num_errors for n={n}, T={T}, p={p}: {e}. Skipping."
            )
            return pd.DataFrame(), 0

        # --- Initialize Histogram approach ---
        if not isinstance(cluster_frac_precision, int) or cluster_frac_precision < 1:
            raise ValueError("cluster_frac_precision must be a positive integer.")
        num_bins = 10**cluster_frac_precision + 1
        # Revert bin_edges to default dtype (float64, Numba handles various float types)
        bin_edges = np.linspace(0, 1, num_bins)
        total_counts_hist = np.zeros(num_bins - 1, dtype=np.int64)
        fail_counts_hist = np.zeros(num_bins - 1, dtype=np.int64)
        total_rows_processed = 0

        # --- Pre-compile Numba function with dummy data ---
        print("Pre-compiling Numba histogram function...")
        try:
            # Create dummy arrays with expected dtypes
            dummy_cluster_frac = np.array([0.5], dtype=np.float64)
            dummy_fail_mask = np.array([True], dtype=bool)
            # Use the actual bin_edges array as its type is known
            _calculate_histograms_numba(
                dummy_cluster_frac,
                dummy_fail_mask,
                bin_edges,
                total_counts_hist,
                fail_counts_hist,
            )
            print("Numba function pre-compiled.")
        except Exception as e:
            print(
                f"Warning: Numba pre-compilation failed: {e}. Proceeding without pre-compilation."
            )
        # -------------------------------------------------

        # --- Benchmarking variables ---
        total_read_time = 0.0
        total_calc_frac_time = 0.0
        total_hist_time = 0.0
        # ----------------------------

        # --- Process files iteratively (Optimized Read, Numba Histograms) ---
        print(f"Processing {len(file_paths)} files iteratively (Numba histograms)...")
        required_columns = ["cluster_size_sum", "fail"]

        for i, file_path in tqdm(list(enumerate(file_paths))):
            try:
                # --- Time File Reading ---
                start_time = time.perf_counter()
                # Read only the required columns
                df_single = pd.read_feather(file_path, columns=required_columns)
                read_time = time.perf_counter() - start_time
                total_read_time += read_time
                # -------------------------

                if df_single.empty:
                    # No processing needed for empty files, skip timing other parts
                    # print(f"  Skipping empty file (or file without required columns): {os.path.basename(file_path)}") # Optional: Keep logging if needed
                    continue

                # Check if required columns were actually loaded (in case file didn't have them)
                if not all(col in df_single.columns for col in required_columns):
                    # print(f"  Skipping file missing required columns ({required_columns}): {os.path.basename(file_path)}") # Optional: Keep logging if needed
                    continue

                current_rows = len(df_single)
                total_rows_processed += current_rows

                # --- Time Cluster Fraction Calculation ---
                start_time = time.perf_counter()
                if num_errors == 0:
                    # Keep original dtype for cluster_frac
                    cluster_frac = pd.Series(dtype=float)  # Use default float
                else:
                    # Calculate using original types
                    cluster_frac = df_single["cluster_size_sum"] / num_errors

                # Keep original dropna method
                cluster_frac = cluster_frac.dropna()

                calc_frac_time = time.perf_counter() - start_time
                total_calc_frac_time += calc_frac_time
                # ---------------------------------------

                # --- Time Histogram Calculation (Using Numba JIT function) ---
                start_time = time.perf_counter()
                if not cluster_frac.empty:
                    # Prepare data for Numba: Convert to NumPy and ensure no NaNs
                    # cluster_frac already has NaNs dropped
                    cluster_frac_np = cluster_frac.to_numpy()

                    # Get fail mask corresponding to the non-NaN cluster_frac indices
                    fail_mask = df_single["fail"].reindex(cluster_frac.index).to_numpy()

                    # Call Numba function
                    total_counts_hist, fail_counts_hist = _calculate_histograms_numba(
                        cluster_frac_np,
                        fail_mask,
                        bin_edges,
                        total_counts_hist,
                        fail_counts_hist,
                    )

                hist_time = time.perf_counter() - start_time
                total_hist_time += hist_time
                # -----------------------------------------------------------

                # --- Memory Management ---
                del df_single, cluster_frac  # Keep simple del
                # Delete numpy arrays if they were created
                if "cluster_frac_np" in locals():
                    del cluster_frac_np
                if "fail_mask" in locals():
                    del fail_mask  # fail_mask is now numpy array
                # Optional: Collect garbage periodically if needed
                # if (i + 1) % 50 == 0: # Adjust frequency as needed
                #     gc.collect()
                #     # print(f"  Processed {i+1}/{len(file_paths)} files.") # Optional: Keep logging if needed

            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}")
                # Decide whether to continue or abort for the combination
                # continue # Continue with next file

        gc.collect()  # Final garbage collection for this combination

        # --- Print Benchmarking Results ---
        # print("--- Benchmarking Results ---")
        # print(f"Total time reading files: {total_read_time:.4f} seconds")
        # print(f"Total time calculating fractions: {total_calc_frac_time:.4f} seconds")
        # print(f"Total time calculating histograms: {total_hist_time:.4f} seconds")
        # print("----------------------------")
        # ----------------------------------

        if total_rows_processed == 0:
            print(f"Warning: Processed 0 valid rows for n={n}, T={T}, p={p}. Skipping.")
            return pd.DataFrame(), 0

        # --- Calculate cumulative counts and thresholds ---
        cum_acc_counts = np.cumsum(total_counts_hist)
        cum_fail_counts = np.cumsum(fail_counts_hist)
        # Use default bin_edges for thresholds
        thr = bin_edges[1:]

        # --- Construct df_ps ---
        df_ps = pd.DataFrame(
            {
                "c": 1 - thr,
                "num_accs": cum_acc_counts,
                "num_fails": cum_fail_counts,
            }
        )

        df_ps = df_ps[df_ps["num_accs"] > 0]
        df_ps = df_ps.sort_values("c", ascending=False).reset_index(drop=True)

        print(
            f"  → Generated df_ps with {len(df_ps)} rows from {total_rows_processed} total samples."
        )
        return df_ps, total_rows_processed

    except (FileNotFoundError, ValueError) as e:
        print(
            f"Info: Skipping n={n}, T={T}, p={p} due to data finding/validation issue: {e}"
        )
        return pd.DataFrame(), 0
    except Exception as e:
        print(f"Error processing n={n}, T={T}, p={p}: {e}. Skipping this combination.")
        return pd.DataFrame(), 0


def _calculate_p_confint(
    num_trials: int | pd.Series | np.ndarray,  # Allow ndarray
    num_successes: int | pd.Series | np.ndarray,  # Allow ndarray
    alpha: float = 0.05,  # Corresponds to 95% confidence interval
) -> Tuple[
    float | pd.Series | np.ndarray, float | pd.Series | np.ndarray
]:  # Update return possibilities
    """
    Calculate proportion and confidence interval half-width using Wilson score interval.

    Parameters
    ----------
    num_trials : int or pd.Series or np.ndarray
        Number of trials.
    num_successes : int or pd.Series or np.ndarray
        Number of successes.
    alpha : float, optional
        Significance level (default is 0.05 for 95% confidence).

    Returns
    -------
    p : float or pd.Series or np.ndarray
        Estimated proportion of successes.
    delta_p : float or pd.Series or np.ndarray
        Half-width of the confidence interval.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    k = np.asarray(num_successes)
    n = np.asarray(num_trials)

    # Determine the broadcast shape
    try:
        broadcast_shape = np.broadcast(n, k).shape
    except ValueError:
        # Handle cases where shapes are incompatible for broadcasting
        raise ValueError(
            f"Input shapes {n.shape} and {k.shape} are not compatible for broadcasting."
        )

    # Initialize output arrays with NaNs, matching the broadcast shape
    p_low = np.full(broadcast_shape, np.nan, dtype=float)
    p_upp = np.full(broadcast_shape, np.nan, dtype=float)

    # Create broadcasted versions for mask calculation (read-only views)
    # Use np.broadcast_to for explicit broadcasting if needed, or rely on implicit broadcasting
    try:
        n_broadcasted, k_broadcasted = np.broadcast_arrays(n, k)
    except ValueError:
        # This should ideally not happen if broadcast_shape was calculated successfully
        raise ValueError(f"Could not broadcast input shapes {n.shape} and {k.shape}")

    # Handle cases where k > n or n = 0 using broadcasted arrays
    # Ensure k>=0 check is included
    valid_mask = (
        (n_broadcasted > 0) & (k_broadcasted >= 0) & (k_broadcasted <= n_broadcasted)
    )

    # Perform calculation only where the mask is True
    if np.any(valid_mask):
        # Extract valid counts based on the mask applied to the broadcasted k
        k_valid = k_broadcasted[valid_mask]

        # Determine the correct value/array for 'nobs' argument in proportion_confint
        if n.ndim == 0:
            # If original n was scalar, use its scalar value.
            # proportion_confint handles broadcasting nobs scalar to count array.
            n_val_for_confint = n.item()
        else:
            # If original n was an array, use the broadcasted and masked version.
            n_val_for_confint = n_broadcasted[valid_mask]

        # Calculate confidence intervals for the valid subset
        p_low_valid, p_upp_valid = proportion_confint(
            k_valid, n_val_for_confint, method="wilson", alpha=alpha
        )

        # Place the results back into the full-sized arrays using the mask
        p_low[valid_mask] = p_low_valid
        p_upp[valid_mask] = p_upp_valid

    # Calculate midpoint p and half-width delta_p
    with np.errstate(
        invalid="ignore"
    ):  # Ignore warnings from NaN comparisons/arithmetic
        p = (p_upp + p_low) / 2
        delta_p = p_upp - p

    # --- Return type handling (similar to original) ---
    # If original inputs were Series, convert back, preserving index
    if isinstance(num_trials, pd.Series):
        return pd.Series(p, index=num_trials.index, name="p"), pd.Series(
            delta_p, index=num_trials.index, name="delta_p"
        )
    elif isinstance(num_successes, pd.Series):
        return pd.Series(p, index=num_successes.index, name="p"), pd.Series(
            delta_p, index=num_successes.index, name="delta_p"
        )
    else:  # Inputs were scalars or numpy arrays
        if p.ndim == 0:  # Return scalars if the result shape is scalar
            return (
                p.item() if p.size == 1 else p,
                delta_p.item() if delta_p.size == 1 else delta_p,
            )
        else:  # Otherwise return numpy arrays
            return p, delta_p


def calculate_all_df_ps(
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
    n: int | None = None,
    T: int | None = None,
    p: float | None = None,
    cluster_frac_precision: int = 3,
) -> pd.DataFrame:
    """
    Calculate and combine df_ps for specified or all (n, T, p) pairs found.

    If n, T, and p are provided, calculates df_ps only for that specific combination.
    Otherwise, scans the directory for simulation files, identifies all unique
    (n, T, p) combinations, calculates df_ps for each, computes post-selection
    probabilities (p_abort, p_fail) with confidence intervals, and concatenates
    them into a single DataFrame with a multi-index (n, T, p, c).

    Parameters
    ----------
    data_dir : str
        Directory path containing the simulation data files.
    n : int, optional
        Specific number of qubits for the BB code. If None, scan for all n.
    T : int, optional
        Specific number of rounds. If None, scan for all T.
    p : float, optional
        Specific physical error rate. If None, scan for all p.
    cluster_frac_precision : int, optional
        Number of decimal places to consider for the cluster fraction histogram bins.
        Higher values increase precision but also memory usage for histograms.
        Defaults to 3 (bins of size 0.001).

    Returns
    -------
    all_df_ps : pd.DataFrame
        Combined DataFrame with multi-index (n, T, p, c) containing
        post-selection statistics and probabilities. Returns an empty DataFrame
        if no valid data is found or processed for the specified or scanned combinations.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return pd.DataFrame()

    target_combinations: List[Tuple[int, int, float]] = []

    # --- Determine target combinations ---
    if n is not None and T is not None and p is not None:
        # Specific combination provided
        print(f"Processing specified combination: n={n}, T={T}, p={p}")
        try:
            # Check if files exist for this specific combination early
            find_bb_simulation_files(n=n, T=T, p=p, data_dir=data_dir)
            target_combinations = [(n, T, p)]
        except (FileNotFoundError, ValueError) as e:
            print(
                f"Info: Cannot proceed with specified combination n={n}, T={T}, p={p}: {e}"
            )
            return pd.DataFrame()
    else:
        # Scan directory for all combinations
        print(f"Scanning {data_dir} for all available (n, T, p) combinations...")
        all_files = glob.glob(os.path.join(data_dir, "*.feather"))
        if not all_files:
            print(f"Error: No .feather files found in {data_dir}")
            return pd.DataFrame()

        param_pattern = re.compile(r"n(\d+)(?:_T(\d+))?_p([\d\.]+)_.*\.feather")
        unique_combinations: Set[Tuple[int, int, float]] = set()

        for fpath in all_files:
            match = param_pattern.search(os.path.basename(fpath))
            if match:
                try:
                    scan_n = int(match.group(1))
                    scan_p = float(match.group(3))
                    T_str = match.group(2)
                    if T_str:
                        scan_T = int(T_str)
                    else:
                        try:
                            scan_T = get_BB_distance(scan_n)
                        except Exception as e:
                            print(
                                f"Warning: Could not determine T for n={scan_n} from filename {os.path.basename(fpath)} and get_BB_distance failed: {e}. Skipping this file."
                            )
                            continue
                    # Filter based on any provided parameters
                    if (
                        (n is None or scan_n == n)
                        and (T is None or scan_T == T)
                        and (p is None or np.isclose(scan_p, p))
                    ):
                        unique_combinations.add((scan_n, scan_T, scan_p))
                except (ValueError, TypeError) as e:
                    print(
                        f"Warning: Could not parse parameters from {os.path.basename(fpath)}: {e}. Skipping this file."
                    )
                except Exception as e:
                    print(
                        f"Warning: Unexpected error processing filename {os.path.basename(fpath)}: {e}. Skipping this file."
                    )

        if not unique_combinations:
            print("Error: No valid (n, T, p) combinations found matching the criteria.")
            return pd.DataFrame()

        target_combinations = sorted(list(unique_combinations))
        print(
            f"Found {len(target_combinations)} matching unique (n, T, p) combinations."
        )

    all_ps_dfs: List[pd.DataFrame] = []
    processed_combinations_count = 0
    # Store total samples for each combination to calculate probabilities later
    # combination_results = {} # No longer needed with this structure

    for comb_n, comb_T, comb_p in target_combinations:
        print(f"Processing combination: n={comb_n}, T={comb_T}, p={comb_p}")
        # Call the function which now returns df_ps and total_samples
        df_ps_single, total_samples = calculate_df_ps_for_combination(
            n=comb_n,
            T=comb_T,
            p=comb_p,
            data_dir=data_dir,
            cluster_frac_precision=cluster_frac_precision,
        )

        if not df_ps_single.empty and total_samples > 0:
            # --- Calculate probabilities using total_samples for the group ---
            print(f"  Calculating probabilities for {total_samples} total samples...")
            # Calculate p_acc and its confidence interval (num_trials = total_samples)
            p_acc, delta_p_acc = _calculate_p_confint(
                total_samples, df_ps_single["num_accs"]
            )

            # Calculate p_fail and its confidence interval (trials = num_accs, successes = num_fails)
            # Ensure num_accs > 0 to avoid division by zero or invalid inputs
            valid_fail_calc = df_ps_single["num_accs"] > 0
            p_fail = pd.Series(np.nan, index=df_ps_single.index)
            delta_p_fail = pd.Series(np.nan, index=df_ps_single.index)

            if valid_fail_calc.any():
                # Use .loc for safe assignment based on the boolean mask
                p_fail_valid, delta_p_fail_valid = _calculate_p_confint(
                    df_ps_single.loc[valid_fail_calc, "num_accs"],
                    df_ps_single.loc[valid_fail_calc, "num_fails"],
                )
                p_fail.loc[valid_fail_calc] = p_fail_valid
                delta_p_fail.loc[valid_fail_calc] = delta_p_fail_valid

            # Add calculated probabilities and parameters to the DataFrame
            df_ps_single["p_abort"] = 1.0 - p_acc
            df_ps_single["delta_p_abort"] = delta_p_acc
            df_ps_single["p_fail"] = p_fail
            df_ps_single["delta_p_fail"] = delta_p_fail
            # --- End Probability Calculation ---

            df_ps_single["n"] = comb_n  # Use loop variables
            df_ps_single["T"] = comb_T  # Use loop variables
            df_ps_single["p"] = comb_p  # Use loop variables
            all_ps_dfs.append(df_ps_single)  # Add the processed DataFrame to the list
            processed_combinations_count += 1
            print(
                f"  → Successfully processed combination, added {len(df_ps_single)} rows to results."
            )
        else:
            print(
                f"  → No df_ps data generated or an error occurred for n={comb_n}, T={comb_T}, p={comb_p}."
            )

    if not all_ps_dfs:
        print(
            "\nError: No df_ps dataframes were successfully generated for any matching combination."
        )
        return pd.DataFrame()

    print(
        f"\nConcatenating results for {processed_combinations_count} successful combinations..."
    )
    try:
        all_df_ps = pd.concat(all_ps_dfs, ignore_index=True)
    except Exception as e:
        print(f"Error during final concatenation: {e}")
        # Attempt to return partial results if concatenation fails? Or empty?
        # For now, return empty as the final structure is compromised.
        return pd.DataFrame()

    print("Setting multi-index (n, T, p, c)...")
    try:
        # Ensure required columns exist before setting index
        required_cols = ["n", "T", "p", "c"]
        if not all(col in all_df_ps.columns for col in required_cols):
            missing = [col for col in required_cols if col not in all_df_ps.columns]
            print(f"Error: Cannot set multi-index. Missing columns: {missing}")
            print("Returning concatenated DataFrame without multi-index.")
            return all_df_ps

        all_df_ps = all_df_ps.set_index(required_cols)
        print("Sorting index...")
        all_df_ps = all_df_ps.sort_index()
        print("Concatenation and indexing complete.")
    except Exception as e:
        print(f"An unexpected error occurred during final indexing/sorting: {e}")
        print("Returning concatenated DataFrame without multi-index.")
        # Return the unindexed but concatenated frame in case of error
        # The probability columns should already be there
        return all_df_ps  # Return the already concatenated frame

    return all_df_ps
