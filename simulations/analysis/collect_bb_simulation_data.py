import glob
import os
import re
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from src.ldpc_post_selection.build_circuit import build_BB_circuit, get_BB_distance


def load_bb_simulation_data(
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
) -> pd.DataFrame:
    """
    Load BB code circuit simulation data from feather files and combine them for a specific (n, T, p) tuple.

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
    combined_df : pd.DataFrame
        Combined DataFrame containing all simulation data for the specified (n, T, p)

    Raise
    ------
    FileNotFoundError
        If the data directory does not exist.
    ValueError
        If no data files arg found for the specified (n, T, p) or if loading fails.
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

    print(f"Loading data for n={n}, T={T}, p={p} from {len(file_paths)} files...")

    dfs: List[pd.DataFrame] = []
    for file_path in file_paths:
        try:
            df = pd.read_feather(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")

    if not dfs:
        raise ValueError(
            f"Failed to load any data for n={n}, T={T}, p={p} despite finding files."
        )

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  → Combined {len(dfs)} files with total {len(combined_df)} rows")

    return combined_df


def calculate_df_ps_for_combination(
    n: int,
    T: int,
    p: float,
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
) -> pd.DataFrame:
    """
    Calculate the post-selection DataFrame (df_ps) for a single (n, T, p) combination.

    This function loads the simulation data, computes the cluster fraction, and then
    calculates the number of accepted trials and failed trials for various
    post-selection thresholds 'c'.

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

    Returns
    -------
    df_ps : pd.DataFrame
        DataFrame with columns ['c', 'num_accs', 'num_fails'] containing the
        post-selection statistics for the given (n, T, p).
        Returns an empty DataFrame if data loading or processing fails.

    Notes
    -----
    The threshold 'c' is defined as 1 - cluster_frac_threshold.
    The calculation is optimized using sorted arrays and cumulative sums.
    """
    try:
        df = load_bb_simulation_data(n=n, T=T, p=p, data_dir=data_dir)

        if df.empty:
            print(f"Warning: Loaded empty DataFrame for n={n}, T={T}, p={p}. Skipping.")
            return pd.DataFrame()

        if "fail" not in df.columns or "cluster_size_sum" not in df.columns:
            print(
                f"Warning: Missing required columns ('fail', 'cluster_size_sum') for n={n}, T={T}, p={p}. Skipping."
            )
            return pd.DataFrame()

        circuit = build_BB_circuit(n=n, T=T, p=p)
        try:
            detector_error_model = circuit.detector_error_model()
            if detector_error_model is None:
                print(
                    f"Warning: Could not get detector_error_model for n={n}, T={T}, p={p}. Skipping."
                )
                return pd.DataFrame()
            num_errors = detector_error_model.num_errors
            if num_errors is None or num_errors <= 0:
                print(
                    f"Warning: Invalid num_errors ({num_errors}) for n={n}, T={T}, p={p}. Skipping."
                )
                return pd.DataFrame()
        except AttributeError:
            print(
                f"Warning: stim.Circuit object has no attribute 'detector_error_model' for n={n}, T={T}, p={p}. Check stim version or circuit construction. Skipping."
            )
            return pd.DataFrame()
        except Exception as e:
            print(
                f"Warning: Error getting num_errors for n={n}, T={T}, p={p}: {e}. Skipping."
            )
            return pd.DataFrame()

        df["cluster_frac"] = df["cluster_size_sum"] / num_errors

        max_frac = df["cluster_frac"].max()
        if pd.isna(max_frac):
            print(
                f"Warning: Max cluster_frac is NaN for n={n}, T={T}, p={p}. Skipping."
            )
            return pd.DataFrame()

        thr_step = 0.001
        thr = np.arange(0, max_frac + thr_step, thr_step).round(6)
        if len(thr) == 0:
            print(
                f"Warning: Threshold array is empty for n={n}, T={T}, p={p}. Skipping."
            )
            return pd.DataFrame()

        df_sorted = df.sort_values("cluster_frac").reset_index(drop=True)
        cluster_vals = df_sorted["cluster_frac"].values
        fail_vals = df_sorted["fail"].astype(int).values

        cum_fails = np.cumsum(fail_vals)
        acc_counts = np.searchsorted(cluster_vals, thr, side="right")

        fail_counts = np.zeros_like(acc_counts, dtype=int)
        valid_indices_mask = acc_counts > 0
        if np.any(valid_indices_mask):
            lookup_indices = acc_counts[valid_indices_mask] - 1
            # Ensure lookup indices are valid
            valid_lookup_mask = lookup_indices < len(cum_fails)
            valid_acc_indices = np.where(valid_indices_mask)[0]

            # Apply valid lookup mask to filter indices before assignment
            assign_indices = valid_acc_indices[valid_lookup_mask]
            assign_values_indices = lookup_indices[valid_lookup_mask]

            if len(assign_indices) > 0:
                fail_counts[assign_indices] = cum_fails[assign_values_indices]

        df_ps = (
            pd.DataFrame(
                {
                    "c": 1 - thr,
                    "num_accs": acc_counts,
                    "num_fails": fail_counts,
                }
            )
            .sort_values("c", ascending=False)  # Sort by c descending as in notebook
            .reset_index(drop=True)
        )

        df_ps = df_ps[df_ps["num_accs"] > 0].reset_index(drop=True)

        return df_ps

    except (FileNotFoundError, ValueError) as e:
        print(
            f"Info: Skipping n={n}, T={T}, p={p} due to data loading/validation issue: {e}"
        )
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing n={n}, T={T}, p={p}: {e}. Skipping this combination.")
        # Consider logging the traceback here for debugging
        # import traceback
        # traceback.print_exc()
        return pd.DataFrame()


def _calculate_p_confint(
    num_trials: int | pd.Series,
    num_successes: int | pd.Series,
    alpha: float = 0.05,  # Corresponds to 95% confidence interval
) -> Tuple[float | pd.Series, float | pd.Series]:
    """
    Calculate proportion and confidence interval half-width using Wilson score interval.

    Parameters
    ----------
    num_trials : int or pd.Series
        Number of trials.
    num_successes : int or pd.Series
        Number of successes.
    alpha : float, optional
        Significance level (default is 0.05 for 95% confidence).

    Returns
    -------
    p : float or pd.Series
        Estimated proportion of successes.
    delta_p : float or pd.Series
        Half-width of the confidence interval.
    """
    # Ensure inputs are numpy arrays for vectorized operations if they are Series
    k = np.asarray(num_successes)
    n = np.asarray(num_trials)

    # Handle cases where k > n or n = 0 to avoid errors in proportion_confint
    valid_mask = (n > 0) & (k >= 0) & (k <= n)
    p_low = np.full_like(n, np.nan, dtype=float)
    p_upp = np.full_like(n, np.nan, dtype=float)

    if np.any(valid_mask):
        k_valid = k[valid_mask]
        n_valid = n[valid_mask]
        p_low_valid, p_upp_valid = proportion_confint(
            k_valid, n_valid, method="wilson", alpha=alpha
        )
        p_low[valid_mask] = p_low_valid
        p_upp[valid_mask] = p_upp_valid

    # Calculate midpoint p and half-width delta_p
    # Handle NaNs potentially introduced
    with np.errstate(
        invalid="ignore"
    ):  # Ignore warnings from NaN comparisons/arithmetic
        p = (p_upp + p_low) / 2
        delta_p = p_upp - p

    # If original inputs were Series, convert back
    if isinstance(num_trials, pd.Series):
        return pd.Series(p, index=num_trials.index), pd.Series(
            delta_p, index=num_trials.index
        )
    elif isinstance(num_successes, pd.Series):
        return pd.Series(p, index=num_successes.index), pd.Series(
            delta_p, index=num_successes.index
        )
    else:  # Return floats if inputs were scalars
        return p.item() if p.size == 1 else p, (
            delta_p.item() if delta_p.size == 1 else delta_p
        )


def calculate_all_df_ps(
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
) -> pd.DataFrame:
    """
    Calculate and combine df_ps for all (n, p) pairs found in the data directory.

    Scans the directory for simulation files, identifies unique (n, T, p) combinations,
    calculates df_ps for each, calculates post-selection probabilities (p_abort, p_fail)
    with confidence intervals, and concatenates them into a single DataFrame
    with a multi-index (n, T, p, c).

    Parameters
    ----------
    data_dir : str
        Directory path containing the simulation data files.

    Returns
    -------
    all_df_ps : pd.DataFrame
        Combined DataFrame with multi-index (n, T, p, c) containing
        post-selection statistics and probabilities for all found combinations.
        Returns an empty DataFrame if no valid data is found or processed.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return pd.DataFrame()

    all_files = glob.glob(os.path.join(data_dir, "*.feather"))
    if not all_files:
        print(f"Error: No .feather files found in {data_dir}")
        return pd.DataFrame()

    param_pattern = re.compile(r"n(\d+)(?:_T(\d+))?_p([\d\.]+)_.*\.feather")
    combinations: Set[Tuple[int, int, float]] = set()

    for fpath in all_files:
        match = param_pattern.search(os.path.basename(fpath))
        if match:
            try:
                n = int(match.group(1))
                p = float(match.group(3))
                T_str = match.group(2)
                if T_str:
                    T = int(T_str)
                else:
                    try:
                        T = get_BB_distance(n)
                    except Exception as e:
                        print(
                            f"Warning: Could not determine T for n={n} from filename {os.path.basename(fpath)} and get_BB_distance failed: {e}. Skipping this file."
                        )
                        continue
                combinations.add((n, T, p))
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Could not parse parameters from {os.path.basename(fpath)}: {e}. Skipping this file."
                )
            except Exception as e:
                print(
                    f"Warning: Unexpected error processing filename {os.path.basename(fpath)}: {e}. Skipping this file."
                )

    if not combinations:
        print("Error: No valid (n, T, p) combinations found from filenames.")
        return pd.DataFrame()

    print(f"Found {len(combinations)} unique (n, T, p) combinations.")

    all_ps_dfs: List[pd.DataFrame] = []
    processed_combinations = 0
    for n, T, p in sorted(list(combinations)):
        print(f"\nProcessing combination: n={n}, T={T}, p={p}")
        df_ps_single = calculate_df_ps_for_combination(n=n, T=T, p=p, data_dir=data_dir)

        if not df_ps_single.empty:
            df_ps_single["n"] = n
            df_ps_single["T"] = T
            df_ps_single["p"] = p
            all_ps_dfs.append(df_ps_single)
            processed_combinations += 1
            print(f"  → Successfully generated df_ps with {len(df_ps_single)} rows.")
        else:
            print(
                f"  → No df_ps data generated or an error occurred for n={n}, T={T}, p={p}."
            )

    if not all_ps_dfs:
        print(
            "\nError: No df_ps dataframes were successfully generated for any combination."
        )
        return pd.DataFrame()

    print(
        f"\nConcatenating results for {processed_combinations} successful combinations..."
    )
    try:
        all_df_ps = pd.concat(all_ps_dfs, ignore_index=True)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return pd.DataFrame()  # Return empty if concat fails

    # --- Calculate probabilities ---
    print("Calculating total samples per group...")
    # Find the max num_accs for each group, which represents the initial total samples
    num_samples_per_group = all_df_ps.groupby(["n", "T", "p"])["num_accs"].transform(
        "max"
    )
    all_df_ps["num_samples"] = num_samples_per_group

    print("Calculating p_abort and p_fail with confidence intervals...")
    # Calculate p_acc and its confidence interval
    p_acc, delta_p_acc = _calculate_p_confint(
        all_df_ps["num_samples"], all_df_ps["num_accs"]
    )

    # Calculate p_fail and its confidence interval (trials = num_accs, successes = num_fails)
    # Ensure num_accs > 0 to avoid division by zero or invalid inputs for proportion_confint
    valid_fail_calc = all_df_ps["num_accs"] > 0
    p_fail = pd.Series(np.nan, index=all_df_ps.index)
    delta_p_fail = pd.Series(np.nan, index=all_df_ps.index)

    if valid_fail_calc.any():
        p_fail_valid, delta_p_fail_valid = _calculate_p_confint(
            all_df_ps.loc[valid_fail_calc, "num_accs"],
            all_df_ps.loc[valid_fail_calc, "num_fails"],
        )
        p_fail.loc[valid_fail_calc] = p_fail_valid
        delta_p_fail.loc[valid_fail_calc] = delta_p_fail_valid

    # Calculate p_abort
    all_df_ps["p_abort"] = 1.0 - p_acc
    all_df_ps["delta_p_abort"] = delta_p_acc  # CI width is the same for p and 1-p
    all_df_ps["p_fail"] = p_fail
    all_df_ps["delta_p_fail"] = delta_p_fail

    # Clean up temporary column
    all_df_ps = all_df_ps.drop(columns=["num_samples"])
    # --- End Probability Calculation ---

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
        return pd.concat(all_ps_dfs, ignore_index=True).drop(
            columns=["num_samples"], errors="ignore"
        )  # Drop again just in case

    return all_df_ps
