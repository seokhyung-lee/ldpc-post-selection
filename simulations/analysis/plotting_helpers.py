import matplotlib
import pandas as pd
from typing import Union, List, Callable
import matplotlib.pyplot as plt
import os
import re
import glob


def error_band_plot(
    x, y, delta_y, ax=None, color=None, alpha=0.3, **kwargs
) -> list[matplotlib.lines.Line2D]:
    if ax is None:
        ax = plt.gca()
    line = ax.plot(x, y, color=color, **kwargs)

    # If color is None, get the color from the plotted line
    if color is None:
        color = line[0].get_color()

    ax.fill_between(
        x,
        y - delta_y,
        y + delta_y,
        alpha=alpha,
        color=color,
    )

    return line


def get_lower_envelope(df: pd.DataFrame, use_pfail_upper: bool = False) -> pd.DataFrame:
    """
    Extract the lower envelope from a DataFrame based on p_abort and p_fail metrics.

    Sorts the DataFrame by p_abort and keeps only rows that form the lower envelope
    of p_fail (or p_fail + delta_p_fail) values.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing columns 'p_abort', 'p_fail', 'delta_p_fail'
    use_pfail_upper : bool, optional
        If True (default), use p_fail + delta_p_fail as the metric for envelope calculation.
        If False, use p_fail only.

    Returns
    -------
    pandas.DataFrame
        DataFrame with only the lower envelope rows, sorted by p_abort
    """
    if df.empty:
        return df.copy()

    # Sort by p_abort
    df_sorted = df.sort_values("p_abort").reset_index(drop=True)

    # Create the metric column based on use_pfail_upper parameter
    if use_pfail_upper:
        df_sorted["p_fail_metric"] = df_sorted["p_fail"] + df_sorted["delta_p_fail"]
    else:
        df_sorted["p_fail_metric"] = df_sorted["p_fail"]

    # Calculate lower envelope using cumulative minimum of the metric
    df_sorted["p_fail_metric_cummin"] = df_sorted["p_fail_metric"].cummin()
    envelope_mask = df_sorted["p_fail_metric"] <= df_sorted["p_fail_metric_cummin"]
    envelope_df = df_sorted[envelope_mask].copy()

    # Clean up temporary columns
    envelope_df = envelope_df.drop(["p_fail_metric", "p_fail_metric_cummin"], axis=1)

    return envelope_df.reset_index(drop=True)


def take_best_by_from_df_ps_dict(
    df_ps_dict: dict[str, pd.DataFrame],
    exclude: Union[List[str], Callable[[str], bool], None] = None,
    use_pfail_upper: bool = False,
) -> pd.DataFrame:
    """
    Extract the best performing configurations from a dictionary of DataFrames.

    For each (code_param, T, p) combination, this function concatenates all DataFrames,
    sorts by p_abort, and keeps only the lower envelope of p_fail values.

    Parameters
    ----------
    df_ps_dict : dict of str to pandas DataFrame
        Dictionary where keys are "by" identifiers and values are DataFrames with
        4-level index (code_param, T, p, by_val) and columns 'p_fail', 'delta_p_fail', 'p_abort'
    exclude : list of str, callable, or None, optional
        Either a list of 'by' keys to exclude from the calculation, or a boolean function
        that takes a 'by' key as input and returns True if it should be excluded.
        If None, no exclusions are applied.
    use_pfail_upper : bool, optional
        If True (default), use p_fail + delta_p_fail as the metric for aggregation.
        If False, use p_fail only.

    Returns
    -------
    pandas.DataFrame
        DataFrame with index (code_param, T, p, p_abort) and columns 'p_fail', 'delta_p_fail', 'by'
    """
    if not df_ps_dict:
        return pd.DataFrame()

    # Filter df_ps_dict based on exclude parameter
    if exclude is not None:
        if callable(exclude):
            # exclude is a function
            filtered_dict = {k: v for k, v in df_ps_dict.items() if not exclude(k)}
        else:
            # exclude is a list of strings
            filtered_dict = {k: v for k, v in df_ps_dict.items() if k not in exclude}
    else:
        filtered_dict = df_ps_dict

    if not filtered_dict:
        return pd.DataFrame()

    # Get index structure from first DataFrame
    first_df = next(iter(filtered_dict.values()))
    index_names = first_df.index.names
    code_param_name, by_val_name = index_names[0], index_names[3]

    # Concatenate all DataFrames with 'by' information
    df_list = []
    for by_key, df in filtered_dict.items():
        df_copy = df.copy()
        df_copy["by"] = by_key
        df_list.append(df_copy.reset_index())

    if not df_list:
        index = pd.MultiIndex.from_tuples(
            [], names=[code_param_name, "T", "p", "p_abort"]
        )
        return pd.DataFrame(columns=["p_fail", "delta_p_fail", "by"], index=index)

    # Combine all data
    combined_df = pd.concat(df_list, ignore_index=True)

    # Group by (code_param, T, p) and process each group
    def process_group(group):
        # Get lower envelope using the separated function
        envelope_df = get_lower_envelope(group, use_pfail_upper=use_pfail_upper)

        # Handle duplicates with same p_abort and p_fail values
        # Create the metric for grouping duplicates
        if use_pfail_upper:
            envelope_df["p_fail_metric"] = (
                envelope_df["p_fail"] + envelope_df["delta_p_fail"]
            )
        else:
            envelope_df["p_fail_metric"] = envelope_df["p_fail"]

        # Group by p_abort and p_fail_metric, then aggregate
        def aggregate_duplicates(subgroup):
            if len(subgroup) == 1:
                return subgroup.iloc[0]

            # Find minimum delta_p_fail
            min_delta = subgroup["delta_p_fail"].min()
            best_rows = subgroup[subgroup["delta_p_fail"] == min_delta]

            if len(best_rows) == 1:
                return best_rows.iloc[0]
            else:
                # Multiple rows with same delta_p_fail, merge them
                result = best_rows.iloc[0].copy()
                result["by"] = ",".join(best_rows["by"].astype(str))
                return result

        # Apply aggregation to handle duplicates
        result = (
            envelope_df.groupby(["p_abort", "p_fail_metric"], group_keys=False)
            .apply(aggregate_duplicates)
            .reset_index(drop=True)
        )

        # Remove the temporary metric column
        result = result.drop("p_fail_metric", axis=1)

        return result

    # Process all groups and concatenate results
    processed_groups = (
        combined_df.groupby([code_param_name, "T", "p"], group_keys=False)
        .apply(process_group)
        .reset_index(drop=True)
    )

    if processed_groups.empty:
        index = pd.MultiIndex.from_tuples(
            [], names=[code_param_name, "T", "p", "p_abort"]
        )
        return pd.DataFrame(columns=["p_fail", "delta_p_fail", "by"], index=index)

    # Set the final multi-index
    result_df = processed_groups.set_index([code_param_name, "T", "p", "p_abort"])

    # Keep only the required columns
    return result_df[["p_fail", "delta_p_fail", "by"]]


def load_ps_data(code: str) -> dict[str, pd.DataFrame]:
    """
    Load post-selection data for a specific code.

    Reads post-selection data from the corresponding directory structure,
    where each subdirectory represents a different "by" value, and combines
    all pkl files within each subdirectory into a single DataFrame with
    parameters extracted from filenames as MultiIndex levels.

    Parameters
    ----------
    code : str
        Code type to load data for. Must be one of 'hgp', 'surface', or 'bb'.
        For 'surface', both 'surface' and 'surface_matching' directories are loaded.

    Returns
    -------
    dict of str to pandas DataFrame
        Dictionary where keys are "by" values (or "matching_" + by for surface_matching),
        and values are DataFrames with MultiIndex containing parameters extracted from
        pkl filenames plus the original DataFrame index.
    """
    if code not in ["hgp", "surface", "bb"]:
        raise ValueError(
            f"Invalid code '{code}'. Must be one of 'hgp', 'surface', 'bb'"
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "..", "data", "post_selection")
    result_dict = {}

    # Determine which directories to process
    if code == "surface":
        dirs_to_process = [
            (os.path.join(base_dir, "surface"), ""),
            (os.path.join(base_dir, "surface_matching"), "matching_"),
        ]
    else:
        dirs_to_process = [(os.path.join(base_dir, code), "")]

    for code_dir, prefix in dirs_to_process:
        if not os.path.exists(code_dir):
            print(f"Skipping {code_dir} because it does not exist")
            continue

        # Get all subdirectories (these are the "by" values)
        for by_dir in os.listdir(code_dir):
            by_path = os.path.join(code_dir, by_dir)
            if not os.path.isdir(by_path):
                continue

            # Find all pkl files in this by directory
            pkl_files = glob.glob(os.path.join(by_path, "*.pkl"))
            if not pkl_files:
                continue

            dataframes_to_concat = []

            for pkl_file in pkl_files:
                # Extract parameters from filename
                filename = os.path.basename(pkl_file)
                filename_no_ext = filename.replace(".pkl", "")

                # Parse parameters using regex - flexible for different parameter patterns
                # Pattern: param1_value1_param2_value2_...
                param_matches = re.findall(r"([a-zA-Z]+)([0-9.]+)", filename_no_ext)

                if not param_matches:
                    continue

                # Create parameter dictionary
                params = {}
                for param_name, param_value in param_matches:
                    # Try to convert to float if possible, otherwise keep as string
                    try:
                        if "." in param_value:
                            params[param_name] = float(param_value)
                        else:
                            params[param_name] = int(param_value)
                    except ValueError:
                        params[param_name] = param_value

                # Load the DataFrame
                try:
                    df = pd.read_pickle(pkl_file)
                except Exception as e:
                    print(f"Warning: Could not load {pkl_file}: {e}")
                    continue

                # Create MultiIndex with parameters
                param_names = list(params.keys())
                param_values = list(params.values())

                if len(param_names) == 1:
                    new_index_tuples = [(param_values[0], idx) for idx in df.index]
                    new_index_names = [param_names[0]] + (
                        df.index.names if df.index.names[0] is not None else ["index"]
                    )
                else:
                    new_index_tuples = [
                        tuple(param_values) + (idx,) for idx in df.index
                    ]
                    new_index_names = param_names + (
                        df.index.names if df.index.names[0] is not None else ["index"]
                    )

                # Create new MultiIndex
                new_index = pd.MultiIndex.from_tuples(
                    new_index_tuples, names=new_index_names
                )
                df_with_params = df.copy()
                df_with_params.index = new_index

                dataframes_to_concat.append(df_with_params)

            if dataframes_to_concat:
                # Concatenate all DataFrames for this "by" value
                combined_df = pd.concat(dataframes_to_concat, axis=0)

                # Sort by index for better organization
                combined_df = combined_df.sort_index()

                # Store in result dictionary with appropriate key
                key = f"{prefix}{by_dir}"
                result_dict[key] = combined_df

    return result_dict


def aggregate_seeds(df_ps: pd.DataFrame, use_pfail_upper: bool = False) -> pd.DataFrame:
    """
    Aggregate post-selection data across multiple seeds by taking lower envelope for each parameter combination.

    Takes a DataFrame with seed in the index, converts seed to a column, then for each (n, T, p)
    combination applies the get_lower_envelope function to find the best performing configurations
    across all seeds and 'by' values.

    Parameters
    ----------
    df_ps : pandas DataFrame
        DataFrame with MultiIndex (n, T, p, seed, by) and columns 'p_fail', 'delta_p_fail', 'p_abort'.
        The 'by' level name can be variable (not necessarily called 'by').
    use_pfail_upper : bool, optional
        If True (default), use p_fail + delta_p_fail as the metric for envelope calculation.
        If False, use p_fail only.

    Returns
    -------
    pandas DataFrame
        DataFrame with MultiIndex (n, T, p, p_abort) and columns 'p_fail', 'delta_p_fail', 'seed'.
        Contains the lower envelope of performance across all seeds for each parameter combination.
    """
    if df_ps.empty:
        return df_ps.copy()

    # Get the index names to understand the structure
    index_names = df_ps.index.names
    if len(index_names) != 5:
        raise ValueError(
            f"Expected 5-level MultiIndex, got {len(index_names)} levels: {index_names}"
        )

    n_name, T_name, p_name, seed_name, by_name = index_names

    # Reset index to work with columns
    df_reset = df_ps.reset_index()

    # Group by (n, T, p) and process each group
    def process_group(group):
        # Apply get_lower_envelope to this group
        envelope_df = get_lower_envelope(group, use_pfail_upper=use_pfail_upper)
        return envelope_df

    # Process all groups
    processed_groups = (
        df_reset.groupby([n_name, T_name, p_name], group_keys=False)
        .apply(process_group)
        .reset_index(drop=True)
    )

    if processed_groups.empty:
        # Return empty DataFrame with correct structure
        index = pd.MultiIndex.from_tuples([], names=[n_name, T_name, p_name, "p_abort"])
        return pd.DataFrame(columns=["p_fail", "delta_p_fail", seed_name], index=index)

    # Set the final multi-index, keeping seed as a column
    result_df = processed_groups.set_index([n_name, T_name, p_name, "p_abort"])

    # Keep required columns (seed, p_fail, delta_p_fail), drop 'by' column
    columns_to_keep = ["p_fail", "delta_p_fail", seed_name]
    result_df = result_df[columns_to_keep]

    return result_df
