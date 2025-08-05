import matplotlib.axes
import matplotlib.lines
import pandas as pd
from typing import Union, List, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.proportion import proportion_confint

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")


def error_band_plot(
    x: np.ndarray,
    y: np.ndarray,
    delta_y: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    color: str | None = None,
    alpha: float = 0.3,
    **kwargs,
) -> list[matplotlib.lines.Line2D]:
    """
    Create a line plot with error bands (confidence intervals).

    This function plots a line with x and y coordinates and fills the area between
    y-delta_y and y+delta_y to visualize uncertainty or error bounds around the main line.

    Parameters
    ----------
    x : 1D numpy array
        X-axis coordinates for the line plot.
    y : 1D numpy array
        Y-axis coordinates for the line plot.
    delta_y : 1D numpy array
        Error/uncertainty values used to create the error band around the main line.
        The band extends from y-delta_y to y+delta_y.
    ax : matplotlib Axes object or None, optional
        Axes to plot on. If None, uses the current axes from plt.gca().
    color : str or None, optional
        Color for both the line and the error band. If None, matplotlib automatically
        assigns a color and uses it for both elements.
    alpha : float, optional
        Transparency level for the error band fill. Must be between 0 (transparent)
        and 1 (opaque). Default is 0.3.
    **kwargs
        Additional keyword arguments passed to the matplotlib plot function.

    Returns
    -------
    list of matplotlib Line2D objects
        List containing the line objects created by the plot function.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    delta_y = np.asarray(delta_y)

    if x.ndim != 1 or y.ndim != 1 or delta_y.ndim != 1:
        raise ValueError("x, y, and delta_y must be 1D arrays")

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
        edgecolor=None,
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


def load_data(
    code: Union[str, List[str]],
    data_type: str = "ps",
    prefixes: Union[str, List[str], None] = None,
    filter: dict[str, Union[int, float]] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load simulation data for one or more codes.

    Reads simulation data from the corresponding directory structure,
    where each subdirectory represents a different "by" value, and combines
    all pkl files within each subdirectory into a single DataFrame with
    parameters extracted from filenames as MultiIndex levels.

    Parameters
    ----------
    code : str or list of str
        Code type(s) to load data for. Can be a single code or a list of codes.
    data_type : str, default "ps"
        Type of data to load. Must be either "ps" (post-selection data from
        simulations/data/post_selection) or "agg" (aggregated data from
        simulations/data/aggregated).
    prefixes : str, list of str, or None, optional
        Prefix(es) to apply to "by" values for dictionary keys. If None, no prefixes
        are applied. If a single string and code is a list, the same prefix is applied
        to all codes. If a list, must have the same length as code list.
    filter : dict of str to int or float, optional
        Dictionary of parameter filters. Only files whose extracted parameters
        match all filter conditions will be processed. For example, {'n': 5, 'k': 3}
        will only process files where parameter 'n' equals 5 and parameter 'k' equals 3.

    Returns
    -------
    dict of str to pandas DataFrame
        Dictionary where keys are prefixed "by" values, and values are DataFrames
        with MultiIndex containing parameters extracted from pkl filenames plus
        the original DataFrame index.

    Raises
    ------
    ValueError
        If data_type is not "ps" or "agg", if prefixes list length doesn't match
        code list length, or if duplicate keys would be generated.
    """
    # Normalize inputs to lists
    if isinstance(code, str):
        code_list = [code]
    else:
        code_list = list(code)

    if prefixes is None:
        prefixes_list = [""] * len(code_list)
    elif isinstance(prefixes, str):
        prefixes_list = [prefixes] * len(code_list)
    else:
        prefixes_list = list(prefixes)

    # Validate inputs
    if data_type not in ["ps", "agg"]:
        raise ValueError(f"data_type must be 'ps' or 'agg', got '{data_type}'")

    if len(code_list) != len(prefixes_list):
        raise ValueError(
            f"Length of code list ({len(code_list)}) must match length of prefixes list ({len(prefixes_list)})"
        )

    data_subdir = "post_selection" if data_type == "ps" else "aggregated"
    base_dir = os.path.join(DATA_DIR, data_subdir)
    result_dict = {}

    # Process each code-prefix pair
    for code_name, prefix in zip(code_list, prefixes_list):
        code_dir = os.path.join(base_dir, code_name)
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

                # Apply filter if provided
                if filter is not None:
                    # Check if all filter conditions are satisfied
                    filter_satisfied = True
                    for filter_param, filter_value in filter.items():
                        if (
                            filter_param not in params
                            or params[filter_param] != filter_value
                        ):
                            filter_satisfied = False
                            break

                    # Skip this file if filter is not satisfied
                    if not filter_satisfied:
                        continue

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

                # Check for duplicate keys
                if key in result_dict:
                    raise ValueError(
                        f"Duplicate key '{key}' generated. This occurs when the same prefix+by_dir combination appears multiple times."
                    )

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


# -----------------------------------------------------------------------------
# Split into two explicit APIs
# -----------------------------------------------------------------------------


def calculate_discrimination_metrics_from_raw(
    metric_values: np.ndarray,
    fails: np.ndarray,
    method: str = "all",
    ascending_confidence: bool = False,
) -> float | dict[str, float]:
    """
    Calculate discrimination metrics from raw per-sample data.

    Parameters
    ----------
    metric_values : 1D numpy array of float
        Metric value for every individual sample.
    fails : 1D numpy array of bool or int
        Indicator for whether each sample resulted in a failure (``True``/1) or
        success (``False``/0). Must have the same shape as ``metric_values``.
    method : {'auprc', 'auc-roc', 'cohen-d', 'lift', 'all'}, default 'all'
        Metric(s) to compute.
    ascending_confidence : bool, default False
        If *True*, lower metric values correspond to higher failure
        probability. The values are negated internally so that a higher score
        always indicates a higher chance of failure for the purpose of the
        calculations.

    Returns
    -------
    float | dict[str, float]
        • A single float for the specified *method*.
        • A dictionary with all metrics when *method* is ``'all'``.
    """

    metric_values = np.asarray(metric_values)
    fails = np.asarray(fails, dtype=bool)

    if metric_values.shape != fails.shape:
        raise ValueError("metric_values and fails must have identical shapes.")

    if ascending_confidence:
        metric_values = -metric_values

    y_true = fails.astype(int)
    y_score = metric_values

    total_fails = int(np.sum(fails))
    total_successes = int(len(fails) - total_fails)

    def _calc_auprc() -> float:
        if total_fails == 0 or total_successes == 0:
            return np.nan
        return average_precision_score(y_true, y_score)

    def _calc_auc_roc() -> float:
        if total_fails == 0 or total_successes == 0:
            return np.nan
        return roc_auc_score(y_true, y_score)

    def _calc_cohen_d() -> float:
        if total_fails < 2 or total_successes < 2:
            return np.nan
        fail_scores = metric_values[fails]
        success_scores = metric_values[~fails]
        mean_fail, mean_success = np.mean(fail_scores), np.mean(success_scores)
        var_fail = np.var(fail_scores, ddof=1)
        var_success = np.var(success_scores, ddof=1)
        s_pooled_sq = (
            (total_fails - 1) * var_fail + (total_successes - 1) * var_success
        ) / (total_fails + total_successes - 2)
        s_pooled_sq = max(s_pooled_sq, 0.0)
        s_pooled = np.sqrt(s_pooled_sq)
        if s_pooled == 0:
            return np.inf * np.sign(mean_fail - mean_success)
        return (mean_fail - mean_success) / s_pooled

    def _calc_lift() -> float:
        auprc_val = _calc_auprc()
        if np.isnan(auprc_val):
            return np.nan
        fail_rate = total_fails / (total_fails + total_successes)
        if fail_rate == 0:
            return np.inf
        return auprc_val / fail_rate

    if method == "all":
        return {
            "auprc": _calc_auprc(),
            "auc-roc": _calc_auc_roc(),
            "cohen-d": _calc_cohen_d(),
            "lift": _calc_lift(),
        }
    if method == "auprc":
        return _calc_auprc()
    if method == "auc-roc":
        return _calc_auc_roc()
    if method == "cohen-d":
        return _calc_cohen_d()
    if method == "lift":
        return _calc_lift()

    raise ValueError(f"Unknown method '{method}'")


def calculate_discrimination_metrics_from_agg(
    metric_values: np.ndarray,
    counts: np.ndarray,
    num_fails: np.ndarray,
    method: str = "all",
    ascending_confidence: bool = False,
) -> float | dict[str, float]:
    """
    Calculate discrimination metrics from aggregated sample data.

    Parameters
    ----------
    metric_values : 1D numpy array of float
        Unique metric values.
    counts : 1D numpy array of int
        Total number of samples corresponding to each ``metric_values`` entry.
    num_fails : 1D numpy array of int
        Number of failures corresponding to each ``metric_values`` entry.
    method : {'auprc', 'auc-roc', 'cohen-d', 'lift', 'all'}, default 'all'
        Metric(s) to compute.
    ascending_confidence : bool, default False
        If *True*, lower metric values correspond to higher failure probability
        and are negated internally before metric computation.

    Returns
    -------
    float | dict[str, float]
        • A single float for the specified *method*.
        • A dictionary with all metrics when *method* is ``'all'``.
    """

    metric_values = np.asarray(metric_values)
    counts = np.asarray(counts)
    num_fails = np.asarray(num_fails)

    if not (metric_values.shape == counts.shape == num_fails.shape):
        raise ValueError(
            "metric_values, counts, and num_fails must share the same shape."
        )

    if ascending_confidence:
        metric_values = -metric_values

    num_successes = counts - num_fails
    total_fails = int(np.sum(num_fails))
    total_successes = int(np.sum(num_successes))

    def _get_sklearn_inputs() -> (
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
    ):
        if total_fails == 0 or total_successes == 0:
            return None, None, None
        fail_mask = num_fails > 0
        success_mask = num_successes > 0
        y_true = np.concatenate(
            [
                np.ones(np.count_nonzero(fail_mask), dtype=int),
                np.zeros(np.count_nonzero(success_mask), dtype=int),
            ]
        )
        y_score = np.concatenate(
            [metric_values[fail_mask], metric_values[success_mask]]
        )
        sample_weight = np.concatenate(
            [num_fails[fail_mask], num_successes[success_mask]]
        )
        return y_true, y_score, sample_weight

    def _calc_auprc() -> float:
        y_true, y_score, sample_weight = _get_sklearn_inputs()
        if y_true is None:
            return np.nan
        return average_precision_score(y_true, y_score, sample_weight=sample_weight)

    def _calc_auc_roc() -> float:
        y_true, y_score, sample_weight = _get_sklearn_inputs()
        if y_true is None:
            return np.nan
        return roc_auc_score(y_true, y_score, sample_weight=sample_weight)

    def _calc_cohen_d() -> float:
        if total_fails < 2 or total_successes < 2:
            return np.nan
        mean_fail = float(np.dot(num_fails, metric_values) / total_fails)
        mean_success = float(np.dot(num_successes, metric_values) / total_successes)
        sum_sq_fail = np.dot(num_fails, metric_values**2)
        sum_sq_success = np.dot(num_successes, metric_values**2)
        var_fail = (sum_sq_fail - total_fails * mean_fail**2) / (total_fails - 1)
        var_success = (sum_sq_success - total_successes * mean_success**2) / (
            total_successes - 1
        )
        var_fail = max(var_fail, 0.0)
        var_success = max(var_success, 0.0)
        s_pooled_sq = (
            (total_fails - 1) * var_fail + (total_successes - 1) * var_success
        ) / (total_fails + total_successes - 2)
        s_pooled_sq = max(s_pooled_sq, 0.0)
        s_pooled = np.sqrt(s_pooled_sq)
        if s_pooled == 0:
            return np.inf * np.sign(mean_fail - mean_success)
        return (mean_fail - mean_success) / s_pooled

    def _calc_lift() -> float:
        auprc_val = _calc_auprc()
        if np.isnan(auprc_val):
            return np.nan
        if total_fails == 0 or total_successes == 0:
            return np.nan
        fail_rate = total_fails / (total_fails + total_successes)
        if fail_rate == 0:
            return np.inf
        return auprc_val / fail_rate

    if method == "all":
        return {
            "auprc": _calc_auprc(),
            "auc-roc": _calc_auc_roc(),
            "cohen-d": _calc_cohen_d(),
            "lift": _calc_lift(),
        }
    if method == "auprc":
        return _calc_auprc()
    if method == "auc-roc":
        return _calc_auc_roc()
    if method == "cohen-d":
        return _calc_cohen_d()
    if method == "lift":
        return _calc_lift()

    raise ValueError(f"Unknown method '{method}'")


##
# -----------------------------------------------------------------------------
# Histogram utilities
# -----------------------------------------------------------------------------


def plot_success_failure_histogram(
    df_agg: pd.DataFrame,
    *,
    bins: int = 500,
    ax: matplotlib.axes.Axes | None = None,
    colors: tuple[str, str] | None = None,
    alpha: float = 0.4,
    rescale_by_rate: bool = False,
    twin_y: bool = False,
    lower_trim_frac: float = 0.0,
    upper_trim_frac: float = 0.0,
) -> matplotlib.axes.Axes:
    """
    Plot overlaid success and failure histograms from an aggregated simulation
    DataFrame.

    The input ``df_agg`` must contain the following columns (case-sensitive):

    * ``count`` – total number of samples for the corresponding metric value.
    * ``num_fails`` – number of failed samples.

    If a column ``num_succs`` is absent it will be derived on the fly as
    ``count - num_fails``.

    Parameters
    ----------
    df_agg : pandas DataFrame
        Aggregated data with a *single-level* index whose values correspond to
        the confidence metric (e.g. *cluster_llr_norm_gap_1*).  Columns must at
        least include ``count`` and ``num_fails``.
    bins : int, default 500
        Number of histogram bins.
    ax : matplotlib.axes.Axes or None, optional
        Target axes.  A new figure and axes are created when *None*.
    colors : tuple of str, optional
        Two matplotlib/Seaborn-compatible colours ``(success, fail)``.  Defaults
        to the first two colours of the current palette.
    alpha : float, default 0.4
        Transparency for the filled histograms.
    rescale_by_rate : bool, default False
        If *True*, histogram heights are scaled so that the *area* under the
        success curve equals the overall success rate (``#succ / (#succ+#fail)``)
        and analogously for failures.  Internally this is achieved by plotting
        histogram *probability mass* (``stat='sum'``) instead of a density.
    twin_y : bool, default False
        Draw failures on a secondary y-axis (``ax.twinx()``) so that each
        histogram has an independent vertical scale.  Useful when success and
        failure counts differ by orders of magnitude.

    lower_trim_frac : float, default 0.0
        Fraction (0‒0.5) of the *total* mass to trim from the lower tail of the
        metric distribution.  Applied independently to successes and failures;
        the final retained range is the union of both trimmed intervals.

    upper_trim_frac : float, default 0.0
        Analogous fraction to trim from the upper tail.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the drawn histograms.
    """

    import seaborn as sns  # local import to avoid mandatory dependency at import time

    if "count" not in df_agg.columns or "num_fails" not in df_agg.columns:
        raise ValueError("df_agg must contain 'count' and 'num_fails' columns.")

    # Work on a copy and ensure index ordering is ascending
    df_agg = df_agg.sort_index().copy()

    # Pre-compute default display range (may be tightened later)
    keep_min: float = float(df_agg.index.min())
    keep_max: float = float(df_agg.index.max())

    # ------------------------------------------------------------------
    # Optional tail trimming (lower_trim_frac & upper_trim_frac)
    # ------------------------------------------------------------------

    if not (0.0 <= lower_trim_frac < 0.5 and 0.0 <= upper_trim_frac < 0.5):
        raise ValueError("lower_trim_frac and upper_trim_frac must be in [0, 0.5).")

    if lower_trim_frac > 0.0 or upper_trim_frac > 0.0:

        def _weighted_quantile(
            values: np.ndarray, weights: np.ndarray, q: float
        ) -> float:
            """Return weighted quantile of *values* at cumulative probability *q*."""
            if len(values) == 0:
                return np.nan
            sorter = np.argsort(values)
            values_sorted = values[sorter]
            weights_sorted = weights[sorter]
            cumulative = np.cumsum(weights_sorted)
            cutoff = q * cumulative[-1]
            idx = np.searchsorted(cumulative, cutoff, side="left")
            return float(values_sorted[min(idx, len(values_sorted) - 1)])

        metrics = df_agg.index.to_numpy(dtype=float)

        succ_weights = (
            df_agg["num_succs"].to_numpy()
            if "num_succs" in df_agg.columns
            else (df_agg["count"] - df_agg["num_fails"]).to_numpy()
        )
        fail_weights = df_agg["num_fails"].to_numpy()

        # Ensure num_succs column exists for weights if not already present
        if "num_succs" not in df_agg.columns:
            df_agg["num_succs"] = succ_weights

        # --- Success range ---
        if succ_weights.sum() > 0:
            succ_lower = _weighted_quantile(metrics, succ_weights, lower_trim_frac)
            succ_upper = _weighted_quantile(metrics, succ_weights, 1 - upper_trim_frac)
        else:  # no successes
            succ_lower, succ_upper = metrics.min(), metrics.max()

        # --- Failure range ---
        if fail_weights.sum() > 0:
            fail_lower = _weighted_quantile(metrics, fail_weights, lower_trim_frac)
            fail_upper = _weighted_quantile(metrics, fail_weights, 1 - upper_trim_frac)
        else:  # no failures
            fail_lower, fail_upper = metrics.min(), metrics.max()

        # Union of intervals; *do not* drop data – limits will be applied post-plot
        keep_min = min(succ_lower, fail_lower)
        keep_max = max(succ_upper, fail_upper)

    # Ensure num_succs column exists
    if "num_succs" not in df_agg.columns:
        df_agg["num_succs"] = df_agg["count"] - df_agg["num_fails"]

    # Colours – fall back to Seaborn default palette if none supplied
    if colors is None:
        palette = sns.color_palette()
        success_col, fail_col = palette[0], palette[1]
    else:
        if len(colors) != 2:
            raise ValueError("'colors' must be a tuple of exactly two colour strings.")
        success_col, fail_col = colors

    # Axes handling
    if ax is None:
        fig, ax = plt.subplots()

    metric_name = df_agg.index.name or "metric"
    bin_range = (df_agg.index.min(), df_agg.index.max())

    df_reset = df_agg.reset_index()
    if metric_name not in df_reset.columns:
        # Happens when original index had no name; pandas uses 'index' by default
        df_reset = df_reset.rename(columns={df_reset.columns[0]: metric_name})

    # Primary (success) and optional secondary (failure) axes
    succ_ax = ax
    fail_ax = succ_ax.twinx() if twin_y else succ_ax

    # ------------------------------------------------------------------
    # Determine weights/stat depending on rescaling option
    # ------------------------------------------------------------------

    if rescale_by_rate:
        total_samples = int(df_agg["count"].sum())
        df_reset["succ_weight"] = df_reset["num_succs"] / total_samples
        df_reset["fail_weight"] = df_reset["num_fails"] / total_samples
        succ_weight_col = "succ_weight"
        fail_weight_col = "fail_weight"
        stat_mode = "count"
    else:
        succ_weight_col = "num_succs"
        fail_weight_col = "num_fails"
        stat_mode = "density"

    # Success histogram – filled then outline for visibility
    sns.histplot(
        df_reset,
        x=metric_name,
        weights=succ_weight_col,
        bins=bins,
        stat=stat_mode,
        binrange=bin_range,
        alpha=alpha,
        color=success_col,
        ax=succ_ax,
        zorder=1,
    )
    sns.histplot(
        df_reset,
        x=metric_name,
        weights=succ_weight_col,
        bins=bins,
        stat=stat_mode,
        binrange=bin_range,
        fill=False,
        element="poly",
        linewidth=1.5,
        color=success_col,
        label="success",
        ax=succ_ax,
        zorder=3,
    )

    # Failure histogram – filled then outline
    sns.histplot(
        df_reset,
        x=metric_name,
        weights=fail_weight_col,
        bins=bins,
        stat=stat_mode,
        binrange=bin_range,
        alpha=alpha,
        color=fail_col,
        ax=fail_ax,
        zorder=2,
    )
    sns.histplot(
        df_reset,
        x=metric_name,
        weights=fail_weight_col,
        bins=bins,
        stat=stat_mode,
        binrange=bin_range,
        fill=False,
        element="poly",
        linewidth=1.5,
        color=fail_col,
        label="fail",
        ax=fail_ax,
        zorder=4,
    )

    # Axis labels and legend
    succ_ax.set_xlabel(metric_name)

    # ------------------------------------------------------------------
    # Customise y-axes when twin_y is enabled
    # ------------------------------------------------------------------
    if twin_y:
        # Label axes
        succ_ax.set_ylabel("Density (success)")
        fail_ax.set_ylabel("Density (fail)")

        # Match axis and tick colours to the corresponding histogram colours
        succ_ax.tick_params(axis="y", colors=success_col)
        fail_ax.tick_params(axis="y", colors=fail_col)

        # Set spine and label colours for better visual association
        succ_ax.spines["left"].set_color(success_col)
        fail_ax.spines["right"].set_color(fail_col)
        succ_ax.yaxis.label.set_color(success_col)
        fail_ax.yaxis.label.set_color(fail_col)

    # Merge legends from both axes when twin_y is used
    if twin_y:
        handles1, labels1 = succ_ax.get_legend_handles_labels()
        handles2, labels2 = fail_ax.get_legend_handles_labels()
        succ_ax.legend(handles1 + handles2, labels1 + labels2)
    else:
        succ_ax.legend()

    # ------------------------------------------------------------------
    # Apply x-axis trimming, if requested, *after* plotting so overall counts
    # and binning remain unchanged.
    # ------------------------------------------------------------------

    if lower_trim_frac > 0.0 or upper_trim_frac > 0.0:
        succ_ax.set_xlim(keep_min, keep_max)

    return succ_ax


def get_required_abort_rate(
    df_ps_sng: pd.DataFrame,
    target_ler: float | None = None,
    target_suppression: float | None = None,
) -> dict[str, float]:
    """
    Calculate the required abort rate for a given target LER or suppression rate.

    Parameters
    ----------
    df_ps_sng : pd.DataFrame
        Post-selection DataFrame for a particular parameter combination and metric.
    target_ler : float, optional
        Target LER.
    target_suppression : float, optional
        Target suppression rate.

    """
    assert sum(prm is not None for prm in [target_ler, target_suppression]) == 1

    shots = int(df_ps_sng["count"].max())

    # Calculate target LER
    p_fail_org = float(df_ps_sng.loc[df_ps_sng["p_abort"].idxmin(), "p_fail"])
    if target_ler is None:
        target_ler = p_fail_org * target_suppression

    # Calculate error bounds for p_abort
    # Use p_fail - delta_p_fail for lower bound (more conservative threshold)
    p_fail_lower = df_ps_sng["p_fail"] - df_ps_sng["delta_p_fail"]
    valid_rows_lower = df_ps_sng[p_fail_lower < target_ler]
    p_abort_low = (
        float(valid_rows_lower["p_abort"].min()) if not valid_rows_lower.empty else 0.0
    )

    # Use p_fail + delta_p_fail for upper bound (less conservative threshold)
    p_fail_upper = df_ps_sng["p_fail"] + df_ps_sng["delta_p_fail"]
    valid_rows_upper = df_ps_sng[p_fail_upper < target_ler]
    p_abort_upp = (
        float(valid_rows_upper["p_abort"].min()) if not valid_rows_upper.empty else 1.0
    )

    # Find p_abort using central p_fail estimate
    valid_rows = df_ps_sng[df_ps_sng["p_fail"] < target_ler]
    if not valid_rows.empty:
        p_abort = float(valid_rows["p_abort"].min())
    elif p_abort_upp == 1:
        p_abort = 1.0
    elif p_abort_low == 0:
        p_abort = 0.0
    else:
        p_abort = (p_abort_low + p_abort_upp) / 2

    results = {
        "p_abort": p_abort,
        "p_abort_low": p_abort_low,
        "p_abort_upp": p_abort_upp,
        "shots": shots,
        "p_fail_org": p_fail_org,
    }

    return results


def get_confint(
    counts: np.ndarray, shots: np.ndarray, alpha: float = 0.05, method: str = "wilson"
) -> Tuple[np.ndarray, np.ndarray]:
    p_lower, p_upper = proportion_confint(counts, shots, alpha=alpha, method=method)
    p = (p_lower + p_upper) / 2
    delta_p = p_upper - p

    return p, delta_p
