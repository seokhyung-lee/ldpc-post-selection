import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from typing import Tuple


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


# Helper function to apply the confidence interval calculation row-wise
def _apply_ci_vectorized(
    N_series: pd.Series, k_series: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Applies the 'calculate_confidence_interval' function to pairs of (N, k) values.
    'calculate_confidence_interval' must be available in the scope and return (value, delta).

    Parameters
    ----------
    N_series : pd.Series
        Series representing the total number of trials (N).
    k_series : pd.Series
        Series representing the number of successes/failures (k).

    Returns
    -------
    p_values : pd.Series
        Series of calculated probabilities (p).
    delta_p_values : pd.Series
        Series of calculated confidence interval widths (delta_p).
    """
    p_output = pd.Series(np.nan, index=N_series.index)
    delta_p_output = pd.Series(np.nan, index=N_series.index)

    # Mask for valid (non-NaN) inputs
    valid_mask = N_series.notna() & k_series.notna()

    if valid_mask.any():
        # Use .loc with the boolean mask to select valid rows for processing
        # This ensures correct alignment if the index is complex (e.g., MultiIndex)
        df_for_ci = pd.DataFrame(
            {"N_val": N_series[valid_mask], "k_val": k_series[valid_mask]}
        )

        if not df_for_ci.empty:
            ci_results = df_for_ci.apply(
                lambda row: calculate_confidence_interval(row["N_val"], row["k_val"]),
                axis=1,
            )

            # Unpack the series of tuples into two lists, then assign to output Series
            # This assumes calculate_confidence_interval always returns a pair (value, delta)
            if not ci_results.empty:
                p_list = [res[0] for res in ci_results]
                delta_list = [res[1] for res in ci_results]

                p_output[valid_mask] = p_list
                delta_p_output[valid_mask] = delta_list

    return p_output, delta_p_output


def get_df_ps(
    df_agg: pd.DataFrame,
    ascending_confidence: bool = True,
    existing_df_ps: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, bool]:
    """
    Calculate post-selection probabilities and statistics from aggregated data
    using vectorized Pandas operations.

    Parameters
    ----------
    df_agg : pd.DataFrame
        Aggregated data with columns like 'count', 'num_fails'.
        Expected convergence columns: 'num_converged', 'num_converged_fails'.
        The index should be the binning variable (e.g., 'pred_llr').
    ascending_confidence : bool, optional
        If True (default), data is processed assuming higher index values mean
        higher confidence. Cumulative sums are performed from "high confidence"
        to "low confidence".
        If False, data is processed in its natural index order.
    existing_df_ps : pd.DataFrame or None, optional
        If provided, existing post-selection results that can be partially reused.
        If the total count matches between existing results and new data,
        the existing results will be reused instead of recalculating.

    Returns
    -------
    df_ps : pd.DataFrame
        DataFrame containing post-selection probabilities (p_fail, p_abort)
        and their confidence intervals (delta_p_fail, delta_p_abort),
        both considering and ignoring convergence.
        Also includes cumulative counts. The index matches the input df_agg index.
    reused_existing : bool
        True if existing data was reused, False if new calculations were performed.
    """
    # If no existing df_ps provided, calculate everything from scratch
    if existing_df_ps is None:
        return _calculate_df_ps_full(df_agg, ascending_confidence), False

    # Check if existing data can be reused
    if _can_reuse_existing_data(df_agg, existing_df_ps):
        # Return existing data filtered to current index
        return existing_df_ps.loc[df_agg.index].copy(), True

    # Calculate everything from scratch
    return _calculate_df_ps_full(df_agg, ascending_confidence), False


def _can_reuse_existing_data(
    df_agg: pd.DataFrame, existing_df_ps: pd.DataFrame
) -> bool:
    """
    Check if existing df_ps data can be reused.

    Parameters
    ----------
    df_agg : pd.DataFrame
        New aggregated data.
    existing_df_ps : pd.DataFrame
        Existing post-selection results.

    Returns
    -------
    bool
        True if existing data can be reused, False otherwise.
    """
    try:
        # Check if all indices in df_agg exist in existing_df_ps
        if not df_agg.index.isin(existing_df_ps.index).all():
            return False

        # Get total counts
        new_total_count = df_agg["count"].sum()

        # For existing data, the maximum count should be the total since it's cumulative
        existing_total_count = existing_df_ps.loc[df_agg.index, "count"].max()

        return new_total_count == existing_total_count

    except (KeyError, IndexError):
        return False


def _calculate_df_ps_full(
    df_agg: pd.DataFrame, ascending_confidence: bool
) -> pd.DataFrame:
    """
    Calculate complete post-selection probabilities for all data in df_agg.

    Parameters
    ----------
    df_agg : pd.DataFrame
        Aggregated data with columns like 'count', 'num_fails'.
    ascending_confidence : bool
        Processing order for confidence intervals.

    Returns
    -------
    pd.DataFrame
        Complete post-selection results.
    """
    df_ps = pd.DataFrame(index=df_agg.index)

    # Total shots across all data
    total_shots = df_agg["count"].sum()

    # --- Ignoring convergence ---
    # Simple cumulative counts and failures based on ascending_confidence
    if ascending_confidence:
        # Reverse, cumsum, reverse back to get "cumulative from high confidence"
        counts_cum = df_agg["count"][::-1].cumsum()[::-1]
        num_fails_cum = df_agg["num_fails"][::-1].cumsum()[::-1]
    else:
        # Normal cumulative sum
        counts_cum = df_agg["count"].cumsum()
        num_fails_cum = df_agg["num_fails"].cumsum()

    # Calculate p_fail and its confidence interval
    df_ps["p_fail"], df_ps["delta_p_fail"] = _apply_ci_vectorized(
        counts_cum, num_fails_cum
    )

    # Calculate p_accept and its confidence interval, then p_abort
    total_shots_series = pd.Series(total_shots, index=df_agg.index)
    p_acc, delta_p_acc = _apply_ci_vectorized(total_shots_series, counts_cum)
    df_ps["p_abort"] = 1.0 - p_acc
    df_ps["delta_p_abort"] = delta_p_acc  # Width is same for p and 1-p

    # Store cumulative counts for reference
    df_ps["count"] = counts_cum
    df_ps["num_fails"] = num_fails_cum

    # --- Treating convergence = confident ---
    if "num_converged" in df_agg.columns and "num_converged_fails" in df_agg.columns:
        # Base series for converged calculations (non-converged part)
        base_counts_conv = df_agg["count"] - df_agg["num_converged"]
        base_num_fails_conv = df_agg["num_fails"] - df_agg["num_converged_fails"]

        # Total converged items across all data
        total_converged = df_agg["num_converged"].sum()
        total_converged_fails = df_agg["num_converged_fails"].sum()

        # Add converged totals to the "starting" element based on processing order
        adj_counts_conv = base_counts_conv.copy()
        adj_num_fails_conv = base_num_fails_conv.copy()

        if ascending_confidence:
            # Add to the last row (which gets processed first when reversed)
            adj_counts_conv.iloc[-1] += total_converged
            adj_num_fails_conv.iloc[-1] += total_converged_fails
        else:
            # Add to the first row (which gets processed first)
            adj_counts_conv.iloc[0] += total_converged
            adj_num_fails_conv.iloc[0] += total_converged_fails

        # Cumulative sums for converged case
        if ascending_confidence:
            counts_conv_cum = adj_counts_conv[::-1].cumsum()[::-1]
            num_fails_conv_cum = adj_num_fails_conv[::-1].cumsum()[::-1]
        else:
            counts_conv_cum = adj_counts_conv.cumsum()
            num_fails_conv_cum = adj_num_fails_conv.cumsum()

        # Calculate p_fail_conv and its CI
        df_ps["p_fail_conv"], df_ps["delta_p_fail_conv"] = _apply_ci_vectorized(
            counts_conv_cum, num_fails_conv_cum
        )

        # Calculate p_accept_conv and its CI, then p_abort_conv
        p_acc_conv, delta_p_acc_conv = _apply_ci_vectorized(
            total_shots_series, counts_conv_cum
        )
        df_ps["p_abort_conv"] = 1.0 - p_acc_conv
        df_ps["delta_p_abort_conv"] = delta_p_acc_conv

        # Store cumulative converged counts
        df_ps["count_conv"] = counts_conv_cum
        df_ps["num_fails_conv"] = num_fails_conv_cum
    else:
        # If convergence columns are not available, fill relevant _conv columns with NaNs
        conv_cols = [
            "p_fail_conv",
            "delta_p_fail_conv",
            "p_abort_conv",
            "delta_p_abort_conv",
            "count_conv",
            "num_fails_conv",
        ]
        for col in conv_cols:
            df_ps[col] = np.nan

    return df_ps
