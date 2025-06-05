import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulations.analysis.data_aggregation import aggregate_data
from simulations.analysis.data_post_processing import get_df_ps


def load_existing_df(
    dataset: str, method: str, param_combo_str: str, data_type: str = "aggregated"
) -> object:
    """
    Load existing DataFrame for a specific parameter combination.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'surface', 'surface_matching', 'bb').
    method : str
        Method name (e.g., 'pred_llr', 'cluster_size_norm_0.5').
    param_combo_str : str
        Parameter combination string (e.g., 'd13_T13_p0.001', 'n144_T12_p0.001').
    data_type : str
        Type of data ('aggregated' or 'post_selection').

    Returns
    -------
    DataFrame or None
        Existing DataFrame if found, None otherwise.
    """
    if data_type == "aggregated":
        base_dir = "simulations/data/aggregated"
    elif data_type == "post_selection":
        base_dir = "simulations/data/post_selection"
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    file_path = os.path.join(base_dir, dataset, method, f"{param_combo_str}.pkl")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def save_df(
    df: object,
    dataset: str,
    method: str,
    param_combo_str: str,
    data_type: str = "aggregated",
) -> None:
    """
    Save DataFrame for a specific parameter combination.

    Parameters
    ----------
    df : DataFrame
        DataFrame to save.
    dataset : str
        Dataset name (e.g., 'surface', 'surface_matching', 'bb').
    method : str
        Method name (e.g., 'pred_llr', 'cluster_size_norm_0.5').
    param_combo_str : str
        Parameter combination string (e.g., 'd13_T13_p0.001', 'n144_T12_p0.001').
    data_type : str
        Type of data ('aggregated' or 'post_selection').
    """
    if data_type == "aggregated":
        base_dir = "simulations/data/aggregated"
    elif data_type == "post_selection":
        base_dir = "simulations/data/post_selection"
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    output_dir = os.path.join(base_dir, dataset, method)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{param_combo_str}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(df, f)


def get_raw_data_subdirectories(data_dir: str) -> List[str]:
    """
    Get all subdirectories in the raw data directory.

    Parameters
    ----------
    data_dir : str
        Path to the raw data directory.

    Returns
    -------
    List[str]
        List of subdirectory paths.
    """
    subdirs = []
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item_path)
    return sorted(subdirs)


def process_dataset(
    data_dir: str,
    dataset_name: str,
    ascending_confidences: Dict[str, bool],
    orders: Optional[List[float]] = None,
    num_hist_bins: int = 10000,
    verbose: bool = False,
) -> None:
    """
    Process a dataset with given parameters.

    Parameters
    ----------
    data_dir : str
        Path to the raw data directory.
    dataset_name : str
        Name of the dataset (e.g., 'surface', 'surface_matching', 'bb').
    ascending_confidences : Dict[str, bool]
        Dictionary mapping aggregation methods to their ascending confidence settings.
    orders : Optional[List[float]], optional
        List of norm orders to use for norm-based methods.
        If None, defaults to [0.5, 1, 2, np.inf]. Only used for norm-based methods.
    num_hist_bins : int, optional
        Number of histogram bins. Defaults to 10000.
    verbose : bool, optional
        Whether to print detailed progress information. Defaults to False.
    """
    # Set default orders if not provided
    for by, ascending_confidence in ascending_confidences.items():
        print(
            f"\nAggregating data for {by} with ascending_confidence={ascending_confidence}..."
        )

        if "norm" in by:
            assert orders is not None, "Norm-based methods require orders"
            norm_orders = orders
        else:
            norm_orders = [None]

        for order in norm_orders:
            if order is not None:
                print(f"norm_order = {order}")
                method_name = f"{by}_{order}"
            else:
                method_name = by

            print(f"Processing method: {method_name}")

            # Get all raw data subdirectories (each corresponds to a parameter combination)
            subdirs = get_raw_data_subdirectories(data_dir)

            for subdir in subdirs:
                subdir_name = os.path.basename(subdir)

                # Skip empty subdirectories
                if not os.listdir(subdir):
                    print(f"Skipping empty subdirectory: {subdir_name}")
                    continue

                print(f"\nProcessing subdirectory: {subdir_name}")

                # Extract parameter combination string from subdirectory name
                param_combo_str = subdir_name

                # Load existing aggregated data if available
                existing_df_agg = load_existing_df(
                    dataset_name, method_name, param_combo_str, "aggregated"
                )

                # Call aggregate_data for this specific subdirectory
                df_agg, reused = aggregate_data(
                    subdir,  # Process individual subdirectory
                    by=by,
                    norm_order=order,
                    num_hist_bins=num_hist_bins,
                    ascending_confidence=ascending_confidence,
                    df_existing=existing_df_agg,
                    verbose=verbose,
                )
                if reused:
                    print("    Aggregation skipped")
                else:
                    save_df(
                        df_agg, dataset_name, method_name, param_combo_str, "aggregated"
                    )

                # Load existing post-selection data if available
                existing_df_ps = load_existing_df(
                    dataset_name, method_name, param_combo_str, "post_selection"
                )

                # Calculate post-selection data
                df_ps, reused = get_df_ps(
                    df_agg=df_agg,
                    ascending_confidence=ascending_confidence,
                    existing_df_ps=existing_df_ps,
                )
                if reused:
                    print("    Post-selection skipped")
                else:
                    print("\nPost-selection successful")
                    # Save post-selection data
                    save_df(
                        df_ps,
                        dataset_name,
                        method_name,
                        param_combo_str,
                        "post_selection",
                    )

        print("=============")
