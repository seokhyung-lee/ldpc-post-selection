import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed

from .data_aggregation import aggregate_data
from .data_post_processing import get_df_ps

DATA_DIR = Path(__file__).parent.parent.parent / "data"


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


def get_raw_data_subdirectories(
    data_dir: str, dataset_type: str = "surface"
) -> List[Dict[str, str]]:
    """
    Get all subdirectories in the raw data directory with parameter info.

    Parameters
    ----------
    data_dir : str
        Path to the raw data directory.
    dataset_type : str
        Type of dataset ('surface', 'bb', 'hgp').

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries containing 'path' and 'param_combo_str' for each subdirectory.
    """
    subdirs = []
    if not os.path.exists(data_dir):
        return subdirs

    if dataset_type == "hgp":
        # Nested directory structure: {circuit_folder}/{circuit_name}
        for circuit_folder in os.listdir(data_dir):
            circuit_folder_path = os.path.join(data_dir, circuit_folder)
            if os.path.isdir(circuit_folder_path):
                for circuit_name in os.listdir(circuit_folder_path):
                    circuit_path = os.path.join(circuit_folder_path, circuit_name)
                    if os.path.isdir(circuit_path):
                        # Create parameter combination string
                        param_combo_str = f"{circuit_folder}_{circuit_name}"
                        subdirs.append(
                            {"path": circuit_path, "param_combo_str": param_combo_str}
                        )

    else:
        # Single-level directory structure: d{d}_T{T}_p{p} or n{n}_T{T}_p{p}
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                subdirs.append({"path": item_path, "param_combo_str": item})

    return sorted(subdirs, key=lambda x: x["param_combo_str"])


def process_single_subdirectory(
    subdir_info: Dict[str, str],
    dataset_name: str,
    method_name: str,
    by: str,
    order: Optional[float],
    decimals: int,
    ascending_confidence: bool,
    verbose: bool,
    eval_windows: Optional[tuple[int, int]] = None,
    adj_matrix: Optional[np.ndarray] = None,
) -> Dict[str, str | bool]:
    """
    Process a single subdirectory for data aggregation and post-selection.

    Parameters
    ----------
    subdir_info : Dict[str, str]
        Dictionary containing 'path' and 'param_combo_str' for the subdirectory.
    dataset_name : str
        Name of the dataset.
    method_name : str
        Method name for saving data.
    by : str
        Aggregation method.
    order : Optional[float]
        Norm order for norm-based methods.
    decimals : int
        Number of decimal places to round to.
    ascending_confidence : bool
        Whether to use ascending confidence.
    verbose : bool
        Whether to print detailed progress information.
    eval_windows : Optional[tuple[int, int]], optional
        If provided, only consider windows from init_eval_window to final_eval_window for sliding window metrics.
    adj_matrix : Optional[np.ndarray], optional
        Adjacency matrix for cluster labeling. Required for committed cluster metrics in sliding window format.

    Returns
    -------
    Dict[str, str | bool]
        Dictionary containing processing results with keys:
        - 'param_combo_str': str
        - 'success': bool
        - 'message': str
        - 'agg_reused': bool
        - 'ps_reused': bool
    """
    subdir_path = subdir_info["path"]
    param_combo_str = subdir_info["param_combo_str"]

    # Skip empty subdirectories
    if not os.listdir(subdir_path):
        return {
            "param_combo_str": param_combo_str,
            "success": False,
            "message": "Empty subdirectory",
            "agg_reused": False,
            "ps_reused": False,
        }

    if verbose:
        print(f"\nProcessing subdirectory: {param_combo_str}")
        print(f"  Path: {subdir_path}")

    try:
        # Load existing aggregated data if available
        existing_df_agg = load_existing_df(
            dataset_name, method_name, param_combo_str, "aggregated"
        )

        # Load priors
        try:
            priors_path = os.path.join(subdir_path, "priors.npy")
            priors = np.load(priors_path)
        except FileNotFoundError:
            priors = None  # pass for now, but will raise error later if needed

        # Load adjacency matrix if not provided and needed for committed cluster metrics
        if adj_matrix is None and "committed_cluster_" in by:
            try:
                adj_matrix_path = os.path.join(subdir_path, "adj_matrix.npy")
                adj_matrix = np.load(adj_matrix_path)
            except FileNotFoundError:
                adj_matrix = None  # Will raise error later if needed

        # Call aggregate_data for this specific subdirectory
        try:
            # Use the original by parameter for all metrics
            aggregation_by = by

            df_agg, agg_reused = aggregate_data(
                subdir_path,  # Process individual subdirectory
                by=aggregation_by,
                norm_order=order,
                decimals=decimals,
                ascending_confidence=ascending_confidence,
                df_existing=existing_df_agg,
                verbose=verbose,
                priors=priors,
                eval_windows=eval_windows,
                adj_matrix=adj_matrix,
            )
        except FileNotFoundError:
            return {
                "param_combo_str": param_combo_str,
                "success": False,
                "message": "FileNotFoundError in aggregate_data",
                "agg_reused": False,
                "ps_reused": False,
            }

        if not agg_reused:
            save_df(df_agg, dataset_name, method_name, param_combo_str, "aggregated")

        # Load existing post-selection data if available
        existing_df_ps = load_existing_df(
            dataset_name, method_name, param_combo_str, "post_selection"
        )

        # Calculate post-selection data
        df_ps, ps_reused = get_df_ps(
            df_agg=df_agg,
            ascending_confidence=ascending_confidence,
            existing_df_ps=existing_df_ps,
        )

        if not ps_reused:
            # Save post-selection data
            save_df(
                df_ps,
                dataset_name,
                method_name,
                param_combo_str,
                "post_selection",
            )

        return {
            "param_combo_str": param_combo_str,
            "success": True,
            "message": "Processed successfully",
            "agg_reused": agg_reused,
            "ps_reused": ps_reused,
        }

    except Exception as e:
        return {
            "param_combo_str": param_combo_str,
            "success": False,
            "message": f"Error: {str(e)}",
            "agg_reused": False,
            "ps_reused": False,
        }


# def determine_decimals(by: str) -> int:
#     if by == "detector_density" or "norm_frac" in by:
#         return 4
#     else:
#         return 2


def process_dataset(
    data_dir: str,
    dataset_name: str,
    ascending_confidences: Dict[str, bool],
    orders: Optional[List[float]] = None,
    decimals: int | Callable[[str], int] = 2,
    verbose: bool = False,
    dataset_type: Optional[str] = None,
    eval_windows: Optional[tuple[int, int]] = None,
    adj_matrix: Optional[np.ndarray] = None,
) -> None:
    """
    Process a dataset with given parameters.

    Parameters
    ----------
    data_dir : str
        Path to the raw data directory.
    dataset_name : str
        Name of the dataset (e.g., 'surface', 'surface_matching', 'bb', 'hgp').
        Also used to infer dataset_type if dataset_type is not provided.
    ascending_confidences : Dict[str, bool]
        Dictionary mapping aggregation methods to their ascending confidence settings.
    orders : Optional[List[float]], optional
        List of norm orders to use for norm-based methods.
        If None, defaults to [0.5, 1, 2, np.inf]. Only used for norm-based methods.
    decimals : int, optional
        Number of decimal places to round to. Defaults to 2.
    verbose : bool, optional
        Whether to print detailed progress information. Defaults to False.
    dataset_type : Optional[str], optional
        Type of dataset ('surface', 'bb', 'hgp'). If None, inferred from dataset_name.
        This determines how directory structure is interpreted.
    eval_windows : Optional[tuple[int, int]], optional
        If provided, only consider windows from init_eval_window to final_eval_window for sliding window metrics.
    adj_matrix : Optional[np.ndarray], optional
        Adjacency matrix for cluster labeling. Required for committed cluster metrics in sliding window format.
        If None, will attempt to load from each subdirectory's adj_matrix.npy file when needed.
    """
    # Infer dataset_type from dataset_name if not provided
    if dataset_type is None:
        dataset_type = dataset_name.lower()

    if verbose:
        print(f"Using dataset_type: {dataset_type}")

    # Set default orders if not provided
    for by, ascending_confidence in ascending_confidences.items():
        print(
            f"\nAggregating data for {by} with ascending_confidence={ascending_confidence}..."
        )

        if "norm" in by or "norm_frac" in by:
            assert orders is not None, "Norm-based methods require orders"
            norm_orders = orders
        else:
            norm_orders = [None]

        if callable(decimals):
            decimals_by = decimals(by)
        else:
            decimals_by = decimals

        for order in norm_orders:
            if order is not None:
                print(f"norm_order = {order}")
                method_name = f"{by}_{order}"
            else:
                method_name = by

            print(f"Processing method: {method_name}")

            # Get all raw data subdirectories (each corresponds to a parameter combination)
            subdirs_info = get_raw_data_subdirectories(data_dir, dataset_type)

            if not subdirs_info:
                print("No subdirectories found to process.")
                continue

            print(f"Processing {len(subdirs_info)} subdirectories...")

            # Process subdirectories in parallel using joblib
            results = [
                process_single_subdirectory(
                    subdir_info,
                    dataset_name,
                    method_name,
                    by,
                    order,
                    decimals_by,
                    ascending_confidence,
                    verbose,
                    eval_windows,
                    adj_matrix,
                )
                for subdir_info in subdirs_info
            ]

            # Process and report results
            successful_count = 0
            failed_count = 0
            agg_reused_count = 0
            ps_reused_count = 0

            for result in results:
                param_combo_str = result["param_combo_str"]

                if result["success"]:
                    successful_count += 1
                    if not verbose:
                        print(f"✓ {param_combo_str}")

                    if result["agg_reused"]:
                        agg_reused_count += 1
                        if verbose:
                            print(f"    Aggregation skipped for {param_combo_str}")

                    if result["ps_reused"]:
                        ps_reused_count += 1
                        if verbose:
                            print(f"    Post-selection skipped for {param_combo_str}")
                    elif verbose:
                        print(f"    Post-selection successful for {param_combo_str}")
                else:
                    failed_count += 1
                    print(f"✗ {param_combo_str}: {result['message']}")

            print(f"  Successful: {successful_count}")
            print(f"  Failed: {failed_count}")
            print(f"  Aggregation reused: {agg_reused_count}")
            print(f"  Post-selection reused: {ps_reused_count}")

        print("=============")
