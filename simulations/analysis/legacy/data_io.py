import glob
import os
import pickle
import re
import time
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import from simulation utils
from simulations.utils.simulation_utils import get_existing_shots


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


def read_pickle(path: str) -> Any:
    """
    Read data from a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    Any
        The data loaded from the pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path: str, data: Any):
    """
    Write data to a pickle file.

    Parameters
    ----------
    path : str
        Path where to save the pickle file.
    data : Any
        The data to be saved.
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)


def find_simulation_files(
    code_param: int,
    T: int,
    p: float,
    code_type: str = "bb",
    data_dir: str = "../data/bb_circuit_iter30_minsum_lsd0",
    verbose: bool = False,
) -> List[str]:
    """
    Find simulation batch directories (`batch_{idx}`) for either BB or Surface codes
    within the specific subdirectory for a (code_param, T, p) tuple.
    Each valid batch directory must contain "scalars.feather" and other required files.

    Parameters
    ----------
    code_param : int
        For BB codes: number of qubits (n). For Surface codes: code distance (d).
    T : int
        Number of rounds
    p : float
        Physical error rate
    code_type : str, optional
        Type of code - "bb" for BB codes or "surface" for surface codes. Defaults to "bb".
    data_dir : str
        Base directory path containing the subdirectories.
    verbose : bool, optional
        Whether to print the number of found valid batch directories. Defaults to False.

    Returns
    -------
    batch_dir_paths : list of str
        List of paths to valid batch directories.

    Raises
    ------
    FileNotFoundError
        If the base data directory or the specific subdirectory does not exist.
    ValueError
        If no valid `batch_{idx}` directories (containing all required files)
        are found within the subdirectory, or if code_type is not supported.
    """
    if code_type not in ["bb", "surface"]:
        raise ValueError(
            f"Unsupported code_type: {code_type}. Must be 'bb' or 'surface'."
        )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Base data directory not found: {data_dir}")

    # Construct the specific subdirectory path
    if code_type == "bb":
        sub_dirname = f"n{code_param}_T{T}_p{p}"
        param_name = "n"
    else:  # surface
        sub_dirname = f"d{code_param}_T{T}_p{p}"
        param_name = "d"

    sub_dir_path = os.path.join(data_dir, sub_dirname)

    if not os.path.isdir(sub_dir_path):
        raise FileNotFoundError(
            f"Subdirectory not found for {param_name}={code_param}, T={T}, p={p}: {sub_dir_path}"
        )

    # Search for batch_* subdirectories within the subdirectory
    batch_dir_pattern = os.path.join(sub_dir_path, "batch_*")
    potential_batch_dirs = glob.glob(batch_dir_pattern)
    potential_batch_dirs = [d for d in potential_batch_dirs if os.path.isdir(d)]

    if not potential_batch_dirs:
        raise ValueError(f"No batch directories (batch_*) found in {sub_dir_path}.")

    valid_batch_dir_paths = []

    # For BB codes, all files are required. For surface codes, only scalars.feather is strictly required
    if code_type == "bb":
        required_files = [
            "scalars.feather",
            "cluster_sizes.npy",
            "cluster_llrs.npy",
            "offsets.npy",
        ]
    else:  # surface
        required_files = ["scalars.feather"]

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


def _get_values_for_binning_from_batch(
    batch_dir_path: str, by: str, norm_order: float | None, verbose: bool = False
) -> Tuple[pd.Series | None, pd.DataFrame | None]:
    """
    Loads data from a single batch directory needed for a specific aggregation method ('by').

    This function loads 'scalars.feather' and, if required by the 'by' method,
    also loads 'cluster_sizes.npy', 'cluster_llrs.npy', and 'offsets.npy'.
    It then calculates or extracts the primary series to be binned.

    Parameters
    ----------
    batch_dir_path : str
        Path to the batch directory.
    by : str
        The aggregation method.
    norm_order : float, optional
        Order for L_p norm calculation, required for norm-based 'by' methods.
    verbose : bool, optional
        If True, prints detailed loading information.

    Returns
    -------
    series_to_bin : pd.Series or None
        The Series containing values to be binned. None if essential files are missing
        or data cannot be computed.
    df_scalars : pd.DataFrame or None
        The DataFrame loaded from 'scalars.feather'. None if 'scalars.feather' is missing.

    Raises
    ------
    FileNotFoundError
        If 'scalars.feather' is not found, or if a .npy file ('cluster_sizes.npy',
        'cluster_llrs.npy', 'offsets.npy') required by the chosen 'by' method
        is not found in the batch directory.
    ValueError
        If `norm_order` is not provided for a norm-based `by` method.
    """
    # Import here to avoid circular imports
    from ..data_aggregation import _calculate_cluster_norms_from_flat_data_numba

    scalars_path = os.path.join(batch_dir_path, "scalars.feather")
    cluster_sizes_path = os.path.join(batch_dir_path, "cluster_sizes.npy")
    cluster_llrs_path = os.path.join(batch_dir_path, "cluster_llrs.npy")
    offsets_path = os.path.join(batch_dir_path, "offsets.npy")

    # Check for scalars.feather first, as it's always needed.
    if not os.path.isfile(scalars_path):
        if verbose:
            print(
                f"  Error: scalars.feather not found in {batch_dir_path}. Cannot process this batch."
            )
        # Raise FileNotFoundError directly as per new requirement
        raise FileNotFoundError(
            f"scalars.feather not found in {batch_dir_path}. Cannot process this batch."
        )

    # Define which 'by' methods require the .npy files
    npy_dependent_methods = [
        "cluster_size_norm",
        "cluster_llr_norm",
        "cluster_size_norm_gap",
        "cluster_llr_norm_gap",
    ]
    files_required_for_by_method = []
    if by in npy_dependent_methods:
        files_required_for_by_method = [
            ("cluster_sizes.npy", cluster_sizes_path),
            ("cluster_llrs.npy", cluster_llrs_path),
            ("offsets.npy", offsets_path),
        ]

    for fname, fpath in files_required_for_by_method:
        if not os.path.isfile(fpath):
            error_msg = (
                f"Error: {fname} is required for 'by={by}' method but not found "
                f"in {batch_dir_path}. Cannot process this batch."
            )
            raise FileNotFoundError(
                error_msg
            )  # Raise error if required .npy is missing

    df_scalars = pd.read_feather(scalars_path)

    if df_scalars.empty:
        if verbose:
            print(f"  Warning: scalars.feather in {batch_dir_path} is empty.")
        # Still return the empty df_scalars as it exists, the caller can decide to skip.
        # The series_to_bin will likely be empty or None.

    series_to_bin: pd.Series | None = None

    if by in npy_dependent_methods:
        if norm_order is None or norm_order <= 0:
            raise ValueError(
                f"'norm_order' must be a positive float when 'by' is '{by}'. Got: {norm_order}"
            )

        cluster_sizes_flat = np.load(cluster_sizes_path, allow_pickle=False)
        cluster_llrs_flat = np.load(cluster_llrs_path, allow_pickle=False)
        offsets = np.load(offsets_path, allow_pickle=False)[:-1]

        num_samples = len(df_scalars)
        inside_cluster_size_norms = np.full(num_samples, np.nan, dtype=float)
        inside_cluster_llr_norms = np.full(num_samples, np.nan, dtype=float)

        if cluster_sizes_flat.size > 0:
            inside_cluster_size_norms, outside_value = (
                _calculate_cluster_norms_from_flat_data_numba(
                    flat_data=cluster_sizes_flat,
                    offsets=offsets,
                    norm_order=norm_order,
                )
            )
        if cluster_llrs_flat.size > 0:
            inside_cluster_llr_norms, outside_value = (
                _calculate_cluster_norms_from_flat_data_numba(
                    flat_data=cluster_llrs_flat,
                    offsets=offsets,
                    norm_order=norm_order,
                )
            )

        if by == "cluster_size_norm":
            series_to_bin = pd.Series(inside_cluster_size_norms, index=df_scalars.index)
        elif by == "cluster_llr_norm":
            series_to_bin = pd.Series(inside_cluster_llr_norms, index=df_scalars.index)
        elif by == "cluster_size_norm_gap":
            series_to_bin = pd.Series(
                outside_value - inside_cluster_size_norms, index=df_scalars.index
            )
        elif by == "cluster_llr_norm_gap":
            series_to_bin = pd.Series(
                outside_value - inside_cluster_llr_norms, index=df_scalars.index
            )
    else:  # 'by' is not an npy_dependent_method, try to get column directly
        if by in df_scalars.columns:
            series_to_bin = df_scalars[by].copy()
        else:
            raise ValueError(
                f"  Error: Column '{by}' not found in scalars.feather for {batch_dir_path}."
            )

    # Clean up to free memory, especially if large arrays were loaded
    if "cluster_sizes_flat" in locals():
        del cluster_sizes_flat
    if "cluster_llrs_flat" in locals():
        del cluster_llrs_flat
    if "offsets" in locals():
        del offsets
    # No need to call gc.collect() aggressively here; Python's GC will handle it.

    return series_to_bin, df_scalars
