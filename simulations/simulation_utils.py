import os
import re
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import stim
from joblib import Parallel, delayed

from src.ldpc_post_selection.decoder import (
    SoftOutputsBpLsdDecoder,
    SoftOutputsMatchingDecoder,
)


def get_existing_shots(data_dir: str) -> Tuple[int, List[Tuple[int, str, int]]]:
    """
    Calculate the total number of shots already processed by summing shots from directory names.

    It looks for subdirectories named 'batch_{idx}_{shots_in_batch_name}' within data_dir.
    The 'shots_in_batch_name' part of the directory name is parsed to determine
    the number of shots processed in that batch.

    Parameters
    ----------
    data_dir : str
        Directory for a specific configuration (e.g., "data/base_dir/n72_T6_p0.002").

    Returns
    -------
    total_existing : int
        The total number of shots found by summing the 'shots_in_batch_name'
        from each valid batch directory name.
    existing_files_info : list of tuple
        A list containing tuples of (batch_index, batch_directory_path, shots_from_dirname)
        for each correctly named batch directory, sorted by batch_index.
        'shots_from_dirname' is the number of shots parsed from the directory name.
    """
    total_existing = 0
    existing_files_info = []

    # Regex to match "batch_{idx}_{shots_per_batch_in_name}"
    pattern = re.compile(r"^batch_(\d+)_(\d+)$")

    if not os.path.isdir(data_dir):  # Check if the base data_dir exists
        return 0, []

    for dirname in os.listdir(data_dir):
        match = pattern.match(dirname)
        if match:
            try:
                batch_idx = int(match.group(1))
                shots_in_name = int(match.group(2))  # Parsed from dirname
            except ValueError:
                warnings.warn(
                    f"Could not parse batch index or shots from directory name {dirname}. Skipping."
                )
                continue

            batch_subdir_path = os.path.join(data_dir, dirname)
            if os.path.isdir(batch_subdir_path):  # Ensure it's a directory
                # Instead of reading feather, we use shots_in_name
                total_existing += shots_in_name
                existing_files_info.append(
                    (batch_idx, batch_subdir_path, shots_in_name)
                )

    existing_files_info.sort(key=lambda x: x[0])  # Sort by batch_index

    return total_existing, existing_files_info


def _convert_df_dtypes_for_feather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts DataFrame column dtypes for optimized Feather storage.

    This function iterates through each column of the input DataFrame.
    If a column's data type is float, it's converted to `float32`.
    If a column's data type is integer, it's converted to `int32`.
    Boolean columns are left as is.
    The modifications are performed in-place on the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose columns are to be type-converted.

    Returns
    -------
    df : pandas.DataFrame
        The same input DataFrame with dtypes of its columns modified.
    """
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        if pd.api.types.is_float_dtype(col_dtype):
            df[col_name] = df[col_name].astype(np.float32)
        elif pd.api.types.is_integer_dtype(col_dtype):  # Only handle integers here
            df[col_name] = df[col_name].astype(np.int32)
        # Boolean columns are intentionally not handled here to keep their 'bool' dtype
    return df


def _get_optimal_uint_dtype(max_val: int) -> np.dtype:
    """
    Determines the smallest NumPy unsigned integer dtype that can hold max_val.

    Parameters
    ----------
    max_val : int
        The maximum possible value that the dtype needs to represent.

    Returns
    -------
    dtype : numpy.dtype
        The optimal NumPy unsigned integer dtype (np.uint16, np.uint32, or np.uint64).
    """
    if max_val < 2**16:
        return np.uint16
    elif max_val < 2**32:
        return np.uint32
    else:
        return np.uint64


def task(
    shots: int,
    circuit: stim.Circuit,
    decoder_prms: Dict[str, Any] | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, float]],
    List[np.ndarray],
    List[np.ndarray],
]:
    """
    Run a single simulation task for a given circuit and decoder parameters.

    Parameters
    ----------
    shots : int
        Number of shots to simulate.
    circuit : stim.Circuit
        The pre-built quantum error correction circuit.
    decoder_prms : Dict[str, Any], optional
        Parameters for the SoftOutputsBpLsdDecoder.

    Returns
    -------
    fails : np.ndarray
        Boolean array indicating if the LSD decoding failed for each shot.
    fails_bp : np.ndarray
        Boolean array indicating if the BP decoding failed for each shot.
    converges : np.ndarray
        Boolean array indicating if the BP algorithm converged for each shot.
    scalar_soft_infos : list of dict
        List of dictionaries, each containing scalar soft information like
        'pred_llr' and 'detector_density' for each shot.
    cluster_sizes_list : list of np.ndarray
        List of 1D NumPy arrays, where each array contains the cluster sizes
        for a single shot.
    cluster_llrs_list : list of np.ndarray
        List of 1D NumPy arrays, where each array contains the cluster LLRs
        for a single shot.
    """
    # circuit = build_BB_circuit(p=p, n=n, T=T) # Removed: circuit is now an argument
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots, separate_observables=True)

    if decoder_prms is None:
        decoder_prms = {}

    decoder = SoftOutputsBpLsdDecoder(
        circuit=circuit,
        **decoder_prms,
    )
    preds_list = []
    preds_bp_list = []
    converges_list = []
    scalar_soft_infos_list = []  # For pred_llr, detector_density
    cluster_sizes_list = []  # For cluster_sizes arrays
    cluster_llrs_list = []  # For cluster_llrs arrays

    for det_sng in det:
        pred, pred_bp, converge, soft_info = decoder.decode(
            det_sng,
        )

        preds_list.append(pred)
        preds_bp_list.append(pred_bp)
        converges_list.append(converge)

        # Extract new soft outputs
        scalar_soft_infos_list.append(
            {
                "pred_llr": soft_info.get("pred_llr"),
                "detector_density": soft_info.get("detector_density"),
            }
        )
        cluster_sizes_list.append(
            soft_info.get("cluster_sizes", np.array([], dtype=int))
        )
        cluster_llrs_list.append(soft_info.get("cluster_llrs", np.array([])))

    converges_arr = np.array(converges_list)

    preds_arr = (
        np.array(preds_list)
        if preds_list
        else np.empty((0, circuit.num_detectors), dtype=bool)
    )
    preds_bp_arr = (
        np.array(preds_bp_list)
        if preds_bp_list
        else np.empty((0, circuit.num_detectors), dtype=bool)
    )

    obs_matrix_T = decoder.obs_matrix.T
    if preds_arr.shape[0] > 0:
        obs_preds_arr = ((preds_arr.astype(np.uint8) @ obs_matrix_T) % 2).astype(bool)
        obs_preds_bp_arr = ((preds_bp_arr.astype(np.uint8) @ obs_matrix_T) % 2).astype(
            bool
        )
        # Compare with the true logical observables 'obs'
        fails_arr = np.any(obs ^ obs_preds_arr, axis=1)
        fails_bp_arr = np.any(obs ^ obs_preds_bp_arr, axis=1)
    else:  # Handle case with 0 shots
        obs_shape = obs.shape[1] if obs.ndim > 1 else 0
        fails_arr = np.empty(0, dtype=bool)
        fails_bp_arr = np.empty(0, dtype=bool)

    return (
        fails_arr,
        fails_bp_arr,
        converges_arr,
        scalar_soft_infos_list,
        cluster_sizes_list,
        cluster_llrs_list,
    )


def task_parallel(
    shots: int,
    circuit: stim.Circuit,
    n_jobs: int,
    repeat: int = 10,
    decoder_prms: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the BB `task` function in parallel and return results including flattened arrays.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate.
    circuit : stim.Circuit
        The pre-built quantum error correction circuit.
    n_jobs : int
        Number of parallel jobs.
    repeat : int
        Number of repeats for parallel execution.
    decoder_prms : Dict[str, Any], optional
        Parameters for the decoder.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing boolean flags (fail, fail_bp, converge) and scalar
        float soft outputs (pred_llr, detector_density) for each sample.
    flat_cluster_sizes : np.ndarray
        1D NumPy array of all cluster sizes, flattened across all samples.
    flat_cluster_llrs : np.ndarray
        1D NumPy array of all cluster LLRs, flattened across all samples.
    offsets : np.ndarray
        1D NumPy array storing the starting indices for each sample's data in
        the flattened `flat_cluster_sizes` and `flat_cluster_llrs` arrays.
        The length of this array is (number of samples + 1).
    """
    # Divide shots among jobs
    chunk_sizes = _calculate_chunk_sizes(shots, n_jobs, repeat)

    # Handle empty shots or chunks
    empty_df_check = _handle_empty_shot_chunks(
        shots,
        chunk_sizes,
        ["fail", "fail_bp", "converge", "pred_llr", "detector_density"],
    )
    if empty_df_check is not None:
        return (
            empty_df_check,
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([0], dtype=int),
        )

    # Execute tasks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(task)(
            shots=chunk,
            circuit=circuit,  # Pass circuit
            decoder_prms=decoder_prms,
        )
        for chunk in chunk_sizes
    )

    # Unpack and combine results
    (
        fails_l,
        fails_bp_l,
        converges_l,
        scalar_soft_infos_nested_l,
        cs_nested_l,
        cl_nested_l,
    ) = zip(*results)

    fails_s = pd.Series(np.concatenate(fails_l), name="fail", dtype=bool)
    fails_bp_s = pd.Series(np.concatenate(fails_bp_l), name="fail_bp", dtype=bool)
    converges_s = pd.Series(np.concatenate(converges_l), name="converge", dtype=bool)

    scalar_soft_infos_flat_list = [
        item for sublist in scalar_soft_infos_nested_l for item in sublist
    ]
    df_soft = pd.DataFrame(
        scalar_soft_infos_flat_list
    )  # Contains pred_llr, detector_density

    # Combine fail flag with soft info
    df = pd.concat([fails_s, fails_bp_s, converges_s, df_soft], axis=1)

    # Process cluster_sizes and cluster_llrs
    cluster_sizes_list_flat = [item for sublist in cs_nested_l for item in sublist]
    cluster_llrs_list_flat = [item for sublist in cl_nested_l for item in sublist]

    if cluster_sizes_list_flat:
        flat_cluster_sizes = np.concatenate(cluster_sizes_list_flat).astype(np.int_)
    else:
        flat_cluster_sizes = np.array([], dtype=np.int_)

    if cluster_llrs_list_flat:
        flat_cluster_llrs = np.concatenate(cluster_llrs_list_flat)
    else:
        flat_cluster_llrs = np.array([], dtype=float)

    lengths = [len(cs_arr) for cs_arr in cluster_sizes_list_flat]
    offsets = np.cumsum([0] + lengths).astype(
        np.int64
    )  # Use int64 for cumsum, then cast later

    df = _convert_df_dtypes_for_feather(df.copy())  # Ensure correct dtypes for output
    return df, flat_cluster_sizes, flat_cluster_llrs, offsets


def task_matching(
    shots: int,
    circuit: stim.Circuit,
    decoder_prms: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Run a single simulation task using the Matching decoder.

    Parameters
    ----------
    shots : int
        Number of shots to simulate.
    circuit : stim.Circuit
        The pre-built quantum error correction circuit.
    decoder_prms : Dict[str, Any], optional
        Parameters for the SoftOutputsMatchingDecoder.

    Returns
    -------
    fails : np.ndarray
        Boolean array indicating if the Matching decoding failed for each shot.
    scalar_soft_infos : list of dict
        List of dictionaries, each containing scalar soft information like
        'pred_llr', 'detector_density', and 'gap' for each shot.
    """
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots, separate_observables=True)

    if decoder_prms is None:
        decoder_prms = {}

    decoder = SoftOutputsMatchingDecoder(
        circuit=circuit,
        **decoder_prms,
    )
    preds_list = []
    scalar_soft_infos_list = []

    for det_sng in det:
        pred, soft_info = decoder.decode(
            det_sng,
        )

        preds_list.append(pred)
        scalar_soft_infos_list.append(soft_info)

    preds_arr = (
        np.array(preds_list)
        if preds_list
        else np.empty((0, circuit.num_detectors), dtype=bool)
    )

    obs_matrix_T = decoder.obs_matrix.T
    if preds_arr.shape[0] > 0:
        obs_preds_arr = ((preds_arr.astype(np.uint8) @ obs_matrix_T) % 2).astype(bool)
        fails_arr = np.any(obs ^ obs_preds_arr, axis=1)
    else:  # Handle case with 0 shots
        fails_arr = np.empty(0, dtype=bool)

    return fails_arr, scalar_soft_infos_list


def task_matching_parallel(
    shots: int,
    circuit: stim.Circuit,
    n_jobs: int,
    repeat: int = 10,
    decoder_prms: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run the `task_matching` function in parallel and return results.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate.
    circuit : stim.Circuit
        The pre-built quantum error correction circuit.
    n_jobs : int
        Number of parallel jobs.
    repeat : int
        Number of repeats for parallel execution.
    decoder_prms : Dict[str, Any], optional
        Parameters for the decoder.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing boolean flags (fail) and scalar
        float soft outputs (pred_llr, detector_density, gap) for each sample.
    """
    # Divide shots among jobs
    chunk_sizes = _calculate_chunk_sizes(shots, n_jobs, repeat)

    # Handle empty shots or chunks
    empty_df_check = _handle_empty_shot_chunks(
        shots, chunk_sizes, ["fail", "pred_llr", "detector_density", "gap"]
    )
    if empty_df_check is not None:
        return empty_df_check

    # Execute tasks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(task_matching)(
            shots=chunk,
            circuit=circuit,  # Pass circuit
            decoder_prms=decoder_prms,
        )
        for chunk in chunk_sizes
    )

    # Unpack and combine results
    fails_l, scalar_soft_infos_nested_l = zip(*results)

    fails_s = pd.Series(np.concatenate(fails_l), name="fail", dtype=bool)

    scalar_soft_infos_flat_list = [
        item for sublist in scalar_soft_infos_nested_l for item in sublist
    ]
    df_soft = pd.DataFrame(
        scalar_soft_infos_flat_list
    )  # Contains pred_llr, detector_density, gap

    # Combine fail flag with soft info
    df = pd.concat([fails_s, df_soft], axis=1)

    df = _convert_df_dtypes_for_feather(df.copy())  # Ensure correct dtypes for output
    return df


def _calculate_chunk_sizes(shots: int, n_jobs: int, repeat: int) -> List[int]:
    """
    Calculates the distribution of shots into chunks for parallel processing.

    Parameters
    ----------
    shots : int
        Total number of shots.
    n_jobs : int
        Number of parallel jobs.
    repeat : int
        Number of repeats for parallel execution.

    Returns
    -------
    chunk_sizes : list of int
        A list where each element is the number of shots for a chunk.
        Zero-sized chunks are filtered out.
    """
    if shots == 0:
        return []
    base = shots // (n_jobs * repeat)
    remainder = shots % (n_jobs * repeat)
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_jobs * repeat)]
    # Filter out zero-sized chunks if any to prevent issues with task
    chunk_sizes = [cs for cs in chunk_sizes if cs > 0]
    return chunk_sizes


def _handle_empty_shot_chunks(
    shots: int, chunk_sizes: List[int], columns: List[str]
) -> pd.DataFrame | None:
    """
    Handles cases where there are no shots or no chunks to process.

    Parameters
    ----------
    shots : int
        Total number of shots.
    chunk_sizes : list of int
        List of chunk sizes.
    columns : list of str
        Column names for the empty DataFrame if returned.

    Returns
    -------
    df : pandas.DataFrame or None
        Returns an empty DataFrame with specified columns if shots is 0
        or if there are no chunks despite shots > 0. Otherwise, returns None.
    """
    if shots == 0:
        empty_df = pd.DataFrame(columns=columns)
        return _convert_df_dtypes_for_feather(empty_df.copy())

    if not chunk_sizes and shots > 0:
        warnings.warn("No chunks to run, though shots > 0. This is unexpected.")
        empty_df = pd.DataFrame(columns=columns)
        return _convert_df_dtypes_for_feather(empty_df.copy())
    return None
