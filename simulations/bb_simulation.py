import os
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import stim
from joblib import Parallel, delayed

from src.ldpc_post_selection.build_circuit import build_BB_circuit, get_BB_distance
from src.ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder


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
    circuit: stim.Circuit,  # Changed: circuit is passed as an argument
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
        cluster_sizes_list.append(soft_info.get("cluster_sizes", np.array([])))
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
    circuit: stim.Circuit,  # Changed: circuit is passed
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
    if shots == 0:  # Handle case with 0 shots
        empty_df = pd.DataFrame(
            columns=["fail", "fail_bp", "converge", "pred_llr", "detector_density"]
        )
        empty_df = _convert_df_dtypes_for_feather(
            empty_df.copy()
        )  # Ensure correct dtypes
        return (
            empty_df,
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([0], dtype=int),
        )

    # Divide shots among jobs
    base = shots // (n_jobs * repeat)
    remainder = shots % (n_jobs * repeat)
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_jobs * repeat)]
    # Filter out zero-sized chunks if any to prevent issues with task
    chunk_sizes = [cs for cs in chunk_sizes if cs > 0]
    if not chunk_sizes and shots > 0:  # Should not happen if shots > 0
        warnings.warn("No chunks to run, though shots > 0. This is unexpected.")
        empty_df = pd.DataFrame(
            columns=["fail", "fail_bp", "converge", "pred_llr", "detector_density"]
        )
        empty_df = _convert_df_dtypes_for_feather(empty_df.copy())
        return (
            empty_df,
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
        flat_cluster_sizes = np.concatenate(cluster_sizes_list_flat)
    else:
        flat_cluster_sizes = np.array([], dtype=int)

    if cluster_llrs_list_flat:
        flat_cluster_llrs = np.concatenate(cluster_llrs_list_flat)
    else:
        flat_cluster_llrs = np.array([], dtype=float)

    lengths = [len(cs_arr) for cs_arr in cluster_sizes_list_flat]
    offsets = np.cumsum([0] + lengths).astype(
        np.int64
    )  # Use int64 for cumsum, then cast later

    return df, flat_cluster_sizes, flat_cluster_llrs, offsets


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


def simulate(
    shots: int,
    p: float,
    n: int,
    T: int,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
) -> None:
    """
    Run the simulation for a given (p, n, T) configuration, saving results in batches.
    Results include a Feather file for scalar data and NumPy files for ragged arrays.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate for this configuration.
    p : float
        Physical error probability.
    n : int
        Number of qubits.
    T : int
        Number of rounds.
    data_dir : str
        Base directory to store output subdirectories.
    n_jobs : int
        Number of parallel jobs for `task_parallel`.
    repeat : int
        Number of repeats for `task_parallel`.
    shots_per_batch : int
        Number of shots to simulate and save per batch file.
    decoder_prms : Dict[str, Any], optional
        Parameters for the SoftOutputsBpLsdDecoder.

    Returns
    -------
    None
        This function writes results to files and prints status messages.
    """
    # Create subdirectory path based on parameters
    sub_dirname = f"n{n}_T{T}_p{p}"
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, n={n}, T={T} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for p={p}, n={n}, T={T} into {sub_dirname}"
    )

    # Create the circuit once for this (p, n, T) configuration
    circuit = build_BB_circuit(p=p, n=n, T=T)
    dem = circuit.detector_error_model()

    # Determine dtypes for NumPy arrays using the helper function
    cluster_size_dtype = _get_optimal_uint_dtype(dem.num_errors)
    offset_dtype = _get_optimal_uint_dtype(dem.num_errors * shots_per_batch)

    # Determine the next file index
    next_idx = (
        max([info[0] for info in existing_files_info], default=0) + 1
        if existing_files_info
        else 1
    )

    # Simulate and save in batches
    current_simulated_for_config = 0
    while remaining > 0:
        to_run = min(shots_per_batch, remaining)
        if to_run == 0:
            break  # Should not happen if remaining > 0

        # Define the specific directory for this batch's output
        batch_output_dir = os.path.join(
            sub_data_dir, f"batch_{next_idx}_{shots_per_batch}"
        )

        t0_batch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n[{t0_batch}] Simulating {to_run} shots for p={p}, n={n}, T={T}. Output to: {batch_output_dir}"
        )

        df_new, flat_cluster_sizes, flat_cluster_llrs, offsets = task_parallel(
            shots=to_run,
            circuit=circuit,  # Pass the pre-built circuit
            n_jobs=n_jobs,
            repeat=repeat,
            decoder_prms=decoder_prms,
        )

        # Prepare filenames for this batch (now with fixed names within batch_output_dir)
        fp_feather = os.path.join(batch_output_dir, "scalars.feather")
        fp_cs = os.path.join(batch_output_dir, "cluster_sizes.npy")
        fp_cl = os.path.join(batch_output_dir, "cluster_llrs.npy")
        fp_offsets = os.path.join(batch_output_dir, "offsets.npy")

        # Convert dtypes and save
        os.makedirs(batch_output_dir, exist_ok=True)
        df_new = _convert_df_dtypes_for_feather(
            df_new.copy()
        )  # Use .copy() to avoid SettingWithCopyWarning
        df_new.to_feather(fp_feather)

        np.save(fp_cs, flat_cluster_sizes.astype(cluster_size_dtype))
        np.save(fp_cl, flat_cluster_llrs.astype(np.float32))
        np.save(fp_offsets, offsets.astype(offset_dtype))

        current_simulated_for_config += to_run
        remaining -= to_run
        total_processed_for_config = current_simulated_for_config  # Corrected logic

        print(
            f"   Created files in {batch_output_dir} with {to_run} shots. "
            f"{total_processed_for_config} / {shots - total_existing} new shots processed for this config. "
            f"{remaining} shots still remaining for this config."
        )
        next_idx += 1


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore", message="A worker stopped while some jobs were given to the executor."
    )

    plist = [1e-3]
    n = 144  # [72, 108, 144, 288]

    # Changed from a list to a single value for shots_per_batch configuration
    shots_per_batch = [round(5e6)]
    total_shots = round(1e9)

    # Estimated time (20 cores):
    # p=1e-3, n=144: 100,000 shots/min
    # p=3e-3, n=144: 50,000 shots/min
    # p=5e-3, n=144: 12,500 shots/min

    decoder_prms = {
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "lsd_method": "LSD_0",
        "lsd_order": 0,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/bb_minsum_iter30_lsd0")
    os.makedirs(data_dir, exist_ok=True)

    print("n =", n)
    print("plist =", plist)
    print("decoder_prms =", decoder_prms)

    print(f"\n==== Starting simulations up to {total_shots} shots ====")
    for i_p, p in enumerate(plist):
        shots_per_batch_now = shots_per_batch[i_p]
        T = get_BB_distance(n)
        simulate(
            shots=total_shots,
            p=p,
            n=n,
            T=T,
            data_dir=data_dir,
            n_jobs=19,
            repeat=10,
            shots_per_batch=shots_per_batch_now,
            decoder_prms=decoder_prms,
        )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Simulations completed ({t0}) ====")
