import os
import re
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import stim
from joblib import Parallel, delayed
from scipy import sparse

from src.ldpc_post_selection.decoder import (
    SoftOutputsBpLsdDecoder,
)
from simulations.utils.simulation_utils import (
    _calculate_chunk_sizes,
    _handle_empty_shot_chunks,
    _convert_df_dtypes_for_feather,
)


def bplsd_simulation_task_single_legacy(
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


def bplsd_simulation_task_parallel_legacy(
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
        delayed(bplsd_simulation_task_single_legacy)(
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

