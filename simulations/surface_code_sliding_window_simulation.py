import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict

import numpy as np
from scipy import sparse

from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import (
    get_existing_shots,
    bplsd_sliding_window_simulation_task_parallel,
)
from simulations.utils.build_circuit import build_surface_code_circuit


def simulate(
    shots: int,
    p: float,
    d: int,
    T: int,
    window_size: int,
    commit_size: int,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
    noise_model: str = "circuit-level",
) -> None:
    """
    Run sliding window simulation for a given (p, d, T) configuration, saving results in batches.
    Results include numpy array for fails and CSR sparse matrices for cluster information.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate for this configuration.
    p : float
        Physical error probability.
    d : int
        Code distance.
    T : int
        Number of rounds.
    window_size : int
        Number of rounds in each window.
    commit_size : int
        Number of rounds for each commitment.
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
    noise_model : str, default="circuit-level"
        The noise model type: ["circuit-level", "code-capacity", "phenom"].

    Returns
    -------
    None
        This function writes results to files and prints status messages.
    """
    # Create subdirectory path based on parameters
    sub_dirname = f"d{d}_T{T}_p{p}_W{window_size}_F{commit_size}"
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, d={d}, T={T}, W={window_size}, F={commit_size} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for p={p}, d={d}, T={T}, W={window_size}, F={commit_size} into {sub_dirname}"
    )

    # Create the circuit once for this (p, d, T) configuration
    circuit = build_surface_code_circuit(
        p=p, d=d, T=T, noise=noise_model, only_z_detectors=True
    )

    # Save H & prior probabilities if not exists
    prior_path = os.path.join(sub_data_dir, "priors.npy")
    H_path = os.path.join(sub_data_dir, "H.npz")
    if not os.path.exists(prior_path) or not os.path.exists(H_path):
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit)
        np.save(prior_path, decoder.priors)
        sparse.save_npz(H_path, decoder.H)

    # Save committed_faults if not exists (deterministic for this configuration)
    committed_faults_path = os.path.join(sub_data_dir, "committed_faults.npz")
    committed_faults_saved = os.path.exists(committed_faults_path)

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
        batch_output_dir = os.path.join(sub_data_dir, f"batch_{next_idx}_{to_run}")

        t0_batch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n[{t0_batch}] Simulating {to_run} shots for p={p}, d={d}, T={T}, W={window_size}, F={commit_size}. Output to: {batch_output_dir}"
        )

        # Run sliding window simulation
        (
            fails,
            all_clusters_csr,
            committed_clusters_csr,
            committed_faults,
        ) = bplsd_sliding_window_simulation_task_parallel(
            shots=to_run,
            circuit=circuit,
            window_size=window_size,
            commit_size=commit_size,
            n_jobs=n_jobs,
            repeat=repeat,
            decoder_prms=decoder_prms,
        )

        # Save committed_faults if not saved yet and available from this batch
        if not committed_faults_saved and committed_faults is not None:
            np.savez_compressed(committed_faults_path, *committed_faults)
            committed_faults_saved = True

        # Prepare filenames for this batch
        fp_fails = os.path.join(batch_output_dir, "fails.npy")
        fp_all_clusters = os.path.join(batch_output_dir, "all_clusters.npz")
        fp_committed_clusters = os.path.join(batch_output_dir, "committed_clusters.npz")

        # Save results
        os.makedirs(batch_output_dir, exist_ok=True)

        # Save fails as numpy array
        np.save(fp_fails, fails)

        # Save cluster CSR arrays as compressed NPZ files
        sparse.save_npz(fp_all_clusters, all_clusters_csr)
        sparse.save_npz(fp_committed_clusters, committed_clusters_csr)

        current_simulated_for_config += to_run
        remaining -= to_run
        total_processed_for_config = current_simulated_for_config

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

    plist = [5e-3]
    dlist = [13]  # [7, 9, 11, 13]

    # Sliding window parameters
    window_size = 3
    commit_size = 1

    noise_model = "circuit-level"

    shots_per_batch = round(1e6)
    total_shots = round(1e7)
    n_jobs = 18
    repeat = 10
    dir_name = "surface_sliding_window_minsum_iter30_lsd0_raw"

    decoder_prms = {
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "lsd_method": "LSD_0",
        "lsd_order": 0,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, f"data/{dir_name}")
    os.makedirs(data_dir, exist_ok=True)

    print("dlist =", dlist)
    print("plist =", plist)
    print("window_size =", window_size)
    print("commit_size =", commit_size)
    print("decoder_prms =", decoder_prms)

    print(f"\n==== Starting sliding window simulations up to {total_shots} shots ====")
    for d in dlist:
        T = d

        for i_p, p in enumerate(plist):
            simulate(
                shots=total_shots,
                p=p,
                d=d,
                T=T,
                window_size=window_size,
                commit_size=commit_size,
                data_dir=data_dir,
                n_jobs=n_jobs,
                repeat=repeat,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
                noise_model=noise_model,
            )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Sliding window simulations completed ({t0}) ====")
