import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict

import numpy as np

from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import (
    get_existing_shots,
    bplsd_sliding_window_simulation_task_parallel,
)
from simulations.utils.build_circuit import build_BB_circuit, get_BB_distance


def simulate(
    shots: int,
    p: float,
    n: int,
    T: int,
    window_size: int,
    commit_size: int,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
) -> None:
    """
    Run sliding window simulation for a given (p, n, T) configuration, saving results in batches.
    Results include numpy array for fails and pickled files for cluster statistics.

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

    Returns
    -------
    None
        This function writes results to files and prints status messages.
    """
    # Create subdirectory path based on parameters
    sub_dirname = f"n{n}_T{T}_p{p}_W{window_size}_F{commit_size}"
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, n={n}, T={T}, W={window_size}, F={commit_size} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for p={p}, n={n}, T={T}, W={window_size}, F={commit_size} into {sub_dirname}"
    )

    # Create the circuit once for this (p, n, T) configuration
    circuit = build_BB_circuit(p=p, n=n, T=T)

    # Save prior probabilities if not exists
    prior_path = os.path.join(sub_data_dir, "priors.npy")
    if not os.path.exists(prior_path):
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit)
        np.save(prior_path, decoder.priors)

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
            f"\n[{t0_batch}] Simulating {to_run} shots for p={p}, n={n}, T={T}, W={window_size}, F={commit_size}. Output to: {batch_output_dir}"
        )

        # Run sliding window simulation
        (
            fails,
            cluster_sizes,
            cluster_llrs,
            committed_cluster_sizes,
            committed_cluster_llrs,
        ) = bplsd_sliding_window_simulation_task_parallel(
            shots=to_run,
            circuit=circuit,
            window_size=window_size,
            commit_size=commit_size,
            n_jobs=n_jobs,
            repeat=repeat,
            decoder_prms=decoder_prms,
        )

        # Prepare filenames for this batch
        fp_fails = os.path.join(batch_output_dir, "fails.npy")
        fp_cluster_sizes = os.path.join(batch_output_dir, "cluster_sizes.pkl")
        fp_cluster_llrs = os.path.join(batch_output_dir, "cluster_llrs.pkl")
        fp_committed_cluster_sizes = os.path.join(
            batch_output_dir, "committed_cluster_sizes.pkl"
        )
        fp_committed_cluster_llrs = os.path.join(
            batch_output_dir, "committed_cluster_llrs.pkl"
        )

        # Save results
        os.makedirs(batch_output_dir, exist_ok=True)

        # Save fails as numpy array
        np.save(fp_fails, fails)

        # Save cluster statistics as pickled files
        with open(fp_cluster_sizes, "wb") as f:
            pickle.dump(cluster_sizes, f)
        with open(fp_cluster_llrs, "wb") as f:
            pickle.dump(cluster_llrs, f)
        with open(fp_committed_cluster_sizes, "wb") as f:
            pickle.dump(committed_cluster_sizes, f)
        with open(fp_committed_cluster_llrs, "wb") as f:
            pickle.dump(committed_cluster_llrs, f)

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
    nlist = [144]  # [72, 108, 144, 288]

    # Sliding window parameters
    window_size = 3
    commit_size = 1

    shots_per_batch = round(1e6)
    total_shots = round(1e7)
    n_jobs = 18
    repeat = 10
    dir_name = "bb_sliding_window_minsum_iter30_lsd0"

    decoder_prms = {
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "lsd_method": "LSD_0",
        "lsd_order": 0,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, f"data/{dir_name}")
    os.makedirs(data_dir, exist_ok=True)

    print("nlist =", nlist)
    print("plist =", plist)
    print("window_size =", window_size)
    print("commit_size =", commit_size)
    print("decoder_prms =", decoder_prms)

    print(f"\n==== Starting sliding window simulations up to {total_shots} shots ====")
    for n in nlist:
        T = get_BB_distance(n)

        for i_p, p in enumerate(plist):
            simulate(
                shots=total_shots,
                p=p,
                n=n,
                T=T,
                window_size=window_size,
                commit_size=commit_size,
                data_dir=data_dir,
                n_jobs=n_jobs,
                repeat=repeat,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
            )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Sliding window simulations completed ({t0}) ====")
