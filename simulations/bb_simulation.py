import os
import warnings
from datetime import datetime
from typing import Any, Dict

import numpy as np
from scipy import sparse

from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import (
    _convert_df_dtypes_for_feather,
    _get_optimal_uint_dtype,
    get_existing_shots,
    bplsd_simulation_task_parallel,
)
from simulations.utils.simulation_utils_legacy import (
    bplsd_simulation_task_parallel_legacy,
)
from simulations.utils.build_circuit import build_BB_circuit, get_BB_distance


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
    compute_logical_gap_proxy: bool = False,
    include_cluster_stats: bool = True,
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
    compute_logical_gap_proxy : bool, optional
        Whether to compute logical gap proxy. Defaults to False.
    include_cluster_stats : bool, optional
        Whether to include cluster statistics. Defaults to True.

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
            f"\n[{t0_batch}] Simulating {to_run} shots for p={p}, n={n}, T={T}. Output to: {batch_output_dir}"
        )

        df_new, clusters_csr, preds_csr, preds_bp_csr = bplsd_simulation_task_parallel(
            shots=to_run,
            circuit=circuit,  # Pass the pre-built circuit
            n_jobs=n_jobs,
            repeat=repeat,
            decoder_prms=decoder_prms,
            compute_logical_gap_proxy=compute_logical_gap_proxy,
            include_cluster_stats=include_cluster_stats,
        )

        # Prepare filenames for this batch (now with fixed names within batch_output_dir)
        fp_feather = os.path.join(batch_output_dir, "scalars.feather")
        fp_clusters = os.path.join(batch_output_dir, "clusters.npz")
        fp_preds = os.path.join(batch_output_dir, "preds.npz")
        fp_preds_bp = os.path.join(batch_output_dir, "preds_bp.npz")

        # Convert dtypes and save
        os.makedirs(batch_output_dir, exist_ok=True)
        df_new = _convert_df_dtypes_for_feather(
            df_new.copy()
        )  # Use .copy() to avoid SettingWithCopyWarning
        df_new.to_feather(fp_feather)

        # Save sparse matrices as compressed NPZ files
        if clusters_csr is not None:
            sparse.save_npz(fp_clusters, clusters_csr)
        sparse.save_npz(fp_preds, preds_csr)
        sparse.save_npz(fp_preds_bp, preds_bp_csr)

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

    plist = np.arange(1e-3, 6.1e-3, 1e-3).round(4)
    nlist = [72, 144]  # [72, 108, 144, 288]

    shots_per_batch = round(1e7)
    total_shots = round(1e7)
    compute_logical_gap_proxy = False
    include_cluster_stats = True
    n_jobs = 18
    repeat = 10
    dir_name = "bb_minsum_iter30_lsd0_raw"

    # Estimated time (19 cores):
    # p=1e-3, n=144: 100,000 shots/min
    # p=3e-3, n=144: 50,000 shots/min
    # p=5e-3, n=144: 12,500 shots/min
    # p=1e-3, n=72: 1,000,000 shots/min
    # p=3e-3, n=72: 500,000 shots/min
    # p=5e-3, n=72: 250,000 shots/min

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
    print("decoder_prms =", decoder_prms)
    print("compute_logical_gap_proxy =", compute_logical_gap_proxy)

    print(f"\n==== Starting simulations up to {total_shots} shots ====")
    for n in nlist:
        T = get_BB_distance(n)
        for i_p, p in enumerate(plist):
            simulate(
                shots=total_shots,
                p=p,
                n=n,
                T=T,
                data_dir=data_dir,
                n_jobs=n_jobs,
                repeat=repeat,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
                compute_logical_gap_proxy=compute_logical_gap_proxy,
                include_cluster_stats=include_cluster_stats,
            )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Simulations completed ({t0}) ====")
