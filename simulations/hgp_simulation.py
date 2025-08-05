import os
import warnings
from datetime import datetime
from typing import Any, Dict

import numpy as np
import stim
from scipy import sparse

from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import (
    _convert_df_dtypes_for_feather,
    get_existing_shots,
    bplsd_simulation_task_parallel,
)


def simulate(
    shots: int,
    dv: int,
    dc: int,
    n: int,
    k: int,
    d: int,
    T: int,
    p: float,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
    compute_logical_gap_proxy: bool = False,
    include_cluster_stats: bool = True,
) -> None:
    """
    Run the simulation for a given HGP code configuration, saving results in batches.
    Results include a Feather file for scalar data and NumPy files for ragged arrays.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate for this configuration.
    dv : int
        Variable node degree.
    dc : int
        Check node degree.
    n : int
        Number of qubits.
    k : int
        Number of logical qubits.
    d : int
        Code distance.
    T : int
        Number of rounds.
    p : float
        Physical error probability.
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
    sub_dirname = f"dv{dv}_dc{dc}_n{n}_k{k}_d{d}_T{T}_p{p}"
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping HGP code dv={dv}, dc={dc}, n={n}, k={k}, d={d}, T={T}, p={p} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for HGP code dv={dv}, dc={dc}, n={n}, k={k}, d={d}, T={T}, p={p} into {sub_dirname}"
    )

    # Load the prebuilt HGP circuit for this configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_filename = f"dv{dv}_dc{dc}_n{n}_k{k}_d{d}_T{T}_p{p}.stim"
    circuit_path = os.path.join(
        current_dir, "data/hgp_prebuilt/circuits", circuit_filename
    )

    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"No prebuilt circuit file found: {circuit_path}")

    print(f"Loading HGP circuit from: {circuit_path}")
    circuit = stim.Circuit.from_file(circuit_path)

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
            f"\n[{t0_batch}] Simulating {to_run} shots for HGP code dv={dv}, dc={dc}, n={n}, k={k}, d={d}, T={T}, p={p}. Output to: {batch_output_dir}"
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

    # HGP code parameters - define based on available prebuilt circuits
    # Example: dv3_dc4_n225_k9_d6_T6_p0.001.stim
    hgp_configs = [
        {"dv": 3, "dc": 4, "n": 225, "k": 9, "d": 6, "T": 6},
    ]

    plist = [1e-3]

    shots_per_batch = round(1e6)
    total_shots = round(1e6)
    n_jobs = 18
    repeat = 10
    dir_name = "hgp_minsum_iter30_lsd0_raw"

    # Estimated time (19 cores):
    # Performance estimates will need to be updated based on HGP code characteristics

    decoder_prms = {
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "lsd_method": "LSD_0",
        "lsd_order": 0,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, f"data/{dir_name}")
    os.makedirs(data_dir, exist_ok=True)

    print("hgp_configs =", hgp_configs)
    print("plist =", plist)
    print("decoder_prms =", decoder_prms)

    print(f"\n==== Starting HGP simulations up to {total_shots} shots ====")
    for config in hgp_configs:
        dv, dc, n, k, d, T = (
            config["dv"],
            config["dc"],
            config["n"],
            config["k"],
            config["d"],
            config["T"],
        )
        for i_p, p in enumerate(plist):
            simulate(
                shots=total_shots,
                dv=dv,
                dc=dc,
                n=n,
                k=k,
                d=d,
                T=T,
                p=p,
                data_dir=data_dir,
                n_jobs=n_jobs,
                repeat=repeat,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
            )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Simulations completed ({t0}) ====")
