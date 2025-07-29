import os
import warnings
from datetime import datetime
from typing import Any, Dict

import numpy as np
from scipy import sparse

from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import (
    _convert_df_dtypes_for_feather,
    get_existing_shots,
    bplsd_simulation_task_parallel,
)
from simulations.utils.simulation_utils_legacy import (
    bplsd_simulation_task_parallel_legacy,
)
from simulations.utils.build_circuit import build_surface_code_circuit


def simulate(
    shots: int,
    d: int,
    T: int,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    sub_dirname: str,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
    noise_model: str | dict[str, float] = "circuit-level",
) -> None:
    """
    Run the simulation for a given (p, d, T) configuration, saving results in batches.
    Results include a Feather file for scalar data and compressed NumPy files for sparse matrices.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate for this configuration.
    d : int
        Code distance.
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
    noise_model : str | dict[str, float], default="circuit-level"
        The noise model type: ["circuit-level", "code-capacity", "phenom"].
        If a dictionary is provided, the keys should be "clifford", "meas", "reset", and "depol",
        and the values should be the corresponding error rates.

    Returns
    -------
    None
        This function writes results to files and prints status messages.
    """
    # Create subdirectory path based on parameters
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping d={d}, T={T} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for d={d}, T={T} into {sub_dirname}"
    )

    # Create the circuit once for this (p, d, T) configuration
    circuit = build_surface_code_circuit(
        d=d, T=T, noise=noise_model, only_z_detectors=True
    )

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
            f"\n[{t0_batch}] Simulating {to_run} shots for d={d}, T={T}. Output to: {batch_output_dir}"
        )

        df_new, clusters_csr, preds_csr, preds_bp_csr = bplsd_simulation_task_parallel(
            shots=to_run,
            circuit=circuit,  # Pass the pre-built circuit
            n_jobs=n_jobs,
            repeat=repeat,
            decoder_prms=decoder_prms,
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

    # Simulate the surface code with only Z-type detectors

    warnings.filterwarnings(
        "ignore", message="A worker stopped while some jobs were given to the executor."
    )

    # T = d
    # prms = []
    # for d in [9]:
    #     T = d
    #     p_depol = 0.02
    #     for p_meas in np.arange(0, 0.021, 0.005).round(3):
    #         noise_model = {
    #             "meas": p_meas,
    #             "depol": p_depol,
    #         }
    #         prms.append((d, T, noise_model))

    # T = 1 (code capacity)
    # d = 9
    # prms = [(9, 1, {"depol": 0.05, "meas": 0})]

    # p_meas=0, varying T
    prms = []
    for d in [9]:
        for T in [1, 3, 5, 7, 9]:
            p_depol = 0.05
            p_meas = 0
            noise_model = {
                "meas": p_meas,
                "depol": p_depol,
            }
            prms.append((d, T, noise_model))

    shots_per_batch = round(1e7)
    total_shots = round(1e7)
    n_jobs = 18
    repeat = 10

    decoder_prms = {
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "lsd_method": "LSD_0",
        "lsd_order": 0,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_name = "surface_minsum_iter30_lsd0_biased_phenom_raw"

    data_dir = os.path.join(current_dir, "data", data_dir_name)
    os.makedirs(data_dir, exist_ok=True)

    print("decoder_prms =", decoder_prms)

    print(f"\n==== Starting simulations up to {total_shots} shots ====")
    for d, T, noise_model in prms:
        print(f"\n--- Simulating d={d}, T={T}, noise_model={noise_model} ---")
        p_depol = noise_model["depol"]
        p_meas = noise_model["meas"]
        simulate(
            shots=total_shots,
            d=d,
            T=T,
            data_dir=data_dir,
            sub_dirname=f"d{d}_T{T}_depol{p_depol:.3f}_meas{p_meas:.3f}",
            n_jobs=n_jobs,
            repeat=repeat,
            shots_per_batch=shots_per_batch,
            decoder_prms=decoder_prms,
            noise_model=noise_model,
        )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Simulations completed ({t0}) ====")
