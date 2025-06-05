import os
import warnings
from datetime import datetime
from typing import Any, Dict

import numpy as np

from simulations.simulation_utils import (
    _convert_df_dtypes_for_feather,
    _get_optimal_uint_dtype,
    get_existing_shots,
    task_parallel,
)
from simulations.build_circuit import build_BB_circuit, get_BB_distance


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
        batch_output_dir = os.path.join(sub_data_dir, f"batch_{next_idx}_{to_run}")

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

    plist = [1e-2]
    nlist = [72, 144]  # [72, 108, 144, 288]

    shots_per_batch = round(1e6)
    total_shots = round(1e7)

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
    data_dir = os.path.join(current_dir, "data/bb_minsum_iter30_lsd0")
    os.makedirs(data_dir, exist_ok=True)

    print("nlist =", nlist)
    print("plist =", plist)
    print("decoder_prms =", decoder_prms)

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
                n_jobs=19,
                repeat=10,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
            )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Simulations completed ({t0}) ====")
