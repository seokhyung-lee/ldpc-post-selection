import os
import warnings
import glob
import re
from datetime import datetime
from typing import Any, Dict

import numpy as np
import stim

from simulations.simulation_utils import (
    _convert_df_dtypes_for_feather,
    _get_optimal_uint_dtype,
    get_existing_shots,
    task_parallel,
)


def get_available_seeds(p: float, n: int, T: int) -> list[int]:
    """
    Scan the circuits directory and return a list of available seeds for given parameters.

    Parameters
    ----------
    p : float
        Physical error probability.
    n : int
        Number of qubits.
    T : int
        Number of rounds.

    Returns
    -------
    list of int
        List of available seed values that have corresponding circuit files.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuits_dir = os.path.join(current_dir, "data/hgp_circuits")

    if not os.path.exists(circuits_dir):
        return []

    # Create folder pattern for the given parameters
    # Expected folder format: (3,4)_n225_k9_d4_T4_p0.001
    folder_pattern = f"*_n{n}_*_T{T}_p{p}"

    available_seeds = []

    # Look for matching folders
    for item in os.listdir(circuits_dir):
        item_path = os.path.join(circuits_dir, item)
        if os.path.isdir(item_path):
            # Check if folder name matches our parameters
            if f"_n{n}_" in item and f"_T{T}_" in item and f"_p{p}" in item:
                # Look for seed files in this folder
                for filename in os.listdir(item_path):
                    if filename.startswith("seed") and filename.endswith(".stim"):
                        # Extract seed number
                        seed_match = re.match(r"seed(\d+)\.stim", filename)
                        if seed_match:
                            seed_value = int(seed_match.group(1))
                            available_seeds.append(seed_value)

    # Remove duplicates and sort
    available_seeds = sorted(list(set(available_seeds)))
    return available_seeds


def load_hgp_circuit(p: float, n: int, T: int, seed: int) -> stim.Circuit:
    """
    Load HGP circuit from stim file based on parameters.

    Parameters
    ----------
    p : float
        Physical error probability.
    n : int
        Number of qubits.
    T : int
        Number of rounds.
    seed : int
        Random seed for circuit generation.

    Returns
    -------
    stim.Circuit
        The loaded circuit from the matching stim file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuits_dir = os.path.join(current_dir, "data/hgp_circuits")

    if not os.path.exists(circuits_dir):
        raise FileNotFoundError(f"Circuits directory does not exist: {circuits_dir}")

    # Look for matching folder
    matching_folder = None
    for item in os.listdir(circuits_dir):
        item_path = os.path.join(circuits_dir, item)
        if os.path.isdir(item_path):
            # Check if folder name matches our parameters
            if f"_n{n}_" in item and f"_T{T}_" in item and f"_p{p}" in item:
                matching_folder = item_path
                break

    if matching_folder is None:
        raise FileNotFoundError(
            f"No folder found with parameters n{n}, T{T}, p{p} in {circuits_dir}"
        )

    # Look for the specific seed file in the matching folder
    seed_filename = f"seed{seed}.stim"
    circuit_file = os.path.join(matching_folder, seed_filename)

    if not os.path.exists(circuit_file):
        raise FileNotFoundError(f"No seed file found: {circuit_file}")

    print(f"Loading circuit from: {circuit_file}")
    circuit = stim.Circuit.from_file(circuit_file)
    return circuit


def simulate(
    shots: int,
    p: float,
    n: int,
    T: int,
    seed: int,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
) -> None:
    """
    Run the simulation for a given (p, n, T, seed) configuration, saving results in batches.
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
    seed : int
        Random seed for circuit generation.
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
    sub_dirname = f"n{n}_T{T}_p{p}_seed{seed}"
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, n={n}, T={T}, seed={seed} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for p={p}, n={n}, T={T}, seed={seed} into {sub_dirname}"
    )

    # Load the circuit from stim file
    circuit = load_hgp_circuit(p=p, n=n, T=T, seed=seed)
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
            f"\n[{t0_batch}] Simulating {to_run} shots for p={p}, n={n}, T={T}, seed={seed}. Output to: {batch_output_dir}"
        )

        df_new, flat_cluster_sizes, flat_cluster_llrs, offsets = task_parallel(
            shots=to_run,
            circuit=circuit,  # Pass the loaded circuit
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

    p = 1e-3
    n = 225
    T = 12

    shots_per_batch = round(1e6)
    total_shots = round(1e6)

    decoder_prms = {
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "lsd_method": "LSD_0",
        "lsd_order": 0,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/hgp_(3,4)_minsum_iter30_lsd0")
    os.makedirs(data_dir, exist_ok=True)

    print(f"n = {n}")
    print(f"T = {T}")
    print(f"p = {p}")
    print("decoder_prms =", decoder_prms)

    print(f"\n==== Starting HGP simulation up to {total_shots} shots per seed ====")

    # Get available seeds by scanning the circuits directory
    available_seeds = get_available_seeds(p=p, n=n, T=T)

    if len(available_seeds) == 0:
        print(f"No circuit files found for parameters p={p}, n={n}, T={T}")
        print("Simulation aborted.")
        exit(1)

    print(f"Found {len(available_seeds)} available seeds: {available_seeds}")

    successful_simulations = 0
    target_successful_simulations = len(available_seeds)

    for seed in available_seeds:
        if successful_simulations >= target_successful_simulations:
            print(
                f"\nReached target of {target_successful_simulations} successful simulations. Stopping."
            )
            break

        print(
            f"\n--- Trying seed = {seed} ({successful_simulations + 1}/{target_successful_simulations}) ---"
        )
        try:
            simulate(
                shots=total_shots,
                p=p,
                n=n,
                T=T,
                seed=seed,
                data_dir=data_dir,
                n_jobs=19,
                repeat=10,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
            )
            successful_simulations += 1
            print(
                f"✓ Seed {seed} completed successfully. Progress: {successful_simulations}/{target_successful_simulations}"
            )

        except FileNotFoundError as e:
            print(f"✗ Seed {seed} skipped: {e}")
        except Exception as e:
            print(f"✗ Seed {seed} failed with error: {e}")

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== HGP simulation completed ({t0}) ====")
    print(
        f"Successfully completed {successful_simulations} simulations out of {len(available_seeds)} available seeds."
    )
