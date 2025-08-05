import os
import warnings
import glob
import re
from datetime import datetime
from typing import Any, Dict

import numpy as np
import stim

from simulations.utils.simulation_utils import (
    _convert_df_dtypes_for_feather,
    _get_optimal_uint_dtype,
    get_existing_shots,
    bplsd_simulation_task_parallel,
)


def get_available_circuits(
    p: float, n: int, k: int, d: int, T: int
) -> list[tuple[str, str]]:
    """
    Scan the circuits directory and return a list of available circuits for given parameters.

    Parameters
    ----------
    p : float
        Physical error probability.
    n : int
        Number of qubits.
    k : int
        Number of logical qubits.
    d : int
        Code distance.
    T : int
        Number of rounds.

    Returns
    -------
    list of tuple[str, str]
        List of tuples containing (folder_name, circuit_filename_without_extension)
        for available circuit files.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuits_dir = os.path.join(current_dir, "data/hgp_prebuilt/circuits")

    if not os.path.exists(circuits_dir):
        return []

    # Create folder pattern for the given parameters
    # Expected folder format: (3,4)_n225_k9_d4_T4_p0.001
    available_circuits = []

    # Look for matching folders
    for item in os.listdir(circuits_dir):
        item_path = os.path.join(circuits_dir, item)
        if os.path.isdir(item_path):
            # Check if folder name matches our parameters
            if (
                f"_n{n}_" in item
                and f"_k{k}_" in item
                and f"_d{d}_" in item
                and f"_T{T}_" in item
                and f"_p{p}" in item
            ):
                # Look for circuit files directly in the folder
                for filename in os.listdir(item_path):
                    if filename.endswith(".stim"):
                        # Extract filename without extension
                        circuit_name = filename[:-5]  # Remove .stim extension
                        available_circuits.append((item, circuit_name))

    # Remove duplicates and sort by folder name then circuit name
    available_circuits = sorted(list(set(available_circuits)))
    return available_circuits


def load_hgp_circuit(
    p: float, n: int, k: int, d: int, T: int, circuit_name: str
) -> tuple[stim.Circuit, str, str]:
    """
    Load HGP circuit from stim file based on parameters and circuit name.

    Parameters
    ----------
    p : float
        Physical error probability.
    n : int
        Number of qubits.
    k : int
        Number of logical qubits.
    d : int
        Code distance.
    T : int
        Number of rounds.
    circuit_name : str
        Name of the circuit file without .stim extension.

    Returns
    -------
    stim.Circuit
        The loaded circuit from the matching stim file.
    str
        The folder name containing the circuit file.
    str
        The circuit filename without extension.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuits_dir = os.path.join(current_dir, "data/hgp_prebuilt/circuits")

    if not os.path.exists(circuits_dir):
        raise FileNotFoundError(f"Circuits directory does not exist: {circuits_dir}")

    # Look for matching folder
    matching_folder = None
    for item in os.listdir(circuits_dir):
        item_path = os.path.join(circuits_dir, item)
        if os.path.isdir(item_path):
            # Check if folder name matches our parameters
            if (
                f"_n{n}_" in item
                and f"_k{k}_" in item
                and f"_d{d}_" in item
                and f"_T{T}_" in item
                and f"_p{p}" in item
            ):
                matching_folder = item_path
                break

    if matching_folder is None:
        raise FileNotFoundError(
            f"No folder found with parameters n{n}, k{k}, d{d}, T{T}, p{p} in {circuits_dir}"
        )

        # Look for the specific circuit file in the matching folder
    circuit_filename = f"{circuit_name}.stim"
    circuit_file = os.path.join(matching_folder, circuit_filename)

    if not os.path.exists(circuit_file):
        raise FileNotFoundError(f"No circuit file found: {circuit_file}")

    print(f"Loading circuit from: {circuit_file}")
    circuit = stim.Circuit.from_file(circuit_file)

    # Extract folder name and circuit filename
    folder_name = os.path.basename(matching_folder)

    return circuit, folder_name, circuit_name


def simulate(
    shots: int,
    p: float,
    n: int,
    k: int,
    d: int,
    T: int,
    circuit_name: str,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    shots_per_batch: int = 1_000_000,
    decoder_prms: Dict[str, Any] | None = None,
) -> None:
    """
    Run the simulation for a given (p, n, k, d, T, circuit_name) configuration, saving results in batches.
    Results include a Feather file for scalar data and NumPy files for ragged arrays.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate for this configuration.
    p : float
        Physical error probability.
    n : int
        Number of qubits.
    k : int
        Number of logical qubits.
    d : int
        Code distance.
    T : int
        Number of rounds.
    circuit_name : str
        Name of the circuit file without .stim extension.
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
    # Load the circuit from stim file and get path information
    circuit, folder_name, circuit_name_verified = load_hgp_circuit(
        p=p, n=n, k=k, d=d, T=T, circuit_name=circuit_name
    )

    # Create subdirectory path based on circuit folder and circuit name
    sub_dirname = os.path.join(folder_name, circuit_name_verified)
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(sub_data_dir, exist_ok=True)

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files_info = get_existing_shots(sub_data_dir)

    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, n={n}, k={k}, d={d}, T={T}, circuit={circuit_name} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for p={p}, n={n}, k={k}, d={d}, T={T}, circuit={circuit_name} into {sub_dirname}"
    )
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
            f"\n[{t0_batch}] Simulating {to_run} shots for p={p}, n={n}, k={k}, d={d}, T={T}, circuit={circuit_name}. Output to: {batch_output_dir}"
        )

        df_new, flat_cluster_sizes, flat_cluster_llrs, offsets = (
            bplsd_simulation_task_parallel(
                shots=to_run,
                circuit=circuit,  # Pass the loaded circuit
                n_jobs=n_jobs,
                repeat=repeat,
                decoder_prms=decoder_prms,
            )
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
    k = 9
    d = 6
    T = 6

    shots_per_batch = round(1e6)
    total_shots = round(1e7)

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
    print(f"k = {k}")
    print(f"d = {d}")
    print(f"T = {T}")
    print(f"p = {p}")
    print("decoder_prms =", decoder_prms)

    print(f"\n==== Starting HGP simulation up to {total_shots} shots per circuit ====")

    # Get available circuits by scanning the circuits directory
    available_circuits = get_available_circuits(p=p, n=n, k=k, d=d, T=T)

    if len(available_circuits) == 0:
        print(
            f"No circuit files found for parameters p={p}, n={n}, k={k}, d={d}, T={T}"
        )
        print("Simulation aborted.")
        exit(1)

    print(f"Found {len(available_circuits)} available circuits:")
    for folder_name, circuit_name in available_circuits:
        print(f"  {folder_name}/{circuit_name}.stim")

    successful_simulations = 0
    target_successful_simulations = len(available_circuits)

    for folder_name, circuit_name in available_circuits:
        if successful_simulations >= target_successful_simulations:
            print(
                f"\nReached target of {target_successful_simulations} successful simulations. Stopping."
            )
            break

        print(
            f"\n--- Trying circuit = {folder_name}/{circuit_name}.stim ({successful_simulations + 1}/{target_successful_simulations}) ---"
        )
        try:
            simulate(
                shots=total_shots,
                p=p,
                n=n,
                k=k,
                d=d,
                T=T,
                circuit_name=circuit_name,
                data_dir=data_dir,
                n_jobs=19,
                repeat=10,
                shots_per_batch=shots_per_batch,
                decoder_prms=decoder_prms,
            )
            successful_simulations += 1
            print(
                f"✓ Circuit {circuit_name} completed successfully. Progress: {successful_simulations}/{target_successful_simulations}"
            )

        except FileNotFoundError as e:
            print(f"✗ Circuit {circuit_name} skipped: {e}")
        except Exception as e:
            print(f"✗ Circuit {circuit_name} failed with error: {e}")

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== HGP simulation completed ({t0}) ====")
    print(
        f"Successfully completed {successful_simulations} simulations out of {len(available_circuits)} available circuits."
    )
