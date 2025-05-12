import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import stim
from joblib import Parallel, delayed

from src.ldpc_post_selection.build_circuit import build_BB_circuit, get_BB_distance
from src.ldpc_post_selection.decoder import BpLsdPsDecoder


def _convert_df_dtypes_for_feather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts DataFrame column dtypes for optimized Feather storage.

    This function iterates through each column of the input DataFrame.
    If a column's data type is float, it's converted to `float32`.
    If a column's data type is integer or boolean, it's converted to `int32`.
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
        elif pd.api.types.is_integer_dtype(col_dtype):
            df[col_name] = df[col_name].astype(np.int32)
    return df


# def load_bb_circuit(p: float, n: int, T: int) -> stim.Circuit:
#     """
#     Load a Stim circuit from a file.

#     The function searches for a .stim file in the 'circuits/bb_codes/' directory.
#     The filename is expected to contain substrings matching "p={p}", "nkd=[[{n}",
#     and "r={T}".

#     Parameters
#     ----------
#     p : float
#         Circuit-level error probability.
#     n : int
#         Number of qubits.
#     T : int
#         Number of rounds.

#     Returns
#     -------
#     circuit : stim.Circuit
#         The loaded Stim circuit object.

#     Raises
#     ------
#     FileNotFoundError
#         If no file matching the criteria is found in 'circuits/bb_codes/'.
#     ValueError
#         If multiple files matching the criteria are found.
#     IOError
#         If the 'circuits/bb_codes/' directory does not exist.
#     """
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     circuits_dir = os.path.join(current_dir, "circuits/bb_codes")

#     if not os.path.isdir(circuits_dir):
#         raise IOError(
#             f"Directory not found: {circuits_dir}. Ensure this path is correct relative to your execution directory or an absolute path."
#         )

#     found_files = []
#     # Construct search patterns based on user's specifications
#     pattern_p = f"p={p}"
#     pattern_n = f"nkd=[[{n}"
#     pattern_T = f"r={T}"
#     pattern_c = "c=bivariate_bicycle_Z"

#     for filename in os.listdir(circuits_dir):
#         if (
#             pattern_p in filename
#             and pattern_n in filename
#             and pattern_T in filename
#             and pattern_c in filename
#             and filename.endswith(".stim")
#         ):
#             found_files.append(filename)

#     if not found_files:
#         raise FileNotFoundError(
#             f"No .stim file found in '{circuits_dir}' matching criteria: "
#             f"p={p}, nkd=[[{n}, r={T}"
#         )
#     if len(found_files) > 1:
#         raise ValueError(
#             f"Multiple .stim files found in '{circuits_dir}' matching criteria: "
#             f"p={p}, nkd=[[{n}, r={T}. Files: {found_files}"
#         )

#     circuit_file_path = os.path.join(circuits_dir, found_files[0])
#     return stim.Circuit.from_file(circuit_file_path)


def task(
    shots: int,
    p: float,
    n: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float | int | bool]]]:
    circuit = build_BB_circuit(p=p, n=n, T=T)
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots, separate_observables=True)

    decoder = BpLsdPsDecoder(circuit=circuit)
    preds_list = []
    preds_bp_list = []
    converges = []
    soft_infos = []

    for det_sng in det:
        pred, pred_bp, converge, soft_info = decoder.decode(det_sng, norm_order=0.5)

        preds_list.append(pred)
        preds_bp_list.append(pred_bp)

        converges.append(converge)
        soft_infos.append(soft_info)

    converges = np.array(converges)

    preds_arr = np.array(preds_list)
    preds_bp_arr = np.array(preds_bp_list)

    obs_matrix_T = decoder.obs_matrix.T
    obs_preds_arr = ((preds_arr.astype(np.uint8) @ obs_matrix_T) % 2).astype(bool)
    obs_preds_bp_arr = ((preds_bp_arr.astype(np.uint8) @ obs_matrix_T) % 2).astype(bool)

    # Compare with the true logical observables 'obs'
    fails = np.any(obs ^ obs_preds_arr, axis=1)
    fails_bp = np.any(obs ^ obs_preds_bp_arr, axis=1)

    return fails, fails_bp, converges, soft_infos


def task_parallel(
    shots: int, p: float, n: int, T: int, n_jobs: int, repeat: int = 10
) -> pd.DataFrame:
    """
    Run the BB `task` function in parallel and return results as a DataFrame.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate.
    p : float
        Error probability parameter.
    n : int
        Circuit parameter n.
    T : int
        Circuit parameter T.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the fail flag (0 or 1) and soft information for each sample.
    """
    # Divide shots among jobs
    base = shots // (n_jobs * repeat)
    remainder = shots % (n_jobs * repeat)
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_jobs * repeat)]

    # Execute tasks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(task)(shots=chunk, p=p, n=n, T=T) for chunk in chunk_sizes
    )

    # Unpack and combine results
    fails, fails_bp, converges, soft_infos = zip(*results)
    fails = pd.Series(np.concatenate(fails), name="fail")
    fails_bp = pd.Series(np.concatenate(fails_bp), name="fail_bp")
    converges = pd.Series(np.concatenate(converges), name="converge")
    soft_infos = [info for sublist in soft_infos for info in sublist]

    # Create DataFrame from soft information
    df_soft = pd.DataFrame(soft_infos)

    # Combine fail flag with soft info
    df = pd.concat([fails, fails_bp, converges, df_soft], axis=1)

    return df


def get_existing_shots(data_dir: str) -> Tuple[int, List[Tuple[int, str, int]]]:
    """
    Calculate the total number of shots already saved in Feather files within a given directory.

    Parameters
    ----------
    data_dir : str
        Directory containing the Feather files (e.g., "data/bb_circuit_iter30_minsum_lsd0/n72_T6_p0.002").

    Returns
    -------
    total_existing : int
        The total number of rows found across all 'data_*.feather' files.
    existing_files_info : list of tuple
        A list containing tuples of (index, file_path, row_count) for each existing file.
    """
    total_existing = 0
    idx = 1
    existing_files_info = []
    while True:
        # Changed filename pattern to data_{idx}.feather
        fp = os.path.join(data_dir, f"data_{idx}.feather")
        if not os.path.exists(fp):
            break
        try:
            # Read file to count rows. Optimization note: This reads the whole file.
            # For very large files, reading only metadata (if possible) or storing
            # row counts separately could be faster.
            data = pd.read_feather(fp)
            rows = len(data)
            del data  # Free memory
        except Exception as e:
            warnings.warn(
                f"Could not read or get row count for {fp}: {e}. Assuming 0 rows."
            )
            rows = 0
        total_existing += rows
        if rows > 0:  # Store info only if file could be read and has rows
            existing_files_info.append((idx, fp, rows))
        idx += 1
    return total_existing, existing_files_info


def simulate(
    shots: int,
    p: float,
    n: int,
    T: int,
    data_dir: str,
    n_jobs: int,
    repeat: int,
    max_shots_per_file: int = 1_000_000,
) -> None:
    """
    Run the simulation for a given (p, n) using Feather files and handle file I/O,
    storing results in subdirectories based on parameters.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate.
    p : float
        Error probability parameter.
    n : int
        Number of qubits.
    T : int
        Number of rounds.
    data_dir : str
        Base directory to store output subdirectories (e.g., "data/bb_circuit_iter30_minsum_lsd0").
    n_jobs : int
        Number of parallel jobs.
    repeat : int
        Number of repeats for parallel execution.
    max_shots_per_file : int
        Maximum number of shots per file.

    Returns
    -------
    None
        This function writes results to Feather files and prints status messages.
    """
    # Create subdirectory path based on parameters
    sub_dirname = f"n{n}_T{T}_p{p}"
    sub_data_dir = os.path.join(data_dir, sub_dirname)
    os.makedirs(
        sub_data_dir, exist_ok=True
    )  # Create the subdirectory if it doesn't exist

    # Count existing files and rows within the specific subdirectory
    total_existing, existing_files = get_existing_shots(sub_data_dir)

    # Skip if already simulated enough
    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, n={n}, T={T} in {sub_dirname}."
        )
        return

    remaining = shots - total_existing
    print(
        f"\nNeed to simulate {remaining} more shots for p={p}, n={n}, T={T} into {sub_dirname}"
    )

    # Append to existing files up to capacity within the subdirectory
    for idx, fp, rows in existing_files:
        if remaining <= 0:
            break
        if rows < max_shots_per_file:
            to_run = min(max_shots_per_file - rows, remaining)
            t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fname = os.path.basename(fp)  # fname will be "data_{idx}.feather"
            print(
                f"\n[{t0}] Simulating {to_run} shots for p={p}, n={n}, T={T}, appending to {sub_dirname}/{fname}"
            )
            # Load existing data only when appending
            try:
                df_existing = pd.read_feather(fp)
            except Exception as e:
                warnings.warn(
                    f"Could not read {fp} for appending: {e}. Starting from new shots for this file."
                )
                df_existing = pd.DataFrame()  # Start fresh for this file if read fails
            df_new = task_parallel(to_run, p, n, T, n_jobs=n_jobs, repeat=repeat)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = _convert_df_dtypes_for_feather(df_combined)
            df_combined.to_feather(fp)
            remaining -= to_run
            print(
                f"   Appended {to_run} shots. {remaining} shots remaining for this config."
            )

    # Create new numbered files within the subdirectory for any remaining shots
    # Start indexing from the next available index after existing files
    next_idx = (
        max([info[0] for info in existing_files], default=0) + 1
        if existing_files
        else 1
    )
    while remaining > 0:
        to_run = min(max_shots_per_file, remaining)
        # Use the new filename pattern "data_{idx}.feather" within the subdirectory
        fp_new = os.path.join(sub_data_dir, f"data_{next_idx}.feather")
        t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname_new = os.path.basename(
            fp_new
        )  # fname_new will be "data_{next_idx}.feather"
        print(
            f"\n[{t0}] Simulating {to_run} shots for p={p}, n={n}, T={T}, creating {sub_dirname}/{fname_new}"
        )
        df_new = task_parallel(to_run, p, n, T, n_jobs=n_jobs, repeat=repeat)
        df_new = _convert_df_dtypes_for_feather(df_new)
        df_new.to_feather(fp_new)
        remaining -= to_run
        next_idx += 1
        print(
            f"   Created file with {to_run} shots. {remaining} shots remaining for this config."
        )


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore", message="A worker stopped while some jobs were given to the executor."
    )

    plist = [1e-3, 3e-3, 5e-3]
    nlist = [144]  # [72, 108, 144, 288]

    max_shots_per_file = round(5e6)
    total_shots = round(1e8)

    # Estimated time (20 cores):
    # p=1e-3, n=144: 100,000 shots/min
    # p=3e-3, n=144: 50,000 shots/min
    # p=5e-3, n=144: 12,500 shots/min

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/bb_circuit_iter30_minsum_lsd0")
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n==== Starting simulations up to {total_shots} shots ====")
    for p in plist:
        for n in nlist:
            T = get_BB_distance(n)
            t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            simulate(
                shots=total_shots,
                p=p,
                n=n,
                T=T,
                data_dir=data_dir,
                n_jobs=20,
                repeat=10,
                max_shots_per_file=max_shots_per_file,
            )

    t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Simulations completed ({t0}) ====")
