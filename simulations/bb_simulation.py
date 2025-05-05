import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.ldpc_post_selection.build_circuit import build_BB_circuit, get_BB_distance
from src.ldpc_post_selection.decoder import BpLsdPsDecoder


def task(
    shots: int,
    p: float,
    n: int,
    T: int,
) -> Tuple[np.ndarray, List[Dict[str, float | int | bool]]]:
    if shots == 0:
        return np.array([], dtype=bool), []

    circuit = build_BB_circuit(n=n, T=T, p=p)
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(shots, separate_observables=True)

    decoder = BpLsdPsDecoder(circuit=circuit)
    preds = []
    soft_infos = []
    for det_sng in det:
        pred, soft_info = decoder.decode(det_sng)
        preds.append(pred)
        soft_infos.append(soft_info)
    preds = np.stack(preds, axis=0)
    obs_preds = ((preds.astype("uint8") @ decoder.obs_matrix.T) % 2).astype(bool)
    fails = np.any(obs ^ obs_preds, axis=1)

    return fails, soft_infos


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
    fails_list, soft_infos_list = zip(*results)
    fails = np.concatenate(fails_list)
    soft_infos_flat = [info for sublist in soft_infos_list for info in sublist]

    # Create DataFrame from soft information
    df_soft = pd.DataFrame(soft_infos_flat)

    # Combine fail flag with soft info
    df = pd.concat([pd.Series(fails, name="fail"), df_soft], axis=1)

    return df


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
    Run the simulation for a given (p, n) using Feather files and handle file I/O.

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
    data_dir : str
        Directory to store output Feather files.
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
    basename = f"n{n}_T{T}_p{p}"

    # Count existing numbered Feather files and total rows without loading full DataFrames
    existing_files = []  # list of (index, path, rows)
    total_existing = 0
    idx = 1
    while True:
        fp = os.path.join(data_dir, f"{basename}_{idx}.feather")
        if not os.path.exists(fp):
            break
        try:
            data = pd.read_feather(fp)
            rows = len(data)
            del data
        except Exception as e:
            warnings.warn(f"Could not read metadata for {fp}: {e}. Assuming 0 rows.")
            rows = 0
        existing_files.append((idx, fp, rows))
        total_existing += rows
        idx += 1

    # Skip if already simulated enough
    if total_existing >= shots:
        print(
            f"\n[SKIP] Already have {total_existing} shots (>= {shots}). Skipping p={p}, n={n}, T={T}."
        )
        return

    remaining = shots - total_existing
    # Append to existing files up to capacity, loading each file only when needed
    for idx, fp, rows in existing_files:
        if remaining <= 0:
            break
        if rows < max_shots_per_file:
            to_run = min(max_shots_per_file - rows, remaining)
            t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fname = os.path.basename(fp)
            print(
                f"\n[{t0}] Simulating {to_run} shots for p={p}, n={n}, T={T}, appending to {fname}"
            )
            # Load existing data only when appending
            try:
                df_existing = pd.read_feather(fp)
            except Exception as e:
                warnings.warn(
                    f"Could not read {fp} for appending: {e}. Starting from new shots."
                )
                df_existing = pd.DataFrame()
            df_new = task_parallel(to_run, p, n, T, n_jobs=n_jobs, repeat=repeat)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_feather(fp)
            remaining -= to_run

    # Create new numbered files for any remaining shots
    next_idx = len(existing_files) + 1
    while remaining > 0:
        to_run = min(max_shots_per_file, remaining)
        fp_new = os.path.join(data_dir, f"{basename}_{next_idx}.feather")
        t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname_new = os.path.basename(fp_new)
        print(
            f"\n[{t0}] Simulating {to_run} shots for p={p}, n={n}, T={T}, creating {fname_new}"
        )
        df_new = task_parallel(to_run, p, n, T, n_jobs=n_jobs, repeat=repeat)
        df_new.to_feather(fp_new)
        remaining -= to_run
        next_idx += 1


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore", message="A worker stopped while some jobs were given to the executor."
    )

    plist = [2e-3, 3e-3, 4e-3, 5e-3]
    # nlist = [72, 108, 144, 288]
    # plist = [1e-3]
    nlist = [144]

    max_shots_per_file = round(2e6)
    total_shots = round(1e8)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/bb_circuit_iter30_minsum_lsd0")
    os.makedirs(data_dir, exist_ok=True)

    for shots in range(0, total_shots + 1, max_shots_per_file)[1:]:
        print(f"\n==== Starting simulations for {shots} shots ====")
        for p in plist:
            for n in nlist:
                T = get_BB_distance(n)
                t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                simulate(
                    shots,
                    p,
                    n,
                    T,
                    data_dir,
                    n_jobs=19,
                    repeat=10,
                    max_shots_per_file=max_shots_per_file,
                )

        print(f"\n==== Simulations completed for {shots} shots ====")
