import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.ldpc_post_selection.build_circuit import build_BB_circuit, get_BB_distance
from src.ldpc_post_selection.decoder import BpLsdPsDecoder


def task(
    shots: int, p: float, n: int
) -> Tuple[np.ndarray, List[Dict[str, float | int | bool]]]:
    if shots == 0:
        return np.array([], dtype=bool), []

    d = get_BB_distance(n)
    circuit = build_BB_circuit(n=n, T=d, p=p)
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
    shots: int, p: float, n: int, n_jobs: int, repeat: int = 10
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
        delayed(task)(shots=chunk, p=p, n=n) for chunk in chunk_sizes
    )

    # Unpack and combine results
    fails_list, soft_infos_list = zip(*results)
    fails = np.concatenate(fails_list)
    soft_infos_flat = [info for sublist in soft_infos_list for info in sublist]

    # Create DataFrame from soft information
    df_soft = pd.DataFrame(soft_infos_flat)

    # Sort columns of df_soft in alphabetical order (for consistent csv save)
    df_soft = df_soft.reindex(sorted(df_soft.columns), axis=1)

    # Combine fail flag with soft info
    df = pd.concat([pd.Series(fails, name="fail"), df_soft], axis=1)

    # Identify column types in soft info
    float_cols = df.select_dtypes(include=["float"]).columns
    # int_cols = df_soft.select_dtypes(include=["int"]).columns
    bool_cols = df.select_dtypes(include=["bool"]).columns

    # Round float columns to two decimal places
    df[float_cols] = df[float_cols].round(2)

    # Convert integer columns to int type (no change)
    # df_soft[int_cols] = df_soft[int_cols].astype(int)

    # Convert boolean columns to int (0 or 1)
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def simulate(
    shots: int, p: float, n: int, data_dir: str, n_jobs: int, repeat: int
) -> None:
    """
    Run the simulation for a given (p, n) and handle file I/O and skipping logic.

    Parameters
    ----------
    shots : int
        Total number of shots to simulate.
    p : float
        Error probability parameter.
    n : int
        Circuit parameter n.
    data_dir : str
        Directory to store output CSV files.
    n_jobs : int
        Number of parallel jobs.
    repeat : int
        Number of repeats for parallel execution.

    Returns
    -------
    None
        This function writes results to a CSV file and prints status messages.
    """
    file_path = os.path.join(data_dir, f"n{n}_p{p}.csv")
    shots_to_run = shots
    existing_rows = 0
    if os.path.exists(file_path):
        try:
            # Only read the first column to minimize memory usage
            existing_rows = pd.read_csv(file_path, usecols=[0]).shape[0]
        except Exception as e:
            print(
                f"Warning: Could not read {file_path} due to error: {e}. Assuming 0 existing rows."
            )
            existing_rows = 0
        if existing_rows >= shots:
            print(
                f"[SKIP] {existing_rows} shots (>= {shots}). Skipping simulation for p={p}, n={n}."
            )
            return
        else:
            shots_to_run = shots - existing_rows
    else:
        pass

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"\n[{current_time}] Starting simulation for p={p}, n={n}, shots={shots_to_run}."
    )
    df = task_parallel(shots_to_run, p, n, n_jobs=n_jobs, repeat=repeat)
    if os.path.exists(file_path):
        # If file exists, append without header
        df.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False,
            float_format="%.2f",
        )
    else:
        # If file doesn't exist, create new file with header
        df.to_csv(
            file_path,
            index=False,
            float_format="%.2f",
        )


if __name__ == "__main__":
    plist = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    nlist = [144]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/bb_circuit_iter30_minsum_lsd0")
    os.makedirs(data_dir, exist_ok=True)

    for shots in range(round(1e5), round(1e6) + 1, round(1e5)):
        print(f"\n==== Starting simulations for {shots} shots ====")

        for p in plist:
            for n in nlist:
                simulate(shots, p, n, data_dir, n_jobs=19, repeat=100)

        print(f"\n==== Simulations completed for {shots} shots ====")
