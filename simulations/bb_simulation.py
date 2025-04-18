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


if __name__ == "__main__":
    plist = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    nlist = [144]
    shots = 19

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data/bb_circuit_iter30_minsum_lsd0")
    os.makedirs(data_dir, exist_ok=True)

    print("\n==== Starting simulations ====")

    for p in plist:
        for n in nlist:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[{current_time}] Starting simulation for p={p}, n={n}, shots={shots}."
            )
            df = task_parallel(shots, p, n, n_jobs=19, repeat=1)
            file_path = os.path.join(data_dir, f"n{n}_p{p}_test4.csv")
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

    print("\n==== Simulations completed ====")
