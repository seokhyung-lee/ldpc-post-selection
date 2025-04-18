# -*- coding: utf-8 -*-
"""
Script to run the BB simulation task function in parallel and save the results to a CSV file.
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from simulations.bb_simulation import task


def run_parallel_task(shots: int, p: float, n: int, n_jobs: int) -> pd.DataFrame:
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
    base = shots // n_jobs
    remainder = shots % n_jobs
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_jobs)]

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

    # Identify column types in soft info
    float_cols = df_soft.select_dtypes(include=["float"]).columns
    int_cols = df_soft.select_dtypes(include=["int"]).columns
    bool_cols = df_soft.select_dtypes(include=["bool"]).columns

    # Round float columns to two decimal places
    df_soft[float_cols] = df_soft[float_cols].round(2)

    # Convert integer columns to int type (no change)
    df_soft[int_cols] = df_soft[int_cols].astype(int)

    # Convert boolean columns to int (0 or 1)
    df_soft[bool_cols] = df_soft[bool_cols].astype(int)

    # Combine fail flag with soft info
    df = pd.concat([pd.Series(fails.astype(int), name="fail"), df_soft], axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run BB simulation in parallel and save results to CSV."
    )
    parser.add_argument(
        "--shots", type=int, required=True, help="Total number of shots"
    )
    parser.add_argument("--p", type=float, required=True, help="Error probability p")
    parser.add_argument("--n", type=int, required=True, help="Circuit parameter n")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--output", type=str, required=True, help="Output CSV file path"
    )

    args = parser.parse_args()

    df = run_parallel_task(args.shots, args.p, args.n, args.jobs)
    df.to_csv(args.output, index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
