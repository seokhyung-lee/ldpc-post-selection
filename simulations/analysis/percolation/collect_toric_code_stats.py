import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import pandas as pd

from simulations.toric_code_bitflip_simulation import (
    load_single_simulation_data,
    get_existing_simulation_params,
)
from simulations.analysis.percolation.toric_code_percolation import ToricCodePercolation


# Utility functions
def write_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def process_single_param(d: int, p: float, stats_dir: str) -> tuple[int, float] | None:
    """
    Process a single (d, p) parameter combination and save results.

    Parameters
    ----------
    d : int
        Code distance
    p : float
        Error probability
    stats_dir : str
        Directory to save statistics

    Returns
    -------
    tuple of (int, float) or None
        Returns (d, p) if processing is completed, None if skipped
    """
    samples_file = os.path.join(stats_dir, f"d{d}_p{p:.3f}_samples.pkl")
    density_file = os.path.join(stats_dir, f"d{d}_p{p:.3f}_density.pkl")

    fails, clusters = load_single_simulation_data(d=d, p=p)
    current_shots = len(fails)

    # Check if we need to update this (d, p) combination
    needs_update = True
    # if os.path.exists(samples_file):
    #     try:
    #         existing_df = read_pickle(samples_file)
    #         existing_shots = len(existing_df)
    #         if existing_shots == current_shots:
    #             needs_update = False
    #     except:
    #         # If file is corrupted or unreadable, we need to update
    #         needs_update = True

    if not needs_update:
        return None

    percolation_analyzer = ToricCodePercolation(d)

    stats_dp = {"fail": fails}
    # stats_dp["perc"] = percolation_analyzer.check_percolation_batch(clusters)

    for k in [1, 2]:
        kth_moment_estimation = (
            percolation_analyzer.calculate_kth_moment_estimation_batch(clusters, k)
        )
        stats_dp[f"moment_{k}"] = kth_moment_estimation
    stats_dp["max_cluster_size"] = (
        percolation_analyzer.calculate_max_cluster_size_batch(clusters)
    )

    # Calculate correlation length for each sample
    corr_lengths, _ = percolation_analyzer.calculate_sample_corr_length_batch(clusters)
    stats_dp["corr_length"] = corr_lengths

    # Calculate correlation length for each sample (manhattan distance)
    corr_lengths_manhattan, _ = percolation_analyzer.calculate_sample_corr_length_batch(
        clusters, use_manhattan=True
    )
    stats_dp["corr_length_manhattan"] = corr_lengths_manhattan

    cluster_sizes, cluster_number_density = (
        percolation_analyzer.calculate_cluster_number_density_batch(clusters)
    )

    # Save df_stats
    df_stats = pd.DataFrame(stats_dp)
    write_pickle(samples_file, df_stats)

    # Save cluster number density
    df_density = pd.DataFrame(
        {
            "cluster_sizes": cluster_sizes,
            "cluster_number_density": cluster_number_density,
        }
    )
    write_pickle(density_file, df_density)

    return d, p


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curr_dir, "../data/toric_minsum_iter30_lsd0_bitflip_raw")
    stats_dir = os.path.join(data_dir, "stats")

    # Create stats directory if it doesn't exist
    os.makedirs(stats_dir, exist_ok=True)

    plist_to_take = np.arange(0.01, 0.1101, 0.005).round(3)

    params = get_existing_simulation_params()
    params = [(d, p) for d, p in params if p in plist_to_take]

    # Process parameters in parallel
    results = Parallel(n_jobs=18)(
        delayed(process_single_param)(d, p, stats_dir) for d, p in tqdm(params)
    )

    # Count processed results
    processed_count = sum(1 for result in results if result is not None)
    print(f"Processed {processed_count} parameter combinations")
