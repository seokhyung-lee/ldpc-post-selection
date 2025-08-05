"""
Real-time post-selection analysis for BB sliding window decoding simulations.

This script performs comprehensive post-selection analysis on existing BB sliding window
simulation data, evaluating the effectiveness of real-time post-selection strategies
without requiring re-simulation. It processes multiple parameter combinations and cutoff
configurations to find optimal operating points.
"""

import pickle
import numpy as np
from pathlib import Path

from simulations.analysis.data_collectors.data_collection import DATA_DIR
from simulations.analysis.data_collectors.sliding_window_post_selection import (
    batch_postselection_analysis,
)


def main():
    """Execute comprehensive real-time post-selection analysis for BB sliding window data."""

    print("=" * 60)
    print("REAL-TIME POST-SELECTION ANALYSIS")
    print("=" * 60)

    # =============================================================================
    # Configuration
    # =============================================================================

    # Data directory configuration
    data_dir_name = "bb_sliding_window_minsum_iter30_lsd0_raw"
    dataset_name = "bb_sliding_window"
    data_dir = str(DATA_DIR / data_dir_name)
    # subdirs = ["n144_T12_p0.003_W3_F1"]
    subdirs = [f"n144_T{T}_p0.003_W3_F1" for T in range(6, 25, 3)]

    # Configuration for post-selection analysis
    postselection_config = {
        "metric_windows": [3],
        "norm_orders": [2],
        "value_types": ["llr"],
        "num_jobs": 18,
        "verbose": True,
        "batch_mode": True,
        "stats_only": True,
    }

    # =============================================================================
    # Cutoff Array Definition
    # =============================================================================

    # Define cutoff array for post-selection analysis
    cutoffs = np.logspace(np.log10(0.004), -1, 10).round(6)

    print(
        f"Using cutoffs: {len(cutoffs)} points, range [{cutoffs[0]:.6f}, {cutoffs[-1]:.6f}]"
    )

    # =============================================================================
    # Analysis Execution
    # =============================================================================

    # Process parameter configurations
    base_results_dir = DATA_DIR / "real_time_post_selection"
    total_saved_files = []

    for metric_windows in postselection_config["metric_windows"]:
        for norm_order in postselection_config["norm_orders"]:
            for value_type in postselection_config["value_types"]:

                config_name = f"mw{metric_windows}_{value_type}_norm_frac_{norm_order}"
                print(f"\nConfiguration: {config_name}")

                # Run batch post-selection analysis
                batch_results = batch_postselection_analysis(
                    data_dir=data_dir,
                    param_combinations=subdirs,
                    cutoffs=cutoffs,
                    metric_windows=metric_windows,
                    norm_order=norm_order,
                    value_type=value_type,
                    num_jobs=postselection_config["num_jobs"],
                    verbose=postselection_config["verbose"],
                    batch_mode=postselection_config["batch_mode"],
                    stats_only=postselection_config["stats_only"],
                )

                # Save results immediately after processing this config
                if batch_results:
                    for subdir, results in batch_results.items():
                        if results:
                            # Create directory structure for this subdir
                            subdir_dir = base_results_dir / "bb" / subdir
                            subdir_dir.mkdir(parents=True, exist_ok=True)

                            # Save individual config results
                            config_path = subdir_dir / f"{config_name}.pkl"
                            with open(config_path, "wb") as f:
                                pickle.dump(results, f)

                            total_saved_files.append(str(config_path))
                            print(
                                f"Saved {config_name}/{subdir} results to: {config_path}"
                            )

                # Clear data from memory after saving
                del batch_results

    print(
        f"\nPost-selection analysis complete! Saved {len(total_saved_files)} result files."
    )

    # =============================================================================
    # Summary Generation
    # =============================================================================
    print("=" * 60)

    print(f"\nReal-time post-selection analysis complete!")
    print(f"Results directory: {base_results_dir}")

    # Calculate number of configurations processed
    num_configs = (
        len(postselection_config["metric_windows"])
        * len(postselection_config["norm_orders"])
        * len(postselection_config["value_types"])
    )

    print(
        f"Saved {len(total_saved_files)} result files across {num_configs} configurations"
    )

    # Show the organized file structure
    if total_saved_files:
        print("\nSaved file structure:")
        current_subdir = None
        for file_path in total_saved_files:
            subdir_name = str(Path(file_path).parent.name)  # Extract subdir name
            config_name = str(Path(file_path).stem)  # Extract config name
            if subdir_name != current_subdir:
                print(f"  bb/{subdir_name}/")
                current_subdir = subdir_name
            print(f"    {config_name}.pkl")


if __name__ == "__main__":
    main()
