"""
Real-time post-selection analysis for surface sliding window decoding simulations.

This script performs comprehensive analysis on existing surface sliding window
simulation data, including both ordinary (no post-selection) baseline analysis
and real-time post-selection analysis. It evaluates the effectiveness of
real-time post-selection strategies without requiring re-simulation.
"""

import pickle
import numpy as np
from pathlib import Path

from simulations.analysis.data_collectors.data_collection import DATA_DIR
from simulations.analysis.data_collectors.sliding_window_post_selection import (
    batch_postselection_analysis,
)
from simulations.analysis.data_collectors.sliding_window_ordinary import (
    batch_ordinary_analysis,
)


def main():
    """Execute comprehensive sliding window analysis including ordinary and post-selection analysis."""

    print("=" * 60)
    print("SLIDING WINDOW ANALYSIS")
    print("=" * 60)

    # =============================================================================
    # Configuration
    # =============================================================================

    # Data directory configuration
    data_dir_name = "surface_sliding_window_minsum_iter30_lsd0_raw"
    dataset_name = "surface_sliding_window"
    data_dir = str(DATA_DIR / data_dir_name)
    subdirs = ["d13_T13_p0.003_W5_F1"]

    # Analysis mode configuration
    # Options: "ordinary_only", "postselection_only", "both"
    analysis_mode = "both"

    # Configuration for ordinary analysis
    ordinary_config = {
        "verbose": True,
        "batch_mode": True,
    }

    # Configuration for post-selection analysis
    postselection_config = {
        "metric_windows": [1, 2, 3, 5, 7],
        "norm_orders": [2],
        "value_types": ["llr"],
        "num_jobs": 8,
        "verbose": True,
        "batch_mode": True,
        "stats_only": True,
    }

    # =============================================================================
    # Cutoff Array Definition (for post-selection analysis)
    # =============================================================================

    # Define cutoff array for post-selection analysis
    cutoffs = np.logspace(-3.5, -1, 20).round(6)

    if analysis_mode in ["postselection_only", "both"]:
        print(
            f"Using cutoffs: {len(cutoffs)} points, range [{cutoffs[0]:.6f}, {cutoffs[-1]:.6f}]"
        )

    # =============================================================================
    # Analysis Execution
    # =============================================================================

    # Process parameter configurations
    base_results_dir = DATA_DIR / "real_time_post_selection"
    total_saved_files = []

    print(f"\nAnalysis mode: {analysis_mode}")

    # =============================================================================
    # Ordinary Analysis (no post-selection)
    # =============================================================================

    if analysis_mode in ["ordinary_only", "both"]:
        print("\n" + "=" * 40)
        print("ORDINARY ANALYSIS (NO POST-SELECTION)")
        print("=" * 40)

        # Run ordinary analysis
        ordinary_results = batch_ordinary_analysis(
            data_dir=data_dir,
            param_combinations=subdirs,
            batch_mode=ordinary_config["batch_mode"],
            verbose=ordinary_config["verbose"],
        )

        # Save ordinary results
        if ordinary_results:
            for subdir, results in ordinary_results.items():
                if results:
                    # Create directory structure for this subdir
                    subdir_dir = base_results_dir / "surface" / subdir
                    subdir_dir.mkdir(parents=True, exist_ok=True)

                    # Save ordinary analysis results
                    ordinary_path = subdir_dir / "ordinary.pkl"
                    with open(ordinary_path, "wb") as f:
                        pickle.dump(results, f)

                    total_saved_files.append(str(ordinary_path))
                    print(f"Saved ordinary analysis results to: {ordinary_path}")

        # Clear data from memory after saving
        del ordinary_results

    # =============================================================================
    # Post-Selection Analysis
    # =============================================================================

    if analysis_mode in ["postselection_only", "both"]:
        print("\n" + "=" * 40)
        print("POST-SELECTION ANALYSIS")
        print("=" * 40)

        for metric_windows in postselection_config["metric_windows"]:
            for norm_order in postselection_config["norm_orders"]:
                for value_type in postselection_config["value_types"]:

                    config_name = (
                        f"mw{metric_windows}_{value_type}_norm_frac_{norm_order}"
                    )
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
                                subdir_dir = base_results_dir / "surface" / subdir
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

    print(f"\nAnalysis complete! Saved {len(total_saved_files)} result files.")

    # =============================================================================
    # Summary Generation
    # =============================================================================
    print("=" * 60)

    print(f"\nSliding window analysis complete!")
    print(f"Analysis mode: {analysis_mode}")
    print(f"Results directory: {base_results_dir}")

    # Calculate number of configurations processed
    num_ordinary_configs = 1 if analysis_mode in ["ordinary_only", "both"] else 0
    num_postselection_configs = (
        len(postselection_config["metric_windows"])
        * len(postselection_config["norm_orders"])
        * len(postselection_config["value_types"])
        if analysis_mode in ["postselection_only", "both"]
        else 0
    )
    total_configs = num_ordinary_configs + num_postselection_configs

    print(
        f"Saved {len(total_saved_files)} result files across {total_configs} configurations"
    )

    if analysis_mode in ["ordinary_only", "both"]:
        print(f"  - Ordinary analysis: {num_ordinary_configs} configuration")
    if analysis_mode in ["postselection_only", "both"]:
        print(
            f"  - Post-selection analysis: {num_postselection_configs} configurations"
        )

    # Show the organized file structure
    if total_saved_files:
        print("\nSaved file structure:")
        current_subdir = None
        for file_path in total_saved_files:
            subdir_name = str(Path(file_path).parent.name)  # Extract subdir name
            config_name = str(Path(file_path).stem)  # Extract config name
            if subdir_name != current_subdir:
                print(f"  surface/{subdir_name}/")
                current_subdir = subdir_name
            print(f"    {config_name}.pkl")


if __name__ == "__main__":
    main()
