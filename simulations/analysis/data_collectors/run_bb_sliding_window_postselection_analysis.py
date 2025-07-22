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
    subdirs = ["n144_T12_p0.005_W3_F1"]

    # Configuration for post-selection analysis
    postselection_config = {
        "metric_windows": [3],  # Different window sizes to test
        "norm_orders": [2],  # L2 norm (can extend to [1.0, 2.0, np.inf])
        "value_types": ["llr"],  # Focus on LLR-based analysis
        "num_jobs": 18,  # Parallel processing
        "verbose": True,
    }

    # =============================================================================
    # Cutoff Array Definition
    # =============================================================================

    # Define cutoff array for post-selection analysis
    cutoffs = np.logspace(-3, -1, 20).round(6)

    print(
        f"Using cutoffs: {len(cutoffs)} points, range [{cutoffs[0]:.6f}, {cutoffs[-1]:.6f}]"
    )

    # =============================================================================
    # Analysis Execution
    # =============================================================================

    # Process parameter configurations
    postselection_results = {}

    for metric_windows in postselection_config["metric_windows"]:
        for norm_order in postselection_config["norm_orders"]:
            for value_type in postselection_config["value_types"]:

                config_name = f"mw{metric_windows}_ord{norm_order}_{value_type}"
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
                )

                # Store results
                postselection_results[config_name] = batch_results

    # =============================================================================
    # Results Saving
    # =============================================================================

    # Save results for each subdir separately
    base_results_dir = DATA_DIR / "real_time_post_selection"
    saved_files = []

    for config_name, batch_results in postselection_results.items():
        if not batch_results:
            continue

        # Create directory structure for this config
        config_dir = base_results_dir / config_name / "bb"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Save each subdir's results separately
        for subdir, results in batch_results.items():
            if not results:
                continue

            # Save individual subdir results
            subdir_path = config_dir / f"{subdir}.pkl"
            with open(subdir_path, "wb") as f:
                pickle.dump(results, f)

            saved_files.append(str(subdir_path))
            print(f"Saved {config_name}/{subdir} results to: {subdir_path}")

    print(f"\nPost-selection analysis complete! Saved {len(saved_files)} result files.")

    # =============================================================================
    # Summary Generation
    # =============================================================================

    # Generate summary statistics
    print("\n" + "=" * 60)
    print("POST-SELECTION ANALYSIS SUMMARY")
    print("=" * 60)

    for config_name, batch_results in postselection_results.items():
        if not batch_results:
            continue

        print(f"\n{config_name}:")

        for param_combo, results in batch_results.items():
            if not results:
                continue

            cutoffs = results["cutoffs"]
            p_abort = results["p_abort"]
            effective_avg_trials = results["effective_avg_trials"]

            # Find interesting operating points
            low_abort_idx = np.argmax(p_abort <= 0.05)  # ~5% abort rate
            med_abort_idx = np.argmax(p_abort <= 0.1)  # ~10% abort rate
            high_abort_idx = np.argmax(p_abort <= 0.2)  # ~20% abort rate

            print(f"  {param_combo}: Operating points")
            for desc, idx in [
                ("5% abort", low_abort_idx),
                ("10% abort", med_abort_idx),
                ("20% abort", high_abort_idx),
            ]:
                if idx > 0 and idx < len(cutoffs):
                    p_fail_val = results["p_fail"][idx]
                    delta_p_fail_val = results["delta_p_fail"][idx]
                    print(
                        f"    {desc}: cutoff={cutoffs[idx]:.4f}, "
                        f"LER={p_fail_val:.2e}±{delta_p_fail_val:.2e}, "
                        f"trials={effective_avg_trials[idx]:.2f}"
                    )

    print(f"\nReal-time post-selection analysis complete!")
    print(f"Results directory: {base_results_dir}")
    print(
        f"Saved {len(saved_files)} result files across {len([r for r in postselection_results.values() if r])} configurations"
    )

    # Show the organized file structure
    if saved_files:
        print("\nSaved file structure:")
        current_config = None
        for file_path in saved_files:
            config_part = str(Path(file_path).parent.parent.name)  # Extract config name
            subdir_name = str(Path(file_path).stem)  # Extract subdir name
            if config_part != current_config:
                print(f"  {config_part}/")
                current_config = config_part
            print(f"    bb/{subdir_name}.pkl")


if __name__ == "__main__":
    main()
