"""
Ordinary (no post-selection) analysis for sliding window decoding.

This module implements simplified analysis of sliding window simulation data
without post-selection. It calculates basic failure statistics assuming all
samples are accepted, providing a baseline for comparison with post-selection
strategies.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Union

from simulations.analysis.plotting_helpers import get_confint


def load_fails_data(data_dir: str, param_combo: str) -> np.ndarray:
    """
    Load failure data from all batches for ordinary analysis.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combo : str
        Parameter combination string (e.g., "n144_T12_p0.003_W3_F1").

    Returns
    -------
    np.ndarray
        Boolean array of decoding failures from all batches combined.
    """
    param_dir = Path(data_dir) / param_combo

    # Find all batch directories
    batch_dirs = [
        d for d in param_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")
    ]

    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found in {param_dir}")

    # Load failure data from all batches
    all_fails = []
    for batch_dir in sorted(batch_dirs):
        fails = np.load(batch_dir / "fails.npy")
        all_fails.append(fails)

    # Concatenate data from all batches
    fails_combined = np.concatenate(all_fails)
    return fails_combined


def load_single_batch_fails(
    data_dir: str, param_combo: str, batch_dir_name: str
) -> np.ndarray:
    """
    Load failure data from a single batch for ordinary analysis.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combo : str
        Parameter combination string (e.g., "n144_T12_p0.003_W3_F1").
    batch_dir_name : str
        Name of the specific batch directory to load (e.g., "batch_1_1000000").

    Returns
    -------
    np.ndarray
        Boolean array of decoding failures for this batch.
    """
    param_dir = Path(data_dir) / param_combo
    batch_dir = param_dir / batch_dir_name

    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    # Load failure data for this specific batch
    fails = np.load(batch_dir / "fails.npy")
    return fails


def analyze_parameter_combination_ordinary(
    data_dir: str,
    param_combo: str,
) -> Dict[str, Union[float, int]]:
    """
    Perform ordinary (no post-selection) analysis for a single parameter combination.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combo : str
        Parameter combination string (e.g., "n144_T12_p0.003_W3_F1").

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary containing basic failure statistics:
        - 'p_fail': Failure rate
        - 'delta_p_fail': Margin of error for p_fail (95% confidence interval)
        - 'total_samples': Total number of samples
        - 'num_failed': Number of failed samples
    """
    # Load failure data
    fails = load_fails_data(data_dir, param_combo)

    # Calculate basic statistics
    total_samples = len(fails)
    num_failed = np.sum(fails)

    # Calculate failure rate and confidence interval
    if total_samples > 0:
        p_fail, delta_p_fail = get_confint(num_failed, total_samples)
    else:
        p_fail = 0.0
        delta_p_fail = 0.0

    return {
        "p_fail": p_fail,
        "delta_p_fail": delta_p_fail,
        "total_samples": total_samples,
        "num_failed": num_failed,
    }


def combine_ordinary_statistics(
    batch_stats_list: List[Dict[str, Union[float, int]]],
) -> Dict[str, Union[float, int]]:
    """
    Combine ordinary analysis statistics from multiple batches.

    Parameters
    ----------
    batch_stats_list : List[Dict[str, Union[float, int]]]
        List of statistics dictionaries from individual batches.

    Returns
    -------
    Dict[str, Union[float, int]]
        Combined statistics across all batches:
        - 'p_fail': Overall failure rate
        - 'delta_p_fail': Margin of error for p_fail (95% confidence interval)
        - 'total_samples': Total number of samples
        - 'num_failed': Total number of failed samples
    """
    if not batch_stats_list:
        raise ValueError("batch_stats_list cannot be empty")

    # Sum across all batches
    total_samples = sum(stats["total_samples"] for stats in batch_stats_list)
    total_failed = sum(stats["num_failed"] for stats in batch_stats_list)

    # Calculate combined failure rate and confidence interval
    if total_samples > 0:
        p_fail, delta_p_fail = get_confint(total_failed, total_samples)
    else:
        p_fail = 0.0
        delta_p_fail = 0.0

    return {
        "p_fail": p_fail,
        "delta_p_fail": delta_p_fail,
        "total_samples": total_samples,
        "num_failed": total_failed,
    }


def analyze_parameter_combination_batch_by_batch_ordinary(
    data_dir: str,
    param_combo: str,
) -> Dict[str, Union[float, int]]:
    """
    Perform batch-by-batch ordinary analysis for a single parameter combination.

    This function processes each batch individually to avoid memory issues with
    large datasets. For each batch, it loads failure data, computes statistics,
    and then combines them at the end.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combo : str
        Parameter combination string (e.g., "n144_T12_p0.003_W3_F1").

    Returns
    -------
    Dict[str, Union[float, int]]
        Combined ordinary analysis statistics across all batches.
    """
    param_dir = Path(data_dir) / param_combo

    # Find all batch directories
    batch_dirs = [
        d for d in param_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")
    ]

    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found in {param_dir}")

    batch_stats_list = []

    for batch_dir in sorted(batch_dirs):
        # Load failure data for this batch only
        fails = load_single_batch_fails(data_dir, param_combo, batch_dir.name)

        # Calculate statistics for this batch
        total_samples = len(fails)
        num_failed = np.sum(fails)

        if total_samples > 0:
            p_fail, delta_p_fail = get_confint(num_failed, total_samples)
        else:
            p_fail = 0.0
            delta_p_fail = 0.0

        batch_stats = {
            "p_fail": p_fail,
            "delta_p_fail": delta_p_fail,
            "total_samples": total_samples,
            "num_failed": num_failed,
        }
        batch_stats_list.append(batch_stats)

        # Clear memory - delete large objects
        del fails

    # Combine statistics from all batches
    combined_stats = combine_ordinary_statistics(batch_stats_list)
    return combined_stats


def batch_ordinary_analysis(
    data_dir: str,
    param_combinations: List[str],
    batch_mode: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    High-performance batch processing of multiple parameter combinations for ordinary analysis.

    Efficiently processes multiple sliding window parameter combinations without
    post-selection, calculating basic failure statistics. Supports memory-efficient
    batch-by-batch processing for large datasets.

    Parameters
    ----------
    data_dir : str
        Path to the raw sliding window data directory.
    param_combinations : List[str]
        List of parameter combination strings to process.
    batch_mode : bool, default=False
        If True, use batch-by-batch processing to handle large datasets that
        exceed memory capacity. Each batch is processed individually with
        statistics combined at the end. If False, load all batches at once.
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    Dict[str, Dict[str, Union[float, int]]]
        Nested dictionary with results for each parameter combination.
        Structure: {param_combo: {result_key: result_value}}

        Each parameter combination contains:
        - 'p_fail': Failure rate (float)
        - 'delta_p_fail': Margin of error for p_fail (95% confidence interval) (float)
        - 'total_samples': Total number of samples (int)
        - 'num_failed': Number of failed samples (int)
    """
    if verbose:
        mode_str = "batch-by-batch" if batch_mode else "all-at-once"
        print(
            f"Starting {mode_str} ordinary analysis for {len(param_combinations)} combinations"
        )
        print("Calculating basic failure statistics without post-selection")

    # Process combinations sequentially
    results_list = []
    for param_combo in param_combinations:
        if verbose:
            print(f"Processing {param_combo}...")

        if batch_mode:
            # Use batch-by-batch processing for memory efficiency
            results = analyze_parameter_combination_batch_by_batch_ordinary(
                data_dir=data_dir,
                param_combo=param_combo,
            )
        else:
            # Use all-at-once processing
            results = analyze_parameter_combination_ordinary(
                data_dir=data_dir,
                param_combo=param_combo,
            )

        results_list.append((param_combo, results))

    # Convert to dictionary
    results_dict = {combo: results for combo, results in results_list}

    if verbose:
        successful = sum(1 for _, results in results_list if results)
        print(
            f"Successfully processed {successful}/{len(param_combinations)} combinations"
        )

    return results_dict
