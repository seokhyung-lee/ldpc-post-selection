#!/usr/bin/env python3
"""
Convert existing aggregated data files to the new format.

This script converts the old aggregated data format (all parameter combinations in one dictionary)
to the new format where each parameter combination is stored separately for use with the
updated `aggregate_data` function's `df_existing` parameter.

Old format: simulations/data/aggregated/df_agg_dict_{dataset}_bin{bins}.pkl
New format: simulations/data/aggregated/{method}/{param_combination}.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def load_existing_aggregated_file(file_path: str) -> Dict[str, Any]:
    """
    Load an existing aggregated data file.

    Parameters
    ----------
    file_path : str
        Path to the existing pkl file.

    Returns
    -------
    dict
        Dictionary containing the aggregated data.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def determine_dataset_and_method_from_key(
    method_key: str, filename: str
) -> tuple[str, str]:
    """
    Determine the dataset name and clean method name from the method key and filename.

    Parameters
    ----------
    method_key : str
        Method key from the dictionary (e.g., 'pred_llr', 'matching_gap').
    filename : str
        Original filename to help determine dataset type.

    Returns
    -------
    dataset : str
        Dataset name ('bb', 'surface', or 'surface_matching').
    clean_method : str
        Clean method name (with 'matching_' prefix removed if applicable).
    """
    # Determine if this is a matching method
    is_matching = method_key.startswith("matching_")

    # Determine dataset based on filename and method
    if "bb" in filename:
        dataset = "bb"
        clean_method = method_key
    elif "surface" in filename:
        if is_matching:
            dataset = "surface_matching"
            clean_method = method_key.replace(
                "matching_", ""
            )  # Remove matching_ prefix
        else:
            dataset = "surface"
            clean_method = method_key
    else:
        # Fallback logic based on method name
        if is_matching:
            dataset = "surface_matching"
            clean_method = method_key.replace("matching_", "")
        else:
            dataset = "unknown"
            clean_method = method_key

    return dataset, clean_method


def create_raw_data_style_filename(param_tuple: tuple, dataset: str) -> str:
    """
    Create a filename in the style of raw data subdirectories.

    Parameters
    ----------
    param_tuple : tuple
        Tuple representing parameter combination from MultiIndex.
    dataset : str
        Dataset name to determine the parameter naming convention.

    Returns
    -------
    str
        Filename in raw data style (e.g., 'n144_T12_p0.001', 'd13_T13_p0.003').
    """
    if dataset == "bb":
        # BB dataset uses: n, T, p
        # Example: n144_T12_p0.001
        if len(param_tuple) >= 3:
            n, T, p = param_tuple[:3]
            return f"n{n}_T{T}_p{p}"
        else:
            # Fallback if not enough parameters
            return "_".join([str(x) for x in param_tuple])
    elif dataset in ["surface", "surface_matching"]:
        # Surface datasets use: d, T, p
        # Example: d13_T13_p0.001
        if len(param_tuple) >= 3:
            d, T, p = param_tuple[:3]
            return f"d{d}_T{T}_p{p}"
        else:
            # Fallback if not enough parameters
            return "_".join([str(x) for x in param_tuple])
    else:
        # Unknown dataset, use generic format
        str_parts = []
        for item in param_tuple:
            if isinstance(item, (int, float)):
                if isinstance(item, float):
                    # Format floats to avoid issues with very small numbers
                    if item == 0:
                        str_parts.append("0")
                    elif abs(item) >= 1:
                        str_parts.append(f"{item:g}")
                    else:
                        str_parts.append(f"{item:.6g}")
                else:
                    str_parts.append(str(item))
            else:
                str_parts.append(str(item))

        clean_name = "_".join(str_parts)

        # Replace problematic characters for filesystem
        clean_name = clean_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        clean_name = clean_name.replace(" ", "_").replace("(", "").replace(")", "")
        clean_name = clean_name.replace(",", "_").replace("=", "").replace(".", "p")

        # Remove consecutive underscores
        while "__" in clean_name:
            clean_name = clean_name.replace("__", "_")

        # Remove leading/trailing underscores
        clean_name = clean_name.strip("_")

        return clean_name


def determine_aggregation_method_from_key(method_key: str) -> str:
    """
    Determine the aggregation method from the dictionary key.

    Parameters
    ----------
    method_key : str
        Method key from the dictionary (e.g., 'pred_llr', 'cluster_size_norm_0.5').

    Returns
    -------
    str
        The base aggregation method name.
    """
    # Handle matching methods first
    if method_key.startswith("matching_"):
        base_method = method_key.replace("matching_", "")
        return base_method

    # Handle norm-based methods with parameters
    if method_key.startswith("cluster_size_norm_gap_"):
        return "cluster_size_norm_gap"
    elif method_key.startswith("cluster_llr_norm_gap_"):
        return "cluster_llr_norm_gap"
    elif method_key.startswith("cluster_size_norm_"):
        return "cluster_size_norm"
    elif method_key.startswith("cluster_llr_norm_"):
        return "cluster_llr_norm"
    else:
        # For other methods, use the key as-is
        return method_key


def split_dataframe_by_parameter_combinations(
    df: pd.DataFrame, aggregation_method: str
) -> Dict[tuple, pd.DataFrame]:
    """
    Split a MultiIndex DataFrame into separate DataFrames for each parameter combination.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex containing parameter combinations.
    aggregation_method : str
        The aggregation method name to use as the new index.

    Returns
    -------
    dict
        Dictionary mapping parameter combination tuples to DataFrames.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex")

    # Get the level names and find the aggregation method level
    level_names = df.index.names
    if aggregation_method not in level_names:
        raise ValueError(
            f"Aggregation method '{aggregation_method}' not found in index levels: {level_names}"
        )

    agg_level_idx = level_names.index(aggregation_method)
    param_levels = [i for i, name in enumerate(level_names) if i != agg_level_idx]

    # Group by parameter combination levels
    result = {}

    for param_combo, group_df in df.groupby(level=param_levels):
        # Create a DataFrame with just the aggregation method as index
        new_df = group_df.droplevel(param_levels).copy()
        new_df.index.name = aggregation_method

        # Sort by the aggregation method values
        new_df = new_df.sort_index()

        # Ensure param_combo is a tuple
        if not isinstance(param_combo, tuple):
            param_combo = (param_combo,)

        result[param_combo] = new_df

    return result


def convert_aggregated_file(
    input_file_path: str,
    output_base_dir: str,
    ps_output_base_dir: str = None,
    verbose: bool = True,
) -> None:
    """
    Convert a single aggregated data file to the new format.

    Parameters
    ----------
    input_file_path : str
        Path to the input aggregated data file.
    output_base_dir : str
        Base directory for the new format output (aggregated files).
    ps_output_base_dir : str, optional
        Base directory for post-selection files. If None, uses output_base_dir.
    verbose : bool
        Whether to print progress information.
    """
    if verbose:
        print(f"Processing: {input_file_path}")

    # Extract info from filename
    filename = os.path.basename(input_file_path)

    # Load the data
    data_dict = load_existing_aggregated_file(input_file_path)

    if verbose:
        print(f"  Found {len(data_dict)} aggregation methods")

    # Determine if this is a post-selection file
    is_ps_file = "ps_dict" in filename

    # Choose the appropriate output directory
    if is_ps_file and ps_output_base_dir is not None:
        current_output_base_dir = ps_output_base_dir
    else:
        current_output_base_dir = output_base_dir

    # Process each aggregation method
    total_successful = 0
    total_failed = 0

    for method_key, method_data in data_dict.items():
        if verbose:
            print(f"\n  Processing method: {method_key}")

        try:
            # Extract DataFrame - handle both tuple (df_agg) and direct DataFrame (df_ps)
            if isinstance(method_data, tuple) and len(method_data) >= 1:
                df_agg = method_data[0]  # First element is the DataFrame (df_agg files)
            elif isinstance(method_data, pd.DataFrame):
                df_agg = method_data  # Direct DataFrame (df_ps files)
            else:
                if verbose:
                    print(
                        f"    Skipping {method_key}: unexpected data type {type(method_data)}"
                    )
                continue

            if df_agg.empty:
                if verbose:
                    print(f"    Skipping {method_key}: empty DataFrame")
                continue

            # Determine dataset and clean method name
            dataset, clean_method = determine_dataset_and_method_from_key(
                method_key, filename
            )

            # Determine the base aggregation method for index level matching
            aggregation_method = determine_aggregation_method_from_key(method_key)

            if verbose:
                print(f"    Dataset: {dataset}")
                print(f"    Clean method: {clean_method}")
                print(f"    Base aggregation method: {aggregation_method}")
                print(f"    DataFrame shape: {df_agg.shape}")
                print(f"    Index levels: {df_agg.index.names}")
                print(
                    f"    File type: {'post-selection' if is_ps_file else 'aggregated'}"
                )
                print(f"    Output base dir: {current_output_base_dir}")

            # Split by parameter combinations
            param_combinations = split_dataframe_by_parameter_combinations(
                df_agg, aggregation_method
            )

            if verbose:
                print(f"    Found {len(param_combinations)} parameter combinations")

            # Create output directory structure (no 'by_' prefix)
            output_dataset_dir = os.path.join(current_output_base_dir, dataset)
            output_method_dir = os.path.join(output_dataset_dir, clean_method)

            os.makedirs(output_method_dir, exist_ok=True)

            # Save each parameter combination
            method_successful = 0
            method_failed = 0

            for param_combo, param_df in param_combinations.items():
                try:
                    # Create filename in raw data style
                    param_filename = create_raw_data_style_filename(
                        param_combo, dataset
                    )
                    output_file_path = os.path.join(
                        output_method_dir, f"{param_filename}.pkl"
                    )

                    # Save the DataFrame
                    with open(output_file_path, "wb") as f:
                        pickle.dump(param_df, f)

                    method_successful += 1

                except Exception as e:
                    method_failed += 1
                    if verbose:
                        print(f"      Error saving {param_combo}: {e}")

            total_successful += method_successful
            total_failed += method_failed

            if verbose:
                print(
                    f"    Method {method_key}: {method_successful} successful, {method_failed} failed"
                )
                print(f"    Output directory: {output_method_dir}")

        except Exception as e:
            total_failed += 1
            if verbose:
                print(f"    Error processing method {method_key}: {e}")

    if verbose:
        print(
            f"\n  Total conversion: {total_successful} successful, {total_failed} failed"
        )


def convert_all_aggregated_data(
    input_dir: str = "simulations/data/aggregated",
    output_base_dir: str = "simulations/data/aggregated",
    ps_output_base_dir: str = "simulations/data/post-selection",
    verbose: bool = True,
) -> None:
    """
    Convert all aggregated data files in the input directory.

    Parameters
    ----------
    input_dir : str
        Directory containing the existing aggregated data files.
    output_base_dir : str
        Base directory for the new format output (aggregated files).
    ps_output_base_dir : str
        Base directory for post-selection files.
    verbose : bool
        Whether to print progress information.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all pkl files in the input directory (both df_agg_dict and df_ps_dict)
    pkl_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl") and (
            "df_agg_dict" in filename or "df_ps_dict" in filename
        ):
            pkl_files.append(os.path.join(input_dir, filename))

    if not pkl_files:
        if verbose:
            print(f"No aggregated data files found in {input_dir}")
        return

    if verbose:
        print(f"Found {len(pkl_files)} aggregated data files to convert")
        print(f"Aggregated files output: {output_base_dir}")
        print(f"Post-selection files output: {ps_output_base_dir}")

    # Create output base directories
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(ps_output_base_dir, exist_ok=True)

    # Convert each file
    for pkl_file in pkl_files:
        try:
            convert_aggregated_file(
                pkl_file, output_base_dir, ps_output_base_dir, verbose
            )
            if verbose:
                print("\n" + "=" * 80 + "\n")  # Separator between files
        except Exception as e:
            if verbose:
                print(f"Error processing {pkl_file}: {e}")
                print("\n" + "=" * 80 + "\n")


def main():
    """Main function to run the conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert aggregated data files to new format"
    )
    parser.add_argument(
        "--input-dir",
        default="../data/aggregated",
        help="Input directory containing existing aggregated data files",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/aggregated",
        help="Output directory for converted data files",
    )
    parser.add_argument(
        "--ps-output-dir",
        default="../data/post-selection",
        help="Output directory for post-selection files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print progress information"
    )

    args = parser.parse_args()

    print("Converting aggregated data files...")
    print(f"Input directory: {args.input_dir}")
    print(f"Aggregated files output: {args.output_dir}")
    print(f"Post-selection files output: {args.ps_output_dir}")
    print()

    convert_all_aggregated_data(
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        ps_output_base_dir=args.ps_output_dir,
        verbose=args.verbose,
    )

    print("Conversion complete!")


if __name__ == "__main__":
    main()
