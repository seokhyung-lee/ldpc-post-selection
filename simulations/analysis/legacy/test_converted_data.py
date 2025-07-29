#!/usr/bin/env python3
"""
Test script to verify converted aggregated data files.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def test_converted_file(file_path: str) -> None:
    """
    Test a converted aggregated data file.

    Parameters
    ----------
    file_path : str
        Path to the converted pickle file.
    """
    print(f"Testing: {file_path}")

    try:
        with open(file_path, "rb") as f:
            df = pickle.load(f)

        print(f"  Type: {type(df)}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Index name: {df.index.name}")
        print(f"  Index type: {type(df.index)}")
        print(f"  Index range: {df.index.min()} to {df.index.max()}")
        print(f"  First few rows:")
        print(df.head())
        print()

    except Exception as e:
        print(f"  Error: {e}")
        print()


def main():
    """Test various converted files from different datasets and methods."""
    test_cases = [
        ("../data/aggregated/bb/pred_llr", "bb pred_llr"),
        ("../data/aggregated/surface/detector_density", "surface detector_density"),
        ("../data/aggregated/surface_matching/gap", "surface_matching gap"),
    ]

    for test_dir, description in test_cases:
        print(f"=== Testing {description} ===")

        if not os.path.isdir(test_dir):
            print(f"Test directory not found: {test_dir}")
            continue

        # Test first file in the directory
        files = os.listdir(test_dir)
        pkl_files = [f for f in files if f.endswith(".pkl")]

        if pkl_files:
            print(f"Found {len(pkl_files)} files in {test_dir}")
            file_path = os.path.join(test_dir, pkl_files[0])
            test_converted_file(file_path)
        else:
            print(f"No pkl files found in {test_dir}")
            print()


if __name__ == "__main__":
    main()
