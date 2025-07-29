#!/usr/bin/env python3
"""
Inspect existing aggregated data files to understand their structure.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def inspect_pkl_file(file_path: str) -> None:
    """
    Inspect a pickle file and print its structure.

    Parameters
    ----------
    file_path : str
        Path to the pickle file to inspect.
    """
    print(f"\n=== Inspecting: {file_path} ===")

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        print(f"Type: {type(data)}")

        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            print("\nFirst few keys:")
            for i, key in enumerate(list(data.keys())[:3]):
                print(f"  {i+1}: {key} (type: {type(key)})")

            # Inspect the first value
            first_key = next(iter(data.keys()))
            first_value = data[first_key]
            print(f"\nFirst value (key: {first_key}):")
            print(f"  Type: {type(first_value)}")

            if isinstance(first_value, pd.DataFrame):
                print(f"  Shape: {first_value.shape}")
                print(f"  Columns: {list(first_value.columns)}")
                print(f"  Index name: {first_value.index.name}")
                print(f"  Index type: {type(first_value.index)}")
                print(f"  First few rows:")
                print(first_value.head())
            elif isinstance(first_value, tuple):
                print(f"  Tuple length: {len(first_value)}")
                for i, item in enumerate(first_value):
                    print(f"    Item {i}: {type(item)}")
                    if isinstance(item, pd.DataFrame):
                        print(f"      Shape: {item.shape}")
                        print(f"      Columns: {list(item.columns)}")
                        print(f"      Index name: {item.index.name}")
            else:
                print(f"  Value: {first_value}")

    except Exception as e:
        print(f"Error loading file: {e}")


def main():
    """Main function to inspect all aggregated data files."""
    input_dir = "../data/aggregated"

    if not os.path.isdir(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    pkl_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            pkl_files.append(os.path.join(input_dir, filename))

    print(f"Found {len(pkl_files)} pickle files to inspect")

    for pkl_file in pkl_files:
        inspect_pkl_file(pkl_file)


if __name__ == "__main__":
    main()
