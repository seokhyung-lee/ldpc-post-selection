import glob
import multiprocessing
import os
from pathlib import Path

import pandas as pd


def process_file(file_path: str) -> None:
    """
    Reads a feather file, negates specified columns, and overwrites the file.

    Parameters
    ----------
    file_path : str
        The path to the feather file.

    Returns
    -------
    None
    """
    try:
        df = pd.read_feather(file_path)

        cols_to_negate = ["pred_llr", "cluster_llr_sum", "outside_cluster_llr"]

        for col in cols_to_negate:
            if col in df.columns:
                df[col] = -df[col]
            else:
                print(f"Warning: Column '{col}' not found in {file_path}")

        df.to_feather(file_path)
        # print(f"Processed and overwritten: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


COLS_TO_NEGATE = ["pred_llr", "cluster_llr_sum", "outside_cluster_llr"]


def main() -> None:
    """
    Finds all feather files matching the specified pattern.
    Processes the first file, allows for verification, and then processes the rest upon confirmation.
    """
    base_data_path = Path("simulations/data/bb_circuit_iter30_minsum_lsd0")

    if not base_data_path.exists() or not base_data_path.is_dir():
        print(f"Error: Base data directory not found: {base_data_path}")
        return

    file_pattern = str(base_data_path / "n*_T*_p*" / "data_*.feather")
    file_paths = sorted(
        glob.glob(file_pattern, recursive=True)
    )  # Sort for consistent first file

    if not file_paths:
        print(f"No files found matching the pattern: {file_pattern}")
        return

    print(f"Found {len(file_paths)} files to process.")

    # --- Process the first file for verification ---
    first_file_path = file_paths[0]
    print(f"\n--- Verification for: {first_file_path} ---")

    # --- Show original values before processing ---
    print("\nOriginal values (before processing):")
    try:
        df_original = pd.read_feather(first_file_path)
        cols_to_display_original = [
            col for col in COLS_TO_NEGATE if col in df_original.columns
        ]
        if cols_to_display_original:
            print(df_original[cols_to_display_original].head())
        else:
            print(
                "None of the target columns for negation were found in the original file."
            )
            # If target columns don't exist, it's important to note before proceeding.
            if not any(col in df_original.columns for col in COLS_TO_NEGATE):
                print(
                    "Warning: None of the specified columns for negation exist in this file."
                )
    except Exception as e:
        print(
            f"Error reading original file {first_file_path} for pre-verification: {e}"
        )
        print("Aborting further processing.")
        return

    print(f"\nProcessing the first file: {first_file_path}")
    process_file(first_file_path)
    print(f"Successfully processed: {first_file_path}")

    # --- Verification step (after processing) ---
    print("\nValues after processing:")
    try:
        df_check = pd.read_feather(first_file_path)
        # print(f"Head of negated columns in {first_file_path}:") # Redundant with new title

        cols_to_display_processed = [
            col for col in COLS_TO_NEGATE if col in df_check.columns
        ]
        if cols_to_display_processed:
            print(df_check[cols_to_display_processed].head())
        else:
            print(
                "None of the target columns for negation were found in the processed file."
            )

        # This check is still relevant for the processed file
        if not any(col in df_check.columns for col in COLS_TO_NEGATE):
            print(
                "Warning: None of the specified columns for negation exist in the processed file."
            )
            print("Please check the file structure and column names.")

    except Exception as e:
        print(f"Error reading file {first_file_path} for verification: {e}")
        print("Aborting further processing due to verification error.")
        return

    # --- Ask for confirmation to process remaining files ---
    remaining_files = file_paths[1:]
    if not remaining_files:
        print("\nNo remaining files to process after the first one.")
        print("Script finished.")
        return

    while True:
        proceed = (
            input(
                f"\nDo you want to proceed with processing the remaining {len(remaining_files)} files? (yes/no): "
            )
            .strip()
            .lower()
        )
        if proceed in ["yes", "y"]:
            break
        elif proceed in ["no", "n"]:
            print("Aborted processing remaining files by user.")
            return
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    # --- Process remaining files using multiprocessing ---
    print(f"\nProcessing {len(remaining_files)} remaining files...")
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_file, remaining_files)

    print("\nAll remaining files have been processed.")


if __name__ == "__main__":
    main()
