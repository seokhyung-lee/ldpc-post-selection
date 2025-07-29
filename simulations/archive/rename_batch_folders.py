import argparse
import os
import re
import warnings

import pandas as pd


def rename_batch_folders_in_config_dir(config_dir_path: str) -> None:
    """
    Renames batch folders within a single configuration directory.

    Old format: batch_{idx}
    New format: batch_{idx}_{num_shots}
    Number of shots is determined by reading 'scalars.feather' in the old batch folder.

    Parameters
    ----------
    config_dir_path : str
        Path to the configuration directory (e.g., "data/base_dir/n72_T6_p0.002").
    """
    if not os.path.isdir(config_dir_path):
        print(f"Configuration directory not found: {config_dir_path}")
        return

    print(f"Processing configuration directory: {config_dir_path}")

    # Regex to match old batch folder names like "batch_1", "batch_12"
    old_batch_pattern = re.compile(r"^batch_(\d+)$")

    for item_name in os.listdir(config_dir_path):
        item_path = os.path.join(config_dir_path, item_name)

        if os.path.isdir(item_path):
            match = old_batch_pattern.match(item_name)
            if match:
                batch_idx_str = match.group(1)
                try:
                    batch_idx = int(batch_idx_str)
                except ValueError:
                    warnings.warn(
                        f"Could not parse batch index from folder name {item_name} in {config_dir_path}. Skipping."
                    )
                    continue

                old_batch_dir_path = item_path
                feather_file_path = os.path.join(old_batch_dir_path, "scalars.feather")

                if os.path.exists(feather_file_path):
                    try:
                        df = pd.read_feather(feather_file_path)
                        num_shots = len(df)
                        del df  # Free memory

                        new_batch_dir_name = f"batch_{batch_idx}_{num_shots}"
                        new_batch_dir_path = os.path.join(
                            config_dir_path, new_batch_dir_name
                        )

                        if old_batch_dir_path == new_batch_dir_path:
                            print(
                                f"  Skipping (already named correctly or no change needed): {old_batch_dir_path}"
                            )
                            continue

                        if os.path.exists(new_batch_dir_path):
                            warnings.warn(
                                f"  Target directory {new_batch_dir_path} already exists. Skipping rename of {old_batch_dir_path}."
                            )
                            continue

                        os.rename(old_batch_dir_path, new_batch_dir_path)
                        print(
                            f"  Renamed: {old_batch_dir_path} -> {new_batch_dir_path}"
                        )

                    except Exception as e:
                        warnings.warn(
                            f"  Could not read or process {feather_file_path}: {e}. Skipping rename for {old_batch_dir_path}."
                        )
                else:
                    warnings.warn(
                        f"  'scalars.feather' not found in {old_batch_dir_path}. Skipping rename."
                    )
            # else:
            #     It might be an already renamed folder or another type of directory, so we ignore it.
            #     print(f"  Skipping (does not match old batch pattern): {item_path}")


def main():
    """
    Main function to parse arguments and initiate the renaming process.
    """
    parser = argparse.ArgumentParser(
        description="Rename batch folders from 'batch_{idx}' to 'batch_{idx}_{num_shots}'. "
        "The number of shots is determined by reading 'scalars.feather' in each old batch folder."
    )
    parser.add_argument(
        "data_root_dir",
        type=str,
        help="The root directory containing configuration subdirectories (e.g., 'simulations/data/my_experiment_data').",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_root_dir):
        print(f"Error: Root data directory not found: {args.data_root_dir}")
        return

    print(
        f"Starting batch folder renaming process for root directory: {args.data_root_dir}"
    )

    for config_dirname in os.listdir(args.data_root_dir):
        config_dir_path = os.path.join(args.data_root_dir, config_dirname)
        if os.path.isdir(config_dir_path):
            rename_batch_folders_in_config_dir(config_dir_path)
        else:
            print(f"Skipping (not a directory): {config_dir_path}")

    print("\\nFolder renaming process completed.")


if __name__ == "__main__":
    main()
