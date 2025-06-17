import os
import re
import warnings
from typing import List

import pandas as pd
from tqdm import tqdm


def check_batch_shot_consistency(base_data_dir: str) -> List[str]:
    """
    Checks if the number of shots recorded in batch directory names matches
    the actual number of shots in the 'scalars.feather' files.
    Uses tqdm to show progress.

    Parameters
    ----------
    base_data_dir : str
        The base directory containing configuration subdirectories.

    Returns
    -------
    inconsistency_messages : list of str
        A list of messages describing any inconsistencies or errors found.
        An empty list indicates full consistency for this base_data_dir.
    """
    inconsistency_messages = []
    batch_dir_pattern = re.compile(r"^batch_(\d+)_(\d+)$")

    if not os.path.isdir(base_data_dir):
        inconsistency_messages.append(
            f"Error: Base data directory '{base_data_dir}' not found."
        )
        return inconsistency_messages

    config_subdirs_found = False
    # Iterate over configuration subdirectories (e.g., n72_T6_p0.001)
    # Outer loop for configuration directories with tqdm
    config_dirnames = sorted(os.listdir(base_data_dir))
    for config_dirname in tqdm(
        config_dirnames,
        desc=f"Scanning {os.path.basename(base_data_dir)}",
        unit="config",
    ):
        config_dir_path = os.path.join(base_data_dir, config_dirname)
        if not os.path.isdir(config_dir_path):
            continue
        config_subdirs_found = True

        batches_found_in_config = False
        batch_dirnames_in_config = sorted(os.listdir(config_dir_path))
        # Inner loop for batch directories with tqdm, nested description
        for batch_dirname in tqdm(
            batch_dirnames_in_config,
            desc=f"  Processing {config_dirname}",
            unit="batch",
            leave=False,
        ):
            match = batch_dir_pattern.match(batch_dirname)
            if not match:
                potential_other_dir = os.path.join(config_dir_path, batch_dirname)
                if os.path.isdir(potential_other_dir):
                    # This warning can be noisy if there are legitimate non-batch subdirs
                    # inconsistency_messages.append(f"Info: Non-standard item '{batch_dirname}' found in config '{config_dirname}'. Skipping.")
                    pass  # Silently skip non-matching directories for now
                continue

            batches_found_in_config = True
            try:
                shots_in_dirname = int(match.group(2))
            except ValueError:
                inconsistency_messages.append(
                    f"Warning: Could not parse shots from directory name {batch_dirname} in {config_dir_path}, despite regex match. Skipping."
                )
                continue

            batch_dir_path = os.path.join(config_dir_path, batch_dirname)
            if not os.path.isdir(batch_dir_path):
                inconsistency_messages.append(
                    f"Warning: Matched item {batch_dirname} in {config_dir_path} is not a directory. Skipping."
                )
                continue

            feather_file_path = os.path.join(batch_dir_path, "scalars.feather")

            if not os.path.exists(feather_file_path):
                message = f"Inconsistent: In {batch_dir_path}, 'scalars.feather' not found. Expected {shots_in_dirname} shots based on directory name."
                inconsistency_messages.append(message)
                continue

            try:
                df = pd.read_feather(feather_file_path)
                actual_shots_in_file = len(df)

                if actual_shots_in_file != shots_in_dirname:
                    message = f"Inconsistent: In {batch_dir_path}, directory name indicates {shots_in_dirname} shots, but 'scalars.feather' contains {actual_shots_in_file} shots."
                    inconsistency_messages.append(message)
            except Exception as e:
                message = f"Error Reading: In {batch_dir_path}, could not read or process 'scalars.feather'. Error: {e}"
                inconsistency_messages.append(message)

        if not batches_found_in_config and os.path.isdir(config_dir_path):
            # Check if config_dir_path itself contains any files that are not batch directories
            # This avoids warning for genuinely empty config dirs or configs with non-batch subdirs
            if any(
                batch_dir_pattern.match(item)
                for item in batch_dirnames_in_config
                if os.path.isdir(os.path.join(config_dir_path, item))
            ):
                pass  # Contains batch-like dirs, but maybe they failed parsing earlier or were not dirs
            elif not any(
                os.path.isdir(os.path.join(config_dir_path, item))
                for item in batch_dirnames_in_config
            ):
                # Only add message if the config dir is truly empty of batch-like subdirectories
                # or contains no subdirectories at all.
                # inconsistency_messages.append(f"Info: No batch directories (e.g., 'batch_X_Y') found or matched in {config_dir_path}.")
                pass  # Keep it silent for now

    if not config_subdirs_found and os.path.isdir(base_data_dir):
        # Only report if base_data_dir itself exists but no valid config subdirs were processed
        if any(
            os.path.isdir(os.path.join(base_data_dir, item))
            for item in os.listdir(base_data_dir)
        ):
            inconsistency_messages.append(
                f"Info: No processable configuration subdirectories found in '{base_data_dir}'. Check directory structure."
            )
        else:
            inconsistency_messages.append(
                f"Info: Base directory '{base_data_dir}' is empty or contains no subdirectories."
            )

    return inconsistency_messages


if __name__ == "__main__":
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    base_sim_data_dir = os.path.join(script_parent_dir, "data")

    excluded_experiment_folders = [
        ".DS_Store",
        "__pycache__",
        "archive",
        ".ipynb_checkpoints",
    ]

    print("####################################################################")
    print("### Running Data Consistency Checker ###")
    print("####################################################################")

    if not os.path.isdir(base_sim_data_dir):
        print(f"Error: Base simulation data directory '{base_sim_data_dir}' not found.")
        print("Cannot proceed with the consistency check.")
        exit()

    overall_inconsistencies = []
    processed_folders_count = 0

    # Main loop for experiment folders with tqdm
    experiment_folders_to_check = []
    for item_name in sorted(os.listdir(base_sim_data_dir)):
        if item_name in excluded_experiment_folders:
            continue
        experiment_data_path = os.path.join(base_sim_data_dir, item_name)
        if os.path.isdir(experiment_data_path):
            experiment_folders_to_check.append(experiment_data_path)

    if not experiment_folders_to_check:
        print(
            f"No experiment data directories found to check in '{base_sim_data_dir}' (after exclusions)."
        )
    else:
        for experiment_data_path in tqdm(
            experiment_folders_to_check, desc="Checking Experiments", unit="experiment"
        ):
            # The description for the inner tqdm will now show which experiment is being scanned
            # by using os.path.basename(experiment_data_path) in the check_batch_shot_consistency function.

            # Temporarily disable outer tqdm to let inner tqdm take over for config/batch level
            # This is a bit of a workaround as nested tqdm can be tricky with `leave=False`
            # A cleaner way might involve custom tqdm positioning or managing multiple bars.
            # For simplicity, we let the inner function handle its progress bar.
            # tqdm.set_description_str(f"Processing Experiment: {os.path.basename(experiment_data_path)}")

            experiment_name = os.path.basename(experiment_data_path)
            # No need to print here, tqdm handles it
            # print(f"\n--- Checking data in: '{experiment_data_path}' ---")

            # Ensure the inner tqdm in check_batch_shot_consistency uses experiment_name in its description
            # This is handled by how `desc` is formatted in the inner function
            current_experiment_inconsistencies = check_batch_shot_consistency(
                experiment_data_path
            )
            processed_folders_count += 1
            if current_experiment_inconsistencies:
                overall_inconsistencies.append(
                    f"\n--- Issues found in experiment: {experiment_name} ({experiment_data_path}) ---"
                )
                overall_inconsistencies.extend(
                    [f"  - {msg}" for msg in current_experiment_inconsistencies]
                )

    print("\n####################################################################")
    print("### Data Consistency Check Finished ###")
    print("####################################################################")

    if not processed_folders_count:
        print(
            "No data directories were found or processed (after exclusions). Ensure 'simulations/data' contains relevant subdirectories."
        )
    elif not overall_inconsistencies:
        print(f"All {processed_folders_count} checked experiment(s) are consistent.")
    else:
        print("Found the following inconsistencies or errors:")
        for msg in overall_inconsistencies:
            print(msg)

    print("\n####################################################################")
