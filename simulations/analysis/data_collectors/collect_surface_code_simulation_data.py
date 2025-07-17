import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import (
    process_dataset,
    DATA_DIR,
)


if __name__ == "__main__":
    ascending_confidences = {
        # "pred_llr": False,
        # "detector_density": False,
        "cluster_llr_norm_frac": False,
        # "cluster_size_norm_frac": False,
    }

    ascending_confidences_matching = {
        "gap": True,
    }

    # orders = [0.5, 1, 2, np.inf]
    orders = [2]

    use_old_format_data = False

    if use_old_format_data:
        data_dir_name = "surface_minsum_iter30_lsd0"
        dataset_name = "surface_old"
    else:
        data_dir_name = "surface_minsum_iter30_lsd0_raw"
        dataset_name = "surface"

    matching_data_dir_name = "surface_matching"
    matching_dataset_name = "surface_matching"

    # Find data directories
    data_dir = str(DATA_DIR / data_dir_name)
    matching_data_dir = str(DATA_DIR / matching_data_dir_name)

    # Process regular surface code data
    print("Processing surface code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        ascending_confidences=ascending_confidences,
        orders=orders,
        dataset_type="surface",
        decimals=(lambda by: 2 if by == "pred_llr" else 4),
        verbose=False,
    )

    # Process matching data
    print("\nProcessing surface code matching data...")
    process_dataset(
        data_dir=matching_data_dir,
        dataset_name=matching_dataset_name,
        ascending_confidences=ascending_confidences_matching,
        dataset_type="surface",
        decimals=2,
        verbose=False,
    )

    print("\nData collection complete!")
