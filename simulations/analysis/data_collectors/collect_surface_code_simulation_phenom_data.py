import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import (
    process_dataset,
    DATA_DIR,
)


if __name__ == "__main__":
    ascending_confidences = {
        "pred_llr": False,
        "detector_density": False,
        "cluster_llr_norm_frac": False,
    }

    # orders = [0.5, 1, 2, np.inf]
    orders = [2]

    use_old_format_data = False

    data_dir_name = "surface_minsum_iter30_lsd0_biased_phenom_raw"
    dataset_name = "surface_biased_phenom"

    # Find data directories
    data_dir = str(DATA_DIR / data_dir_name)

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

    print("\nData collection complete!")
