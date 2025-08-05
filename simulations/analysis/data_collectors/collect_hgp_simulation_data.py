import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import process_dataset


if __name__ == "__main__":
    ascending_confidences = {
        "pred_llr": False,
        "detector_density": False,
        "cluster_size_norm_frac": False,
        "cluster_llr_norm_frac": False,
    }

    orders = [0.5, 1, 2, np.inf]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../../data/hgp_minsum_iter30_lsd0_raw")

    # Process HGP code data
    print("Processing HGP code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name="hgp",
        dataset_type="hgp",
        ascending_confidences=ascending_confidences,
        orders=orders,
        decimals=(lambda by: 2 if by == "pred_llr" else 4),
        verbose=False,
    )

    print("\nData collection complete!")
