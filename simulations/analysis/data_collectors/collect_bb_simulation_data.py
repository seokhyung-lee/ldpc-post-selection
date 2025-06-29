import os
import numpy as np

from simulations.analysis.data_collection import process_dataset


if __name__ == "__main__":
    ascending_confidences = {
        "pred_llr": False,
        "detector_density": False,
        "cluster_size_norm": False,
        "cluster_llr_norm": False,
        "cluster_size_norm_gap": True,
        "cluster_llr_norm_gap": True,
    }

    orders = [0.5, 1, 2, np.inf]
    num_hist_bins = 10000

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../data/bb_minsum_iter30_lsd0")

    # Process BB code data
    print("Processing BB code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name="bb",
        dataset_type="bb",
        ascending_confidences=ascending_confidences,
        orders=orders,
        num_hist_bins=num_hist_bins,
        verbose=False,
    )

    ascending_confidences = {"gap_proxy": True}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../data/bb_minsum_iter30_lsd0_gapproxy")

    # Process BB code data
    print("Processing BB code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name="bb_gapproxy",
        dataset_type="bb",
        ascending_confidences=ascending_confidences,
        num_hist_bins=num_hist_bins,
        verbose=False,
    )

    print("\nData collection complete!")
