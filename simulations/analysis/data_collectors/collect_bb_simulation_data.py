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
        "cluster_size_norm_frac": False,
        "cluster_llr_norm_frac": False,
    }

    # orders = [0.5, 1, 2, np.inf]
    orders = [2]

    data_dir_name = "bb_minsum_iter30_lsd0_raw"
    dataset_name = "bb"

    data_dir = str(DATA_DIR / data_dir_name)

    # Process BB code data
    print("Processing BB code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        dataset_type="bb",
        ascending_confidences=ascending_confidences,
        orders=orders,
        decimals=(lambda by: 4 if by == 'detector_density' else 2),
        verbose=False,
    )

    # ascending_confidences = {"gap_proxy": True}

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(current_dir, "../data/bb_minsum_iter30_lsd0_gapproxy")

    # # Process BB code data
    # print("Processing BB code data...")
    # process_dataset(
    #     data_dir=data_dir,
    #     dataset_name="bb_gapproxy",
    #     dataset_type="bb",
    #     ascending_confidences=ascending_confidences,
    #     num_hist_bins=num_hist_bins,
    #     verbose=False,
    # )

    print("\nData collection complete!")
