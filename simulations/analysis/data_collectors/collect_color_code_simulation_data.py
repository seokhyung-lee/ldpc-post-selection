import os
import numpy as np

from simulations.analysis.data_collection import process_dataset


if __name__ == "__main__":
    ascending_confidences = {
        "pred_llr": False,
        "detector_density": False,
        # "cluster_size_norm": False,
        "cluster_llr_norm": False,
        # "cluster_size_norm_gap": True,
        "cluster_llr_norm_gap": True,
    }

    # ascending_confidences_matching = {
    #     "pred_llr": False,
    #     "detector_density": False,
    #     "gap": True,
    # }

    orders = [0.5, 1, 2, np.inf]
    num_hist_bins = 10000

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../data/color_minsum_iter30_lsd0_raw")
    # matching_data_dir = os.path.join(current_dir, "../data/surface_code_matching")

    # Process regular surface code data
    print("Processing color code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name="color",
        ascending_confidences=ascending_confidences,
        orders=orders,
        num_hist_bins=num_hist_bins,
        dataset_type="color",
        verbose=False,
    )

    # # Process matching data
    # print("\nProcessing surface code matching data...")
    # process_dataset(
    #     data_dir=matching_data_dir,
    #     dataset_name="surface_matching",
    #     ascending_confidences=ascending_confidences_matching,
    #     num_hist_bins=num_hist_bins,
    #     dataset_type="surface",
    #     verbose=False,
    # )

    print("\nData collection complete!")
