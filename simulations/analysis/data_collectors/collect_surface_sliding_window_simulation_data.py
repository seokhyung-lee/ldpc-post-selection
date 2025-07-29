import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import (
    process_dataset,
    DATA_DIR,
)


if __name__ == "__main__":
    # Define sliding window specific metrics
    # Format: {aggregation_type}_cluster_{size/llr}_norm_frac_{order}
    ascending_confidences = {}

    # Define aggregation types and their confidence directions
    aggregation_types = ["committed"]
    # aggregation_types = ["committed"]
    # value_types = ["size", "llr"]
    value_types = ["llr"]

    # For cluster metrics, higher values typically mean lower confidence
    for agg_type in aggregation_types:
        for value_type in value_types:
            metric_base = f"{agg_type}_cluster_{value_type}_norm_frac"
            ascending_confidences[metric_base] = False

    # Define norm orders to calculate
    orders = [2]  # Can be extended to [0.5, 1, 2, np.inf]

    # Data directory configuration
    data_dir_name = "surface_sliding_window_minsum_iter30_lsd0_raw"
    dataset_name = "surface_sliding_window"

    data_dir = str(DATA_DIR / data_dir_name)

    subdirs = ["d13_T13_p0.005_W3_F1"]

    # Process sliding window BB code data
    print("Processing sliding window BB code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        dataset_type="surface",
        ascending_confidences=ascending_confidences,
        orders=orders,
        decimals=4,
        subdirs=subdirs,
        verbose=False,
        # num_jobs=18,
        # num_batches=18 * 10,
    )

    print("\nSliding window data collection complete!")
