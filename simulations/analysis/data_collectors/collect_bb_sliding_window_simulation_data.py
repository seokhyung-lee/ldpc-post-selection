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
    aggregation_types = ["mean", "max", "committed"]
    value_types = ["size", "llr"]
    
    # For cluster metrics, higher values typically mean lower confidence
    for agg_type in aggregation_types:
        for value_type in value_types:
            metric_base = f"{agg_type}_cluster_{value_type}_norm_frac"
            ascending_confidences[metric_base] = False

    # Define norm orders to calculate
    orders = [2]  # Can be extended to [0.5, 1, 2, np.inf]

    # Data directory configuration
    data_dir_name = "bb_sliding_window_minsum_iter30_lsd0"
    dataset_name = "bb_sliding_window"

    data_dir = str(DATA_DIR / data_dir_name)

    # Process sliding window BB code data
    print("Processing sliding window BB code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        dataset_type="bb",  # Use same type as regular BB codes
        ascending_confidences=ascending_confidences,
        orders=orders,
        verbose=False,
    )

    print("\nSliding window data collection complete!")