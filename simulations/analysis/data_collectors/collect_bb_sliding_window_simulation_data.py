import os
import numpy as np

from simulations.analysis.data_collectors.data_collection import (
    process_dataset,
    DATA_DIR,
)
from simulations.analysis.data_collectors.sliding_window_post_selection import (
    analyze_parameter_combination,
)
from simulations.analysis.data_collectors.numpy_utils.sliding_window import (
    batch_postselection_analysis,
    create_cutoff_arrays,
    optimize_postselection_parameters,
)


if __name__ == "__main__":
    # Define sliding window specific metrics
    # Format: {aggregation_type}_cluster_{size/llr}_norm_frac_{order}
    ascending_confidences = {}

    # Define aggregation types and their confidence directions
    aggregation_types = ["avg_window", "max_window", "committed"]
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
    data_dir_name = "bb_sliding_window_minsum_iter30_lsd0_raw"
    dataset_name = "bb_sliding_window"

    data_dir = str(DATA_DIR / data_dir_name)

    subdirs = ["n144_T12_p0.003_W3_F1", "n144_T12_p0.005_W3_F1"]

    # Process sliding window BB code data
    print("Processing sliding window BB code data...")
    process_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        dataset_type="bb",
        ascending_confidences=ascending_confidences,
        orders=orders,
        decimals=4,
        subdirs=subdirs,
        verbose=False,
        # num_jobs=18,
        # num_batches=18 * 10,
    )

    print("\nSliding window data collection complete!")
    
    # =============================================================================
    # Real-Time Post-Selection Analysis
    # =============================================================================
    
    print("\n" + "="*60)
    print("REAL-TIME POST-SELECTION ANALYSIS")
    print("="*60)
    
    # Configuration for post-selection analysis
    postselection_config = {
        'metric_windows': [1, 2, 3],  # Different window sizes to test
        'norm_orders': [2.0],         # L2 norm (can extend to [1.0, 2.0, np.inf])
        'value_types': ['llr'],       # Focus on LLR-based analysis
        'num_jobs': 8,                # Parallel processing
        'verbose': True
    }
    
    # Create comprehensive cutoff arrays for analysis
    cutoff_arrays = create_cutoff_arrays(
        fine_range=(0.002, 0.08),    # Fine-grained range around typical values
        fine_points=100,
        coarse_range=(1e-4, 0.2),    # Broader coarse range
        coarse_points=30,
        log_scale=True
    )
    
    print(f"Created cutoff arrays:")
    for name, cutoffs in cutoff_arrays.items():
        print(f"  {name}: {len(cutoffs)} points, range [{cutoffs[0]:.6f}, {cutoffs[-1]:.6f}]")
    
    # Process each cutoff array configuration
    postselection_results = {}
    
    for cutoff_name, cutoffs in cutoff_arrays.items():
        print(f"\n--- Processing cutoff array: {cutoff_name} ---")
        
        for metric_windows in postselection_config['metric_windows']:
            for norm_order in postselection_config['norm_orders']:
                for value_type in postselection_config['value_types']:
                    
                    config_name = f"{cutoff_name}_mw{metric_windows}_ord{norm_order}_{value_type}"
                    print(f"Configuration: {config_name}")
                    
                    try:
                        # Run batch post-selection analysis
                        batch_results = batch_postselection_analysis(
                            data_dir=data_dir,
                            param_combinations=subdirs,
                            cutoffs=cutoffs,
                            metric_windows=metric_windows,
                            norm_order=norm_order,
                            value_type=value_type,
                            num_jobs=postselection_config['num_jobs'],
                            verbose=postselection_config['verbose']
                        )
                        
                        # Store results
                        postselection_results[config_name] = batch_results
                        
                        # Analyze and print key insights for each parameter combination
                        print(f"\nOptimal parameters for {config_name}:")
                        for param_combo, results in batch_results.items():
                            if not results:  # Skip failed analyses
                                continue
                                
                            # Find optimal cutoff values for different targets
                            optimal_params = optimize_postselection_parameters(
                                results,
                                target_abort_rate=0.1,      # 10% abort rate target
                                target_effective_trials=1.5, # 1.5x effective trials target
                                optimization_metric="p_fail"
                            )
                            
                            print(f"  {param_combo}:")
                            print(f"    Optimal cutoff: {optimal_params['optimal_cutoff']:.6f}")
                            print(f"    LER: {optimal_params['achieved_p_fail']:.2e}")
                            print(f"    Abort rate: {optimal_params['achieved_p_abort']:.3f}")
                            print(f"    Effective trials: {optimal_params['achieved_effective_trials']:.3f}")
                            print(f"    Meets constraints: {optimal_params['meets_constraints']}")
                        
                    except Exception as e:
                        print(f"Error processing {config_name}: {e}")
                        postselection_results[config_name] = {}
    
    # Save comprehensive post-selection results
    results_dir = DATA_DIR / "post_selection_analysis"
    results_dir.mkdir(exist_ok=True)
    
    results_filename = f"bb_sliding_window_postselection_results_{dataset_name}.pkl"
    results_path = results_dir / results_filename
    
    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump(postselection_results, f)
    
    print(f"\nPost-selection results saved to: {results_path}")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("POST-SELECTION ANALYSIS SUMMARY")
    print("="*60)
    
    for config_name, batch_results in postselection_results.items():
        if not batch_results:
            continue
            
        print(f"\n{config_name}:")
        
        for param_combo, results in batch_results.items():
            if not results:
                continue
                
            cutoffs = results['cutoffs']
            p_abort = results['p_abort']
            effective_avg_trials = results['effective_avg_trials']
            
            # Find interesting operating points
            low_abort_idx = np.argmax(p_abort <= 0.05)  # ~5% abort rate
            med_abort_idx = np.argmax(p_abort <= 0.1)   # ~10% abort rate
            high_abort_idx = np.argmax(p_abort <= 0.2)  # ~20% abort rate
            
            print(f"  {param_combo}: Operating points")
            for desc, idx in [("5% abort", low_abort_idx), ("10% abort", med_abort_idx), ("20% abort", high_abort_idx)]:
                if idx > 0 and idx < len(cutoffs):
                    print(f"    {desc}: cutoff={cutoffs[idx]:.4f}, "
                          f"LER={results['p_fail'][idx]:.2e}, "
                          f"trials={effective_avg_trials[idx]:.2f}")
    
    print(f"\nReal-time post-selection analysis complete!")
    print(f"Results available in: {results_path}")
    print(f"Processed {len([r for r in postselection_results.values() if r])} successful configurations")
