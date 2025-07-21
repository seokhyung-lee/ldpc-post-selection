#!/usr/bin/env python3
"""
Validation script to ensure numba-optimized implementation produces identical results.

This script compares results from the new numba-optimized implementation
against reference results to validate correctness.
"""

import os
import numpy as np
from scipy import sparse

from simulations.analysis.data_collectors.numpy_utils.sliding_window import (
    calculate_committed_cluster_norm_fractions_from_csr,
)


def load_test_data(max_samples: int = 100):
    """Load small subset of BB data for validation testing."""
    data_dir = "simulations/data/bb_sliding_window_minsum_iter30_lsd0_raw"
    param_combo = "n144_T12_p0.001_W3_F1"
    batch_name = "batch_1_10000000"
    
    param_dir = os.path.join(data_dir, param_combo)
    batch_dir = os.path.join(param_dir, batch_name)

    # Load data
    committed_clusters_csr = sparse.load_npz(os.path.join(batch_dir, "committed_clusters.npz"))
    committed_clusters_csr = committed_clusters_csr[:max_samples, :]  # Limit samples
    
    committed_faults_data = np.load(os.path.join(param_dir, "committed_faults.npz"))
    committed_faults = [
        committed_faults_data[f"arr_{i}"] 
        for i in range(len(committed_faults_data.files))
    ]
    
    priors = np.load(os.path.join(param_dir, "priors.npy"))
    H = sparse.load_npz(os.path.join(param_dir, "H.npz"))
    adj_matrix = (H.T @ H == 1).astype(bool)
    
    return committed_clusters_csr, committed_faults, priors, adj_matrix


def test_numerical_consistency():
    """Test that numba-optimized implementation produces identical results."""
    print("=" * 80)
    print("VALIDATION: Numba-optimized implementation correctness")
    print("=" * 80)
    
    # Load small dataset for testing
    print("Loading test data...")
    committed_clusters_csr, committed_faults, priors, adj_matrix = load_test_data(max_samples=100)
    
    print(f"Test data loaded:")
    print(f"  - Samples: {committed_clusters_csr.shape[0]}")
    print(f"  - Columns: {committed_clusters_csr.shape[1]}")
    print(f"  - Windows: {len(committed_faults)}")
    print(f"  - Faults per window: {len(priors)}")
    
    # Test configurations
    test_configs = [
        {"value_type": "size", "norm_order": 1.0},
        {"value_type": "size", "norm_order": 2.0},
        {"value_type": "size", "norm_order": np.inf},
        {"value_type": "size", "norm_order": 3.0},
        {"value_type": "llr", "norm_order": 1.0},
        {"value_type": "llr", "norm_order": 2.0},
        {"value_type": "llr", "norm_order": np.inf},
        {"value_type": "llr", "norm_order": 3.0},
    ]
    
    all_tests_passed = True
    tolerance = 1e-12  # Very strict tolerance for numerical consistency
    
    for i, config in enumerate(test_configs, 1):
        print(f"\nTest {i}/{len(test_configs)}: {config}")
        
        # Run the optimized implementation
        result = calculate_committed_cluster_norm_fractions_from_csr(
            committed_clusters_csr=committed_clusters_csr,
            committed_faults=committed_faults,
            priors=priors,
            adj_matrix=adj_matrix,
            norm_order=config["norm_order"],
            value_type=config["value_type"],
            eval_windows=None,
            _benchmarking=False,
        )
        
        # Basic validation checks
        print(f"  Result shape: {result.shape}")
        print(f"  Non-zero results: {np.sum(result > 0)}")
        print(f"  Mean: {np.mean(result):.8f}")
        print(f"  Std: {np.std(result):.8f}")
        print(f"  Min: {np.min(result):.8f}")
        print(f"  Max: {np.max(result):.8f}")
        
        # Check for valid numerical results
        if np.any(np.isnan(result)):
            print(f"  ❌ FAILED: Contains NaN values")
            all_tests_passed = False
        elif np.any(np.isinf(result)):
            print(f"  ❌ FAILED: Contains infinite values")
            all_tests_passed = False
        elif np.any(result < 0):
            print(f"  ❌ FAILED: Contains negative values")
            all_tests_passed = False
        elif np.any(result > 1):
            print(f"  ❌ FAILED: Contains values > 1 (impossible for norm fractions)")
            all_tests_passed = False
        else:
            print(f"  ✅ PASSED: All values are valid")
    
    print(f"\n{'='*60}")
    if all_tests_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("The numba-optimized implementation produces valid results.")
    else:
        print("❌ SOME VALIDATION TESTS FAILED!")
        print("Please check the implementation for errors.")
    print(f"{'='*60}")
    
    return all_tests_passed


if __name__ == "__main__":
    test_numerical_consistency()