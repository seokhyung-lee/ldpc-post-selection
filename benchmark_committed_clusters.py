#!/usr/bin/env python3
"""
Benchmarking script for calculate_committed_cluster_norm_fractions_from_csr function.

This script loads BB sliding window data and runs the function with _benchmarking=True
to identify performance bottlenecks.
"""

import os
import numpy as np
from scipy import sparse

from simulations.analysis.data_collectors.numpy_utils.sliding_window import (
    calculate_committed_cluster_norm_fractions_from_csr
)


def load_bb_data(data_dir: str, param_combo: str, batch_name: str, max_samples: int = 10000):
    """
    Load BB sliding window data for benchmarking.
    
    Parameters
    ----------
    data_dir : str
        Base data directory path.
    param_combo : str
        Parameter combination directory name (e.g., 'n144_T12_p0.001_W3_F1').
    batch_name : str
        Batch directory name (e.g., 'batch_1_10000000').
    max_samples : int, optional
        Maximum number of samples to load for benchmarking. Defaults to 10000.
        
    Returns
    -------
    committed_clusters_csr : scipy.sparse.csr_matrix
        CSR matrix of committed clusters.
    committed_faults : list of np.ndarray
        List of committed faults for each window.
    priors : np.ndarray
        Prior probabilities for each fault.
    adj_matrix : np.ndarray
        Adjacency matrix computed from H matrix.
    """
    param_dir = os.path.join(data_dir, param_combo)
    batch_dir = os.path.join(param_dir, batch_name)
    
    print(f"Loading data from: {param_dir}")
    print(f"Batch directory: {batch_dir}")
    
    # Load committed_clusters.npz from batch directory
    committed_clusters_path = os.path.join(batch_dir, "committed_clusters.npz")
    print(f"Loading committed clusters from: {committed_clusters_path}")
    committed_clusters_csr = sparse.load_npz(committed_clusters_path)
    
    # Limit to max_samples if specified
    if max_samples < committed_clusters_csr.shape[0]:
        print(f"Limiting to first {max_samples} samples (out of {committed_clusters_csr.shape[0]})")
        committed_clusters_csr = committed_clusters_csr[:max_samples, :]
    
    # Load committed_faults.npz from parameter directory
    committed_faults_path = os.path.join(param_dir, "committed_faults.npz")
    print(f"Loading committed faults from: {committed_faults_path}")
    committed_faults_data = np.load(committed_faults_path)
    committed_faults = [committed_faults_data[f'arr_{i}'] for i in range(len(committed_faults_data.files))]
    
    # Load priors.npy from parameter directory
    priors_path = os.path.join(param_dir, "priors.npy")
    print(f"Loading priors from: {priors_path}")
    priors = np.load(priors_path)
    
    # Load H.npz and compute adjacency matrix
    H_path = os.path.join(param_dir, "H.npz")
    print(f"Loading H matrix from: {H_path}")
    H = sparse.load_npz(H_path)
    print(f"Computing adjacency matrix from H matrix (shape: {H.shape})")
    adj_matrix = (H.T @ H == 1).astype(bool)
    
    print(f"Data loaded successfully:")
    print(f"  - Committed clusters shape: {committed_clusters_csr.shape}")
    print(f"  - Number of windows: {len(committed_faults)}")
    print(f"  - Number of faults per window: {len(priors)}")
    print(f"  - Adjacency matrix shape: {adj_matrix.shape}")
    
    return committed_clusters_csr, committed_faults, priors, adj_matrix


def run_benchmark():
    """
    Run the benchmarking test on BB sliding window data.
    """
    print("=" * 80)
    print("BENCHMARKING: calculate_committed_cluster_norm_fractions_from_csr")
    print("=" * 80)
    
    # Configuration
    data_dir = "simulations/data/bb_sliding_window_minsum_iter30_lsd0_raw"
    param_combo = "n144_T12_p0.001_W3_F1"  # Use first available parameter combination
    batch_name = "batch_1_10000000"  # Use first batch
    max_samples = 10000  # Limit to 10,000 samples for faster benchmarking
    
    try:
        # Load data
        committed_clusters_csr, committed_faults, priors, adj_matrix = load_bb_data(
            data_dir, param_combo, batch_name, max_samples
        )
        
        # Test configurations
        test_configs = [
            {"value_type": "size", "norm_order": 2.0},
            {"value_type": "llr", "norm_order": 2.0},
        ]
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}/{len(test_configs)}: {config}")
            print(f"{'='*60}")
            
            # Run the function with benchmarking enabled
            result = calculate_committed_cluster_norm_fractions_from_csr(
                committed_clusters_csr=committed_clusters_csr,
                committed_faults=committed_faults,
                priors=priors,
                adj_matrix=adj_matrix,
                norm_order=config["norm_order"],
                value_type=config["value_type"],
                eval_windows=None,  # Use all windows
                _benchmarking=True  # Enable benchmarking
            )
            
            print(f"Result summary:")
            print(f"  - Output shape: {result.shape}")
            print(f"  - Non-zero results: {np.sum(result > 0)}")
            print(f"  - Mean value: {np.mean(result):.6f}")
            print(f"  - Max value: {np.max(result):.6f}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files.")
        print(f"Please ensure the BB sliding window data is available at: {data_dir}")
        print(f"Specific error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_benchmark()