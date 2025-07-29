#!/usr/bin/env python3
"""
Test script to verify that legacy and optimized histogram calculation functions
produce identical results across various scenarios.
"""

import numpy as np
import time
from typing import Tuple

# Import legacy versions from separate file
from simulations.analysis.legacy.numba_functions_legacy import (
    _calculate_histograms_bplsd_legacy,
    _calculate_histograms_matching_legacy,
)

# Import new optimized versions
from simulations.analysis.data_collectors.utils import (
    _calculate_histograms_bplsd_numba,
    _calculate_histograms_matching_numba,
    _is_uniform_binning,
)


def generate_test_data(
    n_samples: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test data for histogram calculations.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    values : 1D numpy array of float
        Random values between 0 and 1.
    fail_mask : 1D numpy array of bool
        Random boolean mask for failures.
    converge_mask : 1D numpy array of bool
        Random boolean mask for convergence.
    fail_bp_mask : 1D numpy array of bool
        Random boolean mask for BP failures.
    """
    np.random.seed(seed)

    # Generate values with some edge cases
    values = np.random.random(n_samples)

    # Add some edge case values
    if n_samples >= 10:
        values[0] = 0.0  # minimum value
        values[1] = 1.0  # maximum value
        values[2] = 0.5  # middle value
        if n_samples >= 20:
            values[10:15] = np.linspace(0.0, 1.0, 5)  # evenly spaced values

    # Generate random masks
    fail_mask = np.random.random(n_samples) < 0.3
    converge_mask = np.random.random(n_samples) < 0.7
    fail_bp_mask = np.random.random(n_samples) < 0.2

    return values, fail_mask, converge_mask, fail_bp_mask


def test_uniform_binning_detection():
    """Test the uniform binning detection function."""
    print("Testing uniform binning detection...")

    # Test uniform binning
    uniform_bins = np.linspace(0, 1, 21)  # 20 uniform bins
    assert _is_uniform_binning(uniform_bins), "Failed to detect uniform binning"

    # Test non-uniform binning
    non_uniform_bins = np.array([0, 0.1, 0.3, 0.7, 1.0])
    assert not _is_uniform_binning(
        non_uniform_bins
    ), "Incorrectly detected uniform binning"

    # Test edge cases
    assert _is_uniform_binning(np.array([0, 1])), "Failed on 2-element array"
    assert _is_uniform_binning(np.array([5])), "Failed on 1-element array"

    print("✓ Uniform binning detection tests passed")


def test_bplsd_functions(n_samples: int, bin_edges: np.ndarray, test_name: str):
    """
    Test that legacy and optimized BPLSD functions produce identical results.

    Parameters
    ----------
    n_samples : int
        Number of samples to test.
    bin_edges : 1D numpy array
        Bin edges for histogram.
    test_name : str
        Name of the test for reporting.
    """
    print(
        f"Testing BPLSD functions: {test_name} (n={n_samples}, bins={len(bin_edges)-1})"
    )

    # Generate test data
    values, fail_mask, converge_mask, fail_bp_mask = generate_test_data(n_samples)

    # Remove any NaN values (as required by the functions)
    valid_mask = ~np.isnan(values)
    values = values[valid_mask]
    fail_mask = fail_mask[valid_mask]
    converge_mask = converge_mask[valid_mask]
    fail_bp_mask = fail_bp_mask[valid_mask]

    n_bins = len(bin_edges) - 1

    # Initialize histograms for legacy version
    total_hist_legacy = np.zeros(n_bins, dtype=np.int64)
    fail_hist_legacy = np.zeros(n_bins, dtype=np.int64)
    converge_hist_legacy = np.zeros(n_bins, dtype=np.int64)
    fail_converge_hist_legacy = np.zeros(n_bins, dtype=np.int64)

    # Initialize histograms for optimized version
    total_hist_new = np.zeros(n_bins, dtype=np.int64)
    fail_hist_new = np.zeros(n_bins, dtype=np.int64)
    converge_hist_new = np.zeros(n_bins, dtype=np.int64)
    fail_converge_hist_new = np.zeros(n_bins, dtype=np.int64)

    # Run legacy version
    start_time = time.time()
    result_legacy = _calculate_histograms_bplsd_legacy(
        values,
        fail_mask,
        bin_edges,
        total_hist_legacy,
        fail_hist_legacy,
        converge_mask,
        converge_hist_legacy,
        fail_converge_hist_legacy,
        fail_bp_mask,
    )
    legacy_time = time.time() - start_time

    # Run optimized version
    start_time = time.time()
    result_new = _calculate_histograms_bplsd_numba(
        values,
        fail_mask,
        bin_edges,
        total_hist_new,
        fail_hist_new,
        converge_mask,
        converge_hist_new,
        fail_converge_hist_new,
        fail_bp_mask,
    )
    new_time = time.time() - start_time

    # Compare results
    assert np.array_equal(
        result_legacy[0], result_new[0]
    ), "Total histograms don't match"
    assert np.array_equal(
        result_legacy[1], result_new[1]
    ), "Fail histograms don't match"
    assert np.array_equal(
        result_legacy[2], result_new[2]
    ), "Converge histograms don't match"
    assert np.array_equal(
        result_legacy[3], result_new[3]
    ), "Fail_converge histograms don't match"

    speedup = legacy_time / new_time if new_time > 0 else float("inf")
    uniform = _is_uniform_binning(bin_edges)

    print(
        f"  ✓ Results identical. Legacy: {legacy_time:.4f}s, New: {new_time:.4f}s, "
        f"Speedup: {speedup:.2f}x, Uniform: {uniform}"
    )


def test_matching_functions(n_samples: int, bin_edges: np.ndarray, test_name: str):
    """
    Test that legacy and optimized matching functions produce identical results.

    Parameters
    ----------
    n_samples : int
        Number of samples to test.
    bin_edges : 1D numpy array
        Bin edges for histogram.
    test_name : str
        Name of the test for reporting.
    """
    print(
        f"Testing matching functions: {test_name} (n={n_samples}, bins={len(bin_edges)-1})"
    )

    # Generate test data
    values, fail_mask, _, _ = generate_test_data(n_samples)

    # Remove any NaN values
    valid_mask = ~np.isnan(values)
    values = values[valid_mask]
    fail_mask = fail_mask[valid_mask]

    n_bins = len(bin_edges) - 1

    # Initialize histograms for legacy version
    total_hist_legacy = np.zeros(n_bins, dtype=np.int64)
    fail_hist_legacy = np.zeros(n_bins, dtype=np.int64)

    # Initialize histograms for optimized version
    total_hist_new = np.zeros(n_bins, dtype=np.int64)
    fail_hist_new = np.zeros(n_bins, dtype=np.int64)

    # Run legacy version
    start_time = time.time()
    result_legacy = _calculate_histograms_matching_legacy(
        values, fail_mask, bin_edges, total_hist_legacy, fail_hist_legacy
    )
    legacy_time = time.time() - start_time

    # Run optimized version
    start_time = time.time()
    result_new = _calculate_histograms_matching_numba(
        values, fail_mask, bin_edges, total_hist_new, fail_hist_new
    )
    new_time = time.time() - start_time

    # Compare results
    assert np.array_equal(
        result_legacy[0], result_new[0]
    ), "Total histograms don't match"
    assert np.array_equal(
        result_legacy[1], result_new[1]
    ), "Fail histograms don't match"

    speedup = legacy_time / new_time if new_time > 0 else float("inf")
    uniform = _is_uniform_binning(bin_edges)

    print(
        f"  ✓ Results identical. Legacy: {legacy_time:.4f}s, New: {new_time:.4f}s, "
        f"Speedup: {speedup:.2f}x, Uniform: {uniform}"
    )


def run_comprehensive_tests():
    """Run comprehensive tests comparing legacy and optimized functions."""
    print("=" * 60)
    print("COMPREHENSIVE HISTOGRAM FUNCTION TESTS")
    print("=" * 60)

    # Test uniform binning detection
    test_uniform_binning_detection()
    print()

    # Test scenarios
    test_scenarios = [
        # (n_samples, bin_edges_generator, description)
        (1000, lambda: np.linspace(0, 1, 21), "Small dataset, uniform bins"),
        (10000, lambda: np.linspace(0, 1, 51), "Medium dataset, uniform bins"),
        (50000, lambda: np.linspace(0, 1, 101), "Large dataset, uniform bins"),
        (
            1000,
            lambda: np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]),
            "Small dataset, non-uniform bins",
        ),
        (
            10000,
            lambda: np.array([0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]),
            "Medium dataset, non-uniform bins",
        ),
        (1000, lambda: np.linspace(0, 1, 11), "Small dataset, few uniform bins (≤32)"),
        (1000, lambda: np.linspace(0, 1, 65), "Small dataset, many uniform bins (>32)"),
        (
            100,
            lambda: np.array([0, 0.2, 0.8, 1.0]),
            "Very small dataset, few non-uniform bins",
        ),
    ]

    print("Testing BPLSD Functions:")
    print("-" * 30)
    for n_samples, bin_gen, description in test_scenarios:
        bin_edges = bin_gen()
        test_bplsd_functions(n_samples, bin_edges, description)

    print("\nTesting Matching Functions:")
    print("-" * 30)
    for n_samples, bin_gen, description in test_scenarios:
        bin_edges = bin_gen()
        test_matching_functions(n_samples, bin_edges, description)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("Legacy and optimized functions produce identical results.")
    print("=" * 60)


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting Edge Cases:")
    print("-" * 30)

    # Empty arrays
    print("Testing empty arrays...")
    empty_values = np.array([], dtype=np.float64)
    empty_mask = np.array([], dtype=bool)
    bin_edges = np.linspace(0, 1, 11)
    n_bins = len(bin_edges) - 1

    # Test with empty inputs
    total_hist = np.zeros(n_bins, dtype=np.int64)
    fail_hist = np.zeros(n_bins, dtype=np.int64)

    result1 = _calculate_histograms_matching_legacy(
        empty_values, empty_mask, bin_edges, total_hist.copy(), fail_hist.copy()
    )
    result2 = _calculate_histograms_matching_numba(
        empty_values, empty_mask, bin_edges, total_hist.copy(), fail_hist.copy()
    )

    assert np.array_equal(
        result1[0], result2[0]
    ), "Empty array test failed for total hist"
    assert np.array_equal(
        result1[1], result2[1]
    ), "Empty array test failed for fail hist"
    print("  ✓ Empty arrays test passed")

    # Values outside bin range
    print("Testing values outside bin range...")
    out_of_range_values = np.array([-0.5, -0.1, 1.1, 1.5, 2.0])
    out_of_range_mask = np.array([True, False, True, False, True])

    total_hist = np.zeros(n_bins, dtype=np.int64)
    fail_hist = np.zeros(n_bins, dtype=np.int64)

    result1 = _calculate_histograms_matching_legacy(
        out_of_range_values,
        out_of_range_mask,
        bin_edges,
        total_hist.copy(),
        fail_hist.copy(),
    )
    result2 = _calculate_histograms_matching_numba(
        out_of_range_values,
        out_of_range_mask,
        bin_edges,
        total_hist.copy(),
        fail_hist.copy(),
    )

    assert np.array_equal(
        result1[0], result2[0]
    ), "Out of range test failed for total hist"
    assert np.array_equal(
        result1[1], result2[1]
    ), "Out of range test failed for fail hist"
    print("  ✓ Out of range values test passed")

    # Single bin
    print("Testing single bin...")
    single_bin_edges = np.array([0.0, 1.0])
    test_values = np.array([0.0, 0.5, 1.0])
    test_mask = np.array([True, False, True])

    total_hist = np.zeros(1, dtype=np.int64)
    fail_hist = np.zeros(1, dtype=np.int64)

    result1 = _calculate_histograms_matching_legacy(
        test_values, test_mask, single_bin_edges, total_hist.copy(), fail_hist.copy()
    )
    result2 = _calculate_histograms_matching_numba(
        test_values, test_mask, single_bin_edges, total_hist.copy(), fail_hist.copy()
    )

    assert np.array_equal(
        result1[0], result2[0]
    ), "Single bin test failed for total hist"
    assert np.array_equal(
        result1[1], result2[1]
    ), "Single bin test failed for fail hist"
    print("  ✓ Single bin test passed")


if __name__ == "__main__":
    try:
        run_comprehensive_tests()
        test_edge_cases()
        print("\n🎉 All tests completed successfully!")
        print(
            "The optimized histogram functions are working correctly and produce identical results to the legacy versions."
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
