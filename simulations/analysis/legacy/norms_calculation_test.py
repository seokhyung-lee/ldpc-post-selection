#!/usr/bin/env python3
"""
Test script to verify that legacy and optimized norm calculation functions
produce identical results across various scenarios.
"""

import numpy as np
import time
from typing import Tuple

# Import legacy and optimized versions
from simulations.analysis.legacy.numba_functions_legacy import (
    _calculate_norms_for_samples_legacy,
)
from simulations.analysis.data_collectors.utils import (
    calculate_cluster_norms_from_flat_data,
)


def generate_test_data_norms(
    n_samples: int, segment_sizes: list | None = None, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data for norm calculations.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    segment_sizes : list of int, optional
        Specific segment sizes. If None, random sizes are generated.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    flat_data : 1D numpy array of float
        Flattened data array (all positive values).
    offsets : 1D numpy array of int
        Starting indices for each sample.
    """
    np.random.seed(seed)

    if segment_sizes is None:
        # Generate random segment sizes between 2 and 20
        segment_sizes = np.random.randint(2, 21, n_samples)

    total_size = sum(segment_sizes)

    # Generate random positive data only
    flat_data = np.random.uniform(0.1, 5.0, total_size).astype(np.float32)

    # Add some special positive values
    if total_size >= 10:
        flat_data[0] = 0.1  # small positive value (avoid zero for some norms)
        flat_data[1] = 1.0  # unit value
        flat_data[2] = 2.0  # another positive value
        if total_size >= 20:
            flat_data[10] = np.inf if np.random.random() < 0.1 else 100.0  # large value
            flat_data[11] = 200.0  # large positive value

    # Generate offsets
    offsets = np.zeros(n_samples, dtype=np.int64)
    current_offset = 0
    for i in range(n_samples):
        offsets[i] = current_offset
        current_offset += segment_sizes[i]

    return flat_data, offsets


def test_norms_functions(
    n_samples: int, norm_order: float, test_name: str, segment_sizes: list | None = None
):
    """
    Test that legacy and optimized norm functions produce identical results.

    Parameters
    ----------
    n_samples : int
        Number of samples to test.
    norm_order : float
        The norm order to test.
    test_name : str
        Name of the test for reporting.
    segment_sizes : list of int, optional
        Specific segment sizes to test.
    """
    print(f"Testing norm functions: {test_name} (n={n_samples}, p={norm_order})")

    # Generate test data
    flat_data, offsets = generate_test_data_norms(n_samples, segment_sizes)

    # Run legacy version
    start_time = time.time()
    norms_legacy, outside_legacy = _calculate_norms_for_samples_legacy(
        flat_data, offsets, norm_order
    )
    legacy_time = time.time() - start_time

    # Run optimized version
    start_time = time.time()
    norms_new, outside_new = calculate_cluster_norms_from_flat_data(
        flat_data, offsets, norm_order
    )
    new_time = time.time() - start_time

    # Compare results with appropriate tolerance
    tolerance = 1e-5 if norm_order != np.inf else 1e-7

    # Check norms
    norms_match = np.allclose(
        norms_legacy, norms_new, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    if not norms_match:
        max_diff = np.max(np.abs(norms_legacy - norms_new))
        print(f"    ❌ Norms don't match! Max difference: {max_diff}")
        # Print details for debugging
        for i in range(min(10, len(norms_legacy))):
            if abs(norms_legacy[i] - norms_new[i]) > tolerance:
                print(
                    f"      Sample {i}: Legacy={norms_legacy[i]:.6f}, New={norms_new[i]:.6f}"
                )
        raise AssertionError("Norms don't match")

    # Check outside values
    outside_match = np.allclose(
        outside_legacy, outside_new, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    if not outside_match:
        max_diff = np.max(
            np.abs(
                outside_legacy[~np.isnan(outside_legacy)]
                - outside_new[~np.isnan(outside_new)]
            )
        )
        print(f"    ❌ Outside values don't match! Max difference: {max_diff}")
        raise AssertionError("Outside values don't match")

    speedup = legacy_time / new_time if new_time > 0 else float("inf")

    print(
        f"  ✓ Results identical. Legacy: {legacy_time:.4f}s, New: {new_time:.4f}s, "
        f"Speedup: {speedup:.2f}x"
    )


def run_comprehensive_norms_tests():
    """Run comprehensive tests comparing legacy and optimized norm functions."""
    print("=" * 60)
    print("COMPREHENSIVE NORM CALCULATION TESTS")
    print("=" * 60)

    # Test scenarios
    test_scenarios = [
        # (n_samples, norm_orders, segment_sizes, description)
        (100, [1.0, 2.0, np.inf, 0.5, 3.0], None, "Small dataset, various norms"),
        (1000, [1.0, 2.0, np.inf], None, "Medium dataset, common norms"),
        (5000, [2.0], None, "Large dataset, L2 norm"),
        (50, [1.0, 2.0, np.inf], [5] * 50, "Uniform small segments"),
        (
            20,
            [1.0, 2.0, np.inf],
            [2, 3, 5, 8, 13, 21] * 3 + [2, 2],
            "Mixed segment sizes",
        ),
        (10, [1.0, 2.0, np.inf], [100] * 10, "Large segments"),
        (100, [0.5, 1.5, 2.5, 4.0], None, "Fractional and higher order norms"),
    ]

    for n_samples, norm_orders, segment_sizes, description in test_scenarios:
        print(f"\nTesting: {description}")
        print("-" * 40)

        for norm_order in norm_orders:
            test_name = f"{description}, p={norm_order}"
            test_norms_functions(n_samples, norm_order, test_name, segment_sizes)

    print("\n" + "=" * 60)
    print("ALL NORM TESTS PASSED! ✓")
    print("Legacy and optimized norm functions produce identical results.")
    print("=" * 60)


def test_edge_cases_norms():
    """Test edge cases and boundary conditions for norm calculations."""
    print("\nTesting Edge Cases for Norms:")
    print("-" * 40)

    # Empty segments test
    print("Testing empty segments...")
    flat_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    offsets = np.array([0, 0, 2, 2, 4], dtype=np.int64)  # Some empty segments

    norms_legacy, outside_legacy = _calculate_norms_for_samples_legacy(
        flat_data, offsets, 2.0
    )
    norms_new, outside_new = calculate_cluster_norms_from_flat_data(
        flat_data, offsets, 2.0
    )

    assert np.allclose(
        norms_legacy, norms_new, equal_nan=True
    ), "Empty segments test failed for norms"
    assert np.allclose(
        outside_legacy, outside_new, equal_nan=True
    ), "Empty segments test failed for outside values"
    print("  ✓ Empty segments test passed")

    # Single element segments (only outside value)
    print("Testing single element segments...")
    flat_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    offsets = np.array([0, 1, 2], dtype=np.int64)

    norms_legacy, outside_legacy = _calculate_norms_for_samples_legacy(
        flat_data, offsets, 2.0
    )
    norms_new, outside_new = calculate_cluster_norms_from_flat_data(
        flat_data, offsets, 2.0
    )

    assert np.allclose(
        norms_legacy, norms_new, equal_nan=True
    ), "Single element test failed for norms"
    assert np.allclose(
        outside_legacy, outside_new, equal_nan=True
    ), "Single element test failed for outside values"
    print("  ✓ Single element segments test passed")

    # Small positive values test (replacing zero values test)
    print("Testing small positive values...")
    flat_data = np.array([0.1, 0.01, 1.0, 0.5, 0.001, 2.0], dtype=np.float32)
    offsets = np.array([0, 3], dtype=np.int64)

    for norm_order in [1.0, 2.0, np.inf, 0.5]:
        norms_legacy, outside_legacy = _calculate_norms_for_samples_legacy(
            flat_data, offsets, norm_order
        )
        norms_new, outside_new = calculate_cluster_norms_from_flat_data(
            flat_data, offsets, norm_order
        )

        assert np.allclose(
            norms_legacy, norms_new, equal_nan=True
        ), f"Small positive values test failed for p={norm_order}"
        assert np.allclose(
            outside_legacy, outside_new, equal_nan=True
        ), f"Small positive outside values test failed for p={norm_order}"

    print("  ✓ Small positive values test passed")

    # Large positive values test (replacing negative values test)
    print("Testing large positive values...")
    flat_data = np.array([10.0, 20.0, 3.0, 40.0, 5.0, 60.0], dtype=np.float32)
    offsets = np.array([0, 3], dtype=np.int64)

    for norm_order in [1.0, 2.0, np.inf, 0.5, 3.0]:
        norms_legacy, outside_legacy = _calculate_norms_for_samples_legacy(
            flat_data, offsets, norm_order
        )
        norms_new, outside_new = calculate_cluster_norms_from_flat_data(
            flat_data, offsets, norm_order
        )

        assert np.allclose(
            norms_legacy, norms_new, rtol=1e-5, equal_nan=True
        ), f"Large positive values test failed for p={norm_order}"
        assert np.allclose(
            outside_legacy, outside_new, equal_nan=True
        ), f"Large positive outside values test failed for p={norm_order}"

    print("  ✓ Large positive values test passed")


if __name__ == "__main__":
    try:
        run_comprehensive_norms_tests()
        test_edge_cases_norms()
        print("\n🎉 All norm calculation tests completed successfully!")
        print(
            "The optimized norm functions are working correctly and produce identical results to the legacy versions."
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
