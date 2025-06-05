import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import sys
from pathlib import Path

# Add the parent directory to sys.path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_post_processing import (
    _apply_ci_vectorized,
    get_df_ps,
    _can_reuse_existing_data,
    _calculate_df_ps_full,
)


class TestApplyCiVectorized:
    """Test cases for _apply_ci_vectorized function."""

    def test_valid_inputs(self):
        """Test with valid N and k series."""
        N_series = pd.Series([100, 200, 150])
        k_series = pd.Series([30, 80, 45])

        p_values, delta_p_values = _apply_ci_vectorized(N_series, k_series)

        assert len(p_values) == len(N_series), "Output length should match input"
        assert len(delta_p_values) == len(N_series), "Output length should match input"
        assert all(p_values.notna()), "All p_values should be valid"
        assert all(delta_p_values.notna()), "All delta_p_values should be valid"

    def test_nan_inputs(self):
        """Test with NaN inputs."""
        N_series = pd.Series([100, np.nan, 150])
        k_series = pd.Series([30, 80, np.nan])

        p_values, delta_p_values = _apply_ci_vectorized(N_series, k_series)

        # First value should be valid
        assert pd.notna(p_values.iloc[0]), "First p_value should be valid"
        assert pd.notna(delta_p_values.iloc[0]), "First delta_p_value should be valid"

        # Second and third values should be NaN due to NaN inputs
        assert pd.isna(p_values.iloc[1]), "Second p_value should be NaN"
        assert pd.isna(p_values.iloc[2]), "Third p_value should be NaN"

    def test_empty_series(self):
        """Test with empty series."""
        N_series = pd.Series([])
        k_series = pd.Series([])

        p_values, delta_p_values = _apply_ci_vectorized(N_series, k_series)

        assert len(p_values) == 0, "Output should be empty"
        assert len(delta_p_values) == 0, "Output should be empty"

    def test_mismatched_indices(self):
        """Test with different indices."""
        N_series = pd.Series([100, 200], index=["a", "b"])
        k_series = pd.Series([30, 80], index=["a", "b"])

        p_values, delta_p_values = _apply_ci_vectorized(N_series, k_series)

        # Check that indices are preserved
        pd.testing.assert_index_equal(p_values.index, N_series.index)
        pd.testing.assert_index_equal(delta_p_values.index, N_series.index)


class TestCanReuseExistingData:
    """Test cases for _can_reuse_existing_data function."""

    def test_can_reuse_matching_data(self):
        """Test when existing data can be reused."""
        # Create df_agg
        df_agg = pd.DataFrame(
            {"count": [100, 200, 150], "num_fails": [30, 80, 45]}, index=[1, 2, 3]
        )

        # Create existing_df_ps with matching total count
        existing_df_ps = pd.DataFrame(
            {
                "count": [450, 350, 150],  # Cumulative, max = 450 = sum of df_agg
                "p_fail": [0.1, 0.2, 0.3],
            },
            index=[1, 2, 3],
        )

        result = _can_reuse_existing_data(df_agg, existing_df_ps)
        assert result == True, "Should be able to reuse data with matching counts"

    def test_cannot_reuse_different_counts(self):
        """Test when existing data cannot be reused due to different counts."""
        # Create df_agg
        df_agg = pd.DataFrame(
            {"count": [100, 200, 150], "num_fails": [30, 80, 45]}, index=[1, 2, 3]
        )

        # Create existing_df_ps with different total count
        existing_df_ps = pd.DataFrame(
            {"count": [400, 300, 100], "p_fail": [0.1, 0.2, 0.3]},  # Max = 400 != 450
            index=[1, 2, 3],
        )

        result = _can_reuse_existing_data(df_agg, existing_df_ps)
        assert result == False, "Should not reuse data with different counts"

    def test_cannot_reuse_missing_indices(self):
        """Test when existing data cannot be reused due to missing indices."""
        # Create df_agg
        df_agg = pd.DataFrame(
            {"count": [100, 200, 150], "num_fails": [30, 80, 45]}, index=[1, 2, 3]
        )

        # Create existing_df_ps missing some indices
        existing_df_ps = pd.DataFrame(
            {"count": [300, 150], "p_fail": [0.1, 0.3]}, index=[1, 3]  # Missing index 2
        )

        result = _can_reuse_existing_data(df_agg, existing_df_ps)
        assert result == False, "Should not reuse data with missing indices"

    def test_exception_handling(self):
        """Test exception handling in _can_reuse_existing_data."""
        df_agg = pd.DataFrame(
            {"count": [100, 200], "num_fails": [30, 80]}, index=[1, 2]
        )

        # Create existing_df_ps without 'count' column
        existing_df_ps = pd.DataFrame({"p_fail": [0.1, 0.2]}, index=[1, 2])

        result = _can_reuse_existing_data(df_agg, existing_df_ps)
        assert result == False, "Should return False when KeyError occurs"


class TestCalculateDfPsFull:
    """Test cases for _calculate_df_ps_full function."""

    def setup_method(self):
        """Set up test data."""
        self.df_agg_basic = pd.DataFrame(
            {"count": [100, 200, 150], "num_fails": [30, 80, 45]}, index=[1, 2, 3]
        )

        self.df_agg_with_conv = pd.DataFrame(
            {
                "count": [100, 200, 150],
                "num_fails": [30, 80, 45],
                "num_converged": [10, 20, 15],
                "num_converged_fails": [3, 8, 5],
            },
            index=[1, 2, 3],
        )

    def test_basic_calculation_ascending_false(self):
        """Test basic calculation with ascending_confidence=False."""
        result = _calculate_df_ps_full(self.df_agg_basic, ascending_confidence=False)

        # Check that required columns exist
        required_cols = [
            "p_fail",
            "delta_p_fail",
            "p_abort",
            "delta_p_abort",
            "count",
            "num_fails",
        ]
        for col in required_cols:
            assert col in result.columns, f"Column {col} should exist"

        # Check that cumulative counts are correct (ascending=False means normal cumsum)
        expected_count_cum = self.df_agg_basic["count"].cumsum()
        pd.testing.assert_series_equal(result["count"], expected_count_cum)

        expected_fails_cum = self.df_agg_basic["num_fails"].cumsum()
        pd.testing.assert_series_equal(result["num_fails"], expected_fails_cum)

    def test_basic_calculation_ascending_true(self):
        """Test basic calculation with ascending_confidence=True."""
        result = _calculate_df_ps_full(self.df_agg_basic, ascending_confidence=True)

        # Check that cumulative counts are correct (ascending=True means reverse, cumsum, reverse)
        expected_count_cum = self.df_agg_basic["count"][::-1].cumsum()[::-1]
        pd.testing.assert_series_equal(result["count"], expected_count_cum)

        expected_fails_cum = self.df_agg_basic["num_fails"][::-1].cumsum()[::-1]
        pd.testing.assert_series_equal(result["num_fails"], expected_fails_cum)

    def test_with_convergence_data_ascending_false(self):
        """Test with convergence data and ascending_confidence=False."""
        result = _calculate_df_ps_full(
            self.df_agg_with_conv, ascending_confidence=False
        )

        # Check that convergence columns exist
        conv_cols = [
            "p_fail_conv",
            "delta_p_fail_conv",
            "p_abort_conv",
            "delta_p_abort_conv",
            "count_conv",
            "num_fails_conv",
        ]
        for col in conv_cols:
            assert col in result.columns, f"Convergence column {col} should exist"
            assert result[col].notna().any(), f"Column {col} should have valid values"

    def test_with_convergence_data_ascending_true(self):
        """Test with convergence data and ascending_confidence=True."""
        result = _calculate_df_ps_full(self.df_agg_with_conv, ascending_confidence=True)

        # Check that convergence columns exist and have valid values
        conv_cols = [
            "p_fail_conv",
            "delta_p_fail_conv",
            "p_abort_conv",
            "delta_p_abort_conv",
            "count_conv",
            "num_fails_conv",
        ]
        for col in conv_cols:
            assert col in result.columns, f"Convergence column {col} should exist"
            assert result[col].notna().any(), f"Column {col} should have valid values"

    def test_without_convergence_data(self):
        """Test that convergence columns are filled with NaN when no convergence data."""
        result = _calculate_df_ps_full(self.df_agg_basic, ascending_confidence=False)

        conv_cols = [
            "p_fail_conv",
            "delta_p_fail_conv",
            "p_abort_conv",
            "delta_p_abort_conv",
            "count_conv",
            "num_fails_conv",
        ]
        for col in conv_cols:
            assert col in result.columns, f"Convergence column {col} should exist"
            assert result[col].isna().all(), f"Column {col} should be all NaN"

    def test_probability_constraints(self):
        """Test that calculated probabilities satisfy basic constraints."""
        result = _calculate_df_ps_full(self.df_agg_basic, ascending_confidence=False)

        # Check that probabilities are between 0 and 1
        assert (result["p_fail"] >= 0).all() and (
            result["p_fail"] <= 1
        ).all(), "p_fail should be in [0,1]"
        assert (result["p_abort"] >= 0).all() and (
            result["p_abort"] <= 1
        ).all(), "p_abort should be in [0,1]"

        # Check that confidence intervals are non-negative
        assert (
            result["delta_p_fail"] >= 0
        ).all(), "delta_p_fail should be non-negative"
        assert (
            result["delta_p_abort"] >= 0
        ).all(), "delta_p_abort should be non-negative"

    def test_empty_dataframe(self):
        """Test with empty input dataframe."""
        empty_df = pd.DataFrame(columns=["count", "num_fails"])
        result = _calculate_df_ps_full(empty_df, ascending_confidence=False)

        assert len(result) == 0, "Result should be empty for empty input"

        # Check that all expected columns exist even for empty result
        expected_cols = [
            "p_fail",
            "delta_p_fail",
            "p_abort",
            "delta_p_abort",
            "count",
            "num_fails",
            "p_fail_conv",
            "delta_p_fail_conv",
            "p_abort_conv",
            "delta_p_abort_conv",
            "count_conv",
            "num_fails_conv",
        ]
        for col in expected_cols:
            assert (
                col in result.columns
            ), f"Column {col} should exist even for empty result"


class TestGetDfPs:
    """Test cases for get_df_ps function."""

    def setup_method(self):
        """Set up test data."""
        self.df_agg = pd.DataFrame(
            {"count": [100, 200, 150], "num_fails": [30, 80, 45]}, index=[1, 2, 3]
        )

        self.df_agg_with_conv = pd.DataFrame(
            {
                "count": [100, 200, 150],
                "num_fails": [30, 80, 45],
                "num_converged": [10, 20, 15],
                "num_converged_fails": [3, 8, 5],
            },
            index=[1, 2, 3],
        )

    def test_no_existing_data(self):
        """Test with no existing df_ps data."""
        result = get_df_ps(self.df_agg, ascending_confidence=True)

        # Should return a complete calculation
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(self.df_agg), "Length should match input"
        pd.testing.assert_index_equal(result.index, self.df_agg.index)

    def test_reusable_existing_data(self):
        """Test with reusable existing data."""
        # First calculate df_ps
        initial_result = get_df_ps(self.df_agg, ascending_confidence=True)

        # Now call again with existing data that can be reused
        result = get_df_ps(
            self.df_agg, ascending_confidence=True, existing_df_ps=initial_result
        )

        # Should get the same result
        pd.testing.assert_frame_equal(result, initial_result)

    def test_non_reusable_existing_data(self):
        """Test with non-reusable existing data."""
        # Create initial df_ps with different total count
        fake_existing = pd.DataFrame(
            {"count": [400, 300, 100], "p_fail": [0.1, 0.2, 0.3]},  # Different total
            index=[1, 2, 3],
        )

        # Should recalculate everything
        result = get_df_ps(
            self.df_agg, ascending_confidence=True, existing_df_ps=fake_existing
        )

        # Should not match the fake existing data
        assert not result["count"].equals(
            fake_existing["count"]
        ), "Should recalculate with different totals"

    def test_ascending_confidence_parameter(self):
        """Test different values of ascending_confidence parameter."""
        result_asc = get_df_ps(self.df_agg, ascending_confidence=True)
        result_desc = get_df_ps(self.df_agg, ascending_confidence=False)

        # Results should be different
        assert not result_asc["count"].equals(
            result_desc["count"]
        ), "Different ascending should give different results"

    def test_with_convergence_columns(self):
        """Test with convergence columns in input data."""
        result = get_df_ps(self.df_agg_with_conv, ascending_confidence=True)

        # Should have convergence columns
        conv_cols = [
            "p_fail_conv",
            "delta_p_fail_conv",
            "p_abort_conv",
            "delta_p_abort_conv",
            "count_conv",
            "num_fails_conv",
        ]
        for col in conv_cols:
            assert col in result.columns, f"Should have convergence column {col}"
            assert (
                result[col].notna().any()
            ), f"Convergence column {col} should have valid values"

    def test_index_preservation(self):
        """Test that the index is properly preserved."""
        # Use a more complex index
        complex_index = pd.Index([0.1, 0.5, 1.0], name="pred_llr")
        df_with_complex_index = self.df_agg.copy()
        df_with_complex_index.index = complex_index

        result = get_df_ps(df_with_complex_index, ascending_confidence=True)

        pd.testing.assert_index_equal(result.index, complex_index)

    def test_data_types(self):
        """Test that output data types are appropriate."""
        result = get_df_ps(self.df_agg, ascending_confidence=True)

        # Numeric columns should be float
        numeric_cols = ["p_fail", "delta_p_fail", "p_abort", "delta_p_abort"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(
                result[col]
            ), f"Column {col} should be numeric"

    def test_single_row_input(self):
        """Test with single row input."""
        single_row = self.df_agg.iloc[[0]]
        result = get_df_ps(single_row, ascending_confidence=True)

        assert len(result) == 1, "Should handle single row input"

        # Check that basic columns have valid values (convergence columns will be NaN)
        basic_cols = [
            "p_fail",
            "delta_p_fail",
            "p_abort",
            "delta_p_abort",
            "count",
            "num_fails",
        ]
        for col in basic_cols:
            assert (
                result[col].notna().all()
            ), f"Basic column {col} should have valid values"


if __name__ == "__main__":
    pytest.main([__file__])
