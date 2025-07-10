import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from simulations.analysis.data_aggregation import (
    aggregate_data,
    calculate_df_agg_for_combination,
    _get_values_for_binning_from_batch,
    _create_bin_edges,
    _determine_value_range_single,
    _process_histograms_from_batches_single,
    _build_result_dataframe_single,
    aggregate_data_batch,
)


class TestDataAggregation:
    """Test suite for data aggregation functions."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.data_dir)

    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)

    def create_test_batch(
        self,
        batch_name: str,
        num_samples: int = 100,
        include_bplsd: bool = True,
        include_npy: bool = False,
    ):
        """Create a test batch directory with mock data."""
        batch_dir = os.path.join(self.data_dir, batch_name)
        os.makedirs(batch_dir)

        # Create scalars.feather
        np.random.seed(42)  # For reproducible tests
        data = {
            "fail": np.random.choice([True, False], size=num_samples),
            "pred_llr": np.random.normal(0, 1, size=num_samples),
            "detector_density": np.random.uniform(0, 1, size=num_samples),
        }

        if include_bplsd:
            data["converge"] = np.random.choice([True, False], size=num_samples)
            data["fail_bp"] = np.random.choice([True, False], size=num_samples)

        df = pd.DataFrame(data)
        df.to_feather(os.path.join(batch_dir, "scalars.feather"))

        # Create .npy files if requested
        if include_npy:
            # Create cluster data
            total_clusters = num_samples * 3  # Average 3 clusters per sample
            cluster_sizes = np.random.randint(1, 10, size=total_clusters)
            cluster_llrs = np.random.normal(0, 2, size=total_clusters)

            # Create offsets (cumulative sum)
            clusters_per_sample = np.random.poisson(3, size=num_samples)
            offsets = np.concatenate([[0], np.cumsum(clusters_per_sample)])

            # Adjust arrays to match offsets
            actual_total = offsets[-1]
            cluster_sizes = cluster_sizes[:actual_total]
            cluster_llrs = cluster_llrs[:actual_total]

            np.save(os.path.join(batch_dir, "cluster_sizes.npy"), cluster_sizes)
            np.save(os.path.join(batch_dir, "cluster_llrs.npy"), cluster_llrs)
            np.save(os.path.join(batch_dir, "offsets.npy"), offsets)

        return batch_dir

    def test_get_values_for_binning_basic_column(self):
        """Test _get_values_for_binning_from_batch with basic column."""
        batch_dir = self.create_test_batch("batch_0001")

        series, df_scalars, original_batch_size = _get_values_for_binning_from_batch(
            batch_dir, by="pred_llr", norm_order=None, verbose=False
        )

        assert series is not None
        assert df_scalars is not None
        assert len(series) == 100
        assert len(df_scalars) == 100
        assert "fail" in df_scalars.columns
        assert "pred_llr" in df_scalars.columns

    def test_get_values_for_binning_missing_file(self):
        """Test _get_values_for_binning_from_batch with missing scalars.feather."""
        batch_dir = os.path.join(self.data_dir, "empty_batch")
        os.makedirs(batch_dir)

        with pytest.raises(FileNotFoundError):
            _get_values_for_binning_from_batch(
                batch_dir, by="pred_llr", norm_order=None, verbose=False
            )

    def test_get_values_for_binning_norm_methods(self):
        """Test _get_values_for_binning_from_batch with norm-based methods."""
        batch_dir = self.create_test_batch("batch_0001", include_npy=True)

        # Test cluster_size_norm
        series, df_scalars, original_batch_size = _get_values_for_binning_from_batch(
            batch_dir, by="cluster_size_norm", norm_order=2.0, verbose=False
        )

        assert series is not None
        assert df_scalars is not None
        assert len(series) == 100

    def test_get_values_for_binning_norm_missing_files(self):
        """Test norm methods with missing .npy files."""
        batch_dir = self.create_test_batch("batch_0001", include_npy=False)

        with pytest.raises(FileNotFoundError):
            _get_values_for_binning_from_batch(
                batch_dir, by="cluster_size_norm", norm_order=2.0, verbose=False
            )

    def test_create_bin_edges_normal(self):
        """Test _create_bin_edges with normal inputs."""
        bin_edges, num_bins = _create_bin_edges(
            actual_min_val=0.0,
            actual_max_val=10.0,
            num_hist_bins=10,
            by="pred_llr",
            verbose=False,
        )

        assert len(bin_edges) == 11  # n+1 edges for n bins
        assert num_bins == 10
        assert bin_edges[0] == 0.0
        assert bin_edges[-1] == 10.0

    def test_create_bin_edges_edge_cases(self):
        """Test _create_bin_edges with edge cases."""
        # Test when min == max
        bin_edges, num_bins = _create_bin_edges(
            actual_min_val=5.0,
            actual_max_val=5.0,
            num_hist_bins=10,
            by="pred_llr",
            verbose=False,
        )

        assert num_bins == 1
        assert len(bin_edges) == 2

    def test_determine_value_range_single(self):
        """Test _determine_value_range_single."""
        # Create multiple batches
        batch_dirs = []
        for i in range(3):
            batch_dirs.append(self.create_test_batch(f"batch_{i:04d}"))

        min_val, max_val = _determine_value_range_single(
            batch_dir_paths=batch_dirs,
            by="pred_llr",
            norm_order=None,
            min_value_override=None,
            max_value_override=None,
            verbose=False,
        )

        assert isinstance(min_val, float)
        assert isinstance(max_val, float)
        assert min_val <= max_val

    def test_determine_value_range_single_with_overrides(self):
        """Test _determine_value_range_single with user overrides."""
        batch_dirs = [self.create_test_batch("batch_0001")]

        min_val, max_val = _determine_value_range_single(
            batch_dir_paths=batch_dirs,
            by="pred_llr",
            norm_order=None,
            min_value_override=-5.0,
            max_value_override=5.0,
            verbose=False,
        )

        assert min_val == -5.0
        assert max_val == 5.0

    def test_build_result_dataframe_single(self):
        """Test _build_result_dataframe_single."""
        # Mock histogram data
        total_counts = np.array([10, 20, 30, 15, 5])
        fail_counts = np.array([2, 4, 6, 3, 1])
        converge_counts = np.array([8, 16, 24, 12, 4])
        fail_converge_counts = np.array([1, 2, 3, 2, 1])
        bin_edges = np.array([0, 1, 2, 3, 4, 5])

        df = _build_result_dataframe_single(
            total_counts_hist=total_counts,
            fail_counts_hist=fail_counts,
            converge_counts_hist=converge_counts,
            fail_converge_counts_hist=fail_converge_counts,
            bin_edges=bin_edges,
            num_hist_bins=5,
            by="pred_llr",
            ascending_confidence=True,
            has_bplsd_data=True,
        )

        assert not df.empty
        assert df.index.name == "pred_llr"
        assert "count" in df.columns
        assert "num_fails" in df.columns
        assert "num_converged" in df.columns
        assert "num_converged_fails" in df.columns
        assert len(df) == 5  # All bins have counts > 0

    def test_calculate_df_agg_for_combination(self):
        """Test calculate_df_agg_for_combination."""
        # Create test batches
        for i in range(3):
            self.create_test_batch(f"batch_{i:04d}")

        df_agg, total_rows = calculate_df_agg_for_combination(
            data_dir=self.data_dir, num_hist_bins=10, by="pred_llr", verbose=False
        )

        assert not df_agg.empty
        assert df_agg.index.name == "pred_llr"
        assert "count" in df_agg.columns
        assert "num_fails" in df_agg.columns
        assert total_rows > 0

    def test_calculate_df_agg_for_combination_no_batches(self):
        """Test calculate_df_agg_for_combination with no batch directories."""
        with pytest.raises(FileNotFoundError):
            calculate_df_agg_for_combination(
                data_dir=self.data_dir, by="pred_llr", verbose=False
            )

    def test_aggregate_data_basic(self):
        """Test aggregate_data basic functionality."""
        # Create test batches
        for i in range(2):
            self.create_test_batch(f"batch_{i:04d}")

        df_agg = aggregate_data(
            self.data_dir, by="pred_llr", num_hist_bins=5, verbose=False
        )

        assert not df_agg.empty
        assert df_agg.index.name == "pred_llr"
        assert "count" in df_agg.columns
        assert "num_fails" in df_agg.columns

    def test_aggregate_data_with_value_range(self):
        """Test aggregate_data with user-specified value range."""
        self.create_test_batch("batch_0001")

        # Use a wider range that will encompass the test data
        df_agg = aggregate_data(
            self.data_dir,
            by="pred_llr",
            value_range=(-5.0, 5.0),  # Wider range to accommodate test data
            num_hist_bins=5,
            verbose=False,
        )

        assert not df_agg.empty

    def test_aggregate_data_with_existing_data_match(self):
        """Test aggregate_data with existing data that matches."""
        self.create_test_batch("batch_0001")

        # First run to get baseline
        df_original = aggregate_data(
            self.data_dir, by="pred_llr", num_hist_bins=5, verbose=False
        )

        # Mock get_existing_shots to return matching count
        with patch(
            "simulations.analysis.data_aggregation.get_existing_shots"
        ) as mock_shots:
            mock_shots.return_value = (df_original["count"].sum(), {})

            df_reused = aggregate_data(
                self.data_dir,
                by="pred_llr",
                df_existing=df_original,
                num_hist_bins=5,
                verbose=False,
            )

            pd.testing.assert_frame_equal(df_original, df_reused)

    def test_aggregate_data_with_existing_data_no_match(self):
        """Test aggregate_data with existing data that doesn't match."""
        self.create_test_batch("batch_0001")

        # Create fake existing data
        existing_data = pd.DataFrame({"count": [10, 20, 30], "num_fails": [1, 2, 3]})
        existing_data.index = pd.Index([0.1, 0.2, 0.3], name="pred_llr")

        # Mock get_existing_shots to return different count
        with patch(
            "simulations.analysis.data_aggregation.get_existing_shots"
        ) as mock_shots:
            mock_shots.return_value = (1000, {})  # Different from existing sum

            df_new = aggregate_data(
                self.data_dir,
                by="pred_llr",
                df_existing=existing_data,
                num_hist_bins=5,
                verbose=False,
            )

            # Should reprocess and return new data
            assert not df_new.equals(existing_data)

    def test_aggregate_data_norm_methods(self):
        """Test aggregate_data with norm-based methods."""
        self.create_test_batch("batch_0001", include_npy=True)

        df_agg = aggregate_data(
            self.data_dir,
            by="cluster_size_norm",
            norm_order=2.0,
            num_hist_bins=5,
            verbose=False,
        )

        assert not df_agg.empty
        assert df_agg.index.name == "cluster_size_norm"

    def test_aggregate_data_invalid_directory(self):
        """Test aggregate_data with invalid directory."""
        with pytest.raises(FileNotFoundError):
            aggregate_data("/nonexistent/path", by="pred_llr", verbose=False)

    def test_aggregate_data_missing_column(self):
        """Test aggregate_data with missing column."""
        self.create_test_batch("batch_0001")

        with pytest.raises(ValueError):
            aggregate_data(self.data_dir, by="nonexistent_column", verbose=False)

    def test_aggregate_data_invalid_value_range(self):
        """Test aggregate_data with invalid value range."""
        self.create_test_batch("batch_0001")

        # Should handle invalid range gracefully
        df_agg = aggregate_data(
            self.data_dir,
            by="pred_llr",
            value_range=(5.0, 1.0),  # max < min
            verbose=False,
        )

        assert not df_agg.empty  # Should still work with auto-detected range

    def test_aggregate_data_norm_without_norm_order(self):
        """Test aggregate_data with norm method but no norm_order."""
        self.create_test_batch("batch_0001", include_npy=True)

        with pytest.raises(ValueError):
            aggregate_data(
                self.data_dir,
                by="cluster_size_norm",
                norm_order=None,  # Should be required
                verbose=False,
            )

    def test_process_histograms_from_batches_single(self):
        """Test _process_histograms_from_batches_single."""
        # Create test batches
        batch_dirs = []
        for i in range(2):
            batch_dirs.append(self.create_test_batch(f"batch_{i:04d}"))

        bin_edges = np.linspace(-3, 3, 11)

        result = _process_histograms_from_batches_single(
            batch_dir_paths=batch_dirs,
            by="pred_llr",
            norm_order=None,
            bin_edges=bin_edges,
            num_hist_bins=10,
            min_value_override=None,
            max_value_override=None,
            verbose=False,
        )

        (
            total_counts,
            fail_counts,
            converge_counts,
            fail_converge_counts,
            total_rows,
            total_samples,
            read_time,
            calc_time,
            hist_time,
        ) = result

        assert len(total_counts) == 10
        assert len(fail_counts) == 10
        assert total_rows > 0
        assert total_samples > 0
        assert read_time >= 0
        assert calc_time >= 0
        assert hist_time >= 0

    def create_test_subdir_structure(self, base_dir: str, subdirs: list[str]):
        """Create a test directory structure with multiple subdirectories containing batches."""
        for subdir_name in subdirs:
            subdir_path = os.path.join(base_dir, subdir_name)
            os.makedirs(subdir_path)

            # Create a few batches in each subdirectory
            for i in range(2):
                batch_name = f"batch_{i:04d}"
                batch_dir = os.path.join(subdir_path, batch_name)
                os.makedirs(batch_dir)

                # Create scalars.feather
                seed_value = (
                    42 + (hash(subdir_name) % 1000) + i
                )  # Ensure seed is reasonable
                np.random.seed(seed_value)
                data = {
                    "fail": np.random.choice([True, False], size=50),
                    "pred_llr": np.random.normal(0, 1, size=50),
                    "detector_density": np.random.uniform(0, 1, size=50),
                    "converge": np.random.choice([True, False], size=50),
                    "fail_bp": np.random.choice([True, False], size=50),
                }

                df = pd.DataFrame(data)
                df.to_feather(os.path.join(batch_dir, "scalars.feather"))

    def test_aggregate_data_batch_basic(self):
        """Test aggregate_data_batch basic functionality."""
        # Create test subdirectory structure
        subdirs = ["combo_1", "combo_2", "combo_3"]
        self.create_test_subdir_structure(self.data_dir, subdirs)

        results = aggregate_data_batch(
            self.data_dir, by="pred_llr", num_hist_bins=5, verbose=False
        )

        assert isinstance(results, dict)
        assert len(results) == 3

        for subdir_name in subdirs:
            assert subdir_name in results
            df = results[subdir_name]
            assert not df.empty
            assert df.index.name == "pred_llr"
            assert "count" in df.columns
            assert "num_fails" in df.columns

    def test_aggregate_data_batch_no_subdirs(self):
        """Test aggregate_data_batch with no valid subdirectories."""
        # Create empty directory
        results = aggregate_data_batch(self.data_dir, by="pred_llr", verbose=False)

        assert isinstance(results, dict)
        assert len(results) == 0

    def test_aggregate_data_batch_invalid_directory(self):
        """Test aggregate_data_batch with invalid base directory."""
        with pytest.raises(FileNotFoundError):
            aggregate_data_batch("/nonexistent/path", by="pred_llr", verbose=False)

    def test_aggregate_data_batch_with_existing_data(self):
        """Test aggregate_data_batch with existing data dictionary."""
        # Create test subdirectory structure
        subdirs = ["combo_1", "combo_2"]
        self.create_test_subdir_structure(self.data_dir, subdirs)

        # Create fake existing data for one subdirectory
        existing_data = pd.DataFrame({"count": [10, 20, 30], "num_fails": [1, 2, 3]})
        existing_data.index = pd.Index([0.1, 0.2, 0.3], name="pred_llr")

        df_existing_dict = {"combo_1": existing_data}

        # Mock get_existing_shots to return matching count for combo_1
        with patch(
            "simulations.analysis.data_aggregation.get_existing_shots"
        ) as mock_shots:

            def mock_shots_side_effect(path):
                if "combo_1" in path:
                    return (existing_data["count"].sum(), {})
                else:
                    return (1000, {})  # Different count for combo_2

            mock_shots.side_effect = mock_shots_side_effect

            results = aggregate_data_batch(
                self.data_dir,
                by="pred_llr",
                df_existing_dict=df_existing_dict,
                num_hist_bins=5,
                verbose=False,
            )

        assert len(results) == 2

        # combo_1 should reuse existing data
        pd.testing.assert_frame_equal(results["combo_1"], existing_data)

        # combo_2 should be newly processed
        assert not results["combo_2"].empty
        assert not results["combo_2"].equals(existing_data)

    def test_aggregate_data_batch_with_errors(self):
        """Test aggregate_data_batch with some subdirectories causing errors."""
        # Create test subdirectory structure
        subdirs = ["combo_1", "combo_2", "combo_bad"]
        self.create_test_subdir_structure(self.data_dir, subdirs)

        # Remove scalars.feather from one batch to cause an error
        bad_batch_path = os.path.join(
            self.data_dir, "combo_bad", "batch_0000", "scalars.feather"
        )
        os.remove(bad_batch_path)

        results = aggregate_data_batch(
            self.data_dir, by="pred_llr", num_hist_bins=5, verbose=False
        )

        assert len(results) == 3

        # combo_1 and combo_2 should succeed
        assert not results["combo_1"].empty
        assert not results["combo_2"].empty

        # combo_bad should have empty DataFrame due to error
        assert results["combo_bad"].empty

    def test_aggregate_data_batch_norm_methods(self):
        """Test aggregate_data_batch with norm-based methods."""
        # Create test subdirectory structure with .npy files
        subdirs = ["combo_1", "combo_2"]
        for subdir_name in subdirs:
            subdir_path = os.path.join(self.data_dir, subdir_name)
            os.makedirs(subdir_path)

            # Create batches with .npy files
            for i in range(2):
                batch_name = f"batch_{i:04d}"
                self.create_test_batch(batch_name, include_npy=True)

                # Move the batch to the subdirectory
                old_path = os.path.join(self.data_dir, batch_name)
                new_path = os.path.join(subdir_path, batch_name)
                shutil.move(old_path, new_path)

        results = aggregate_data_batch(
            self.data_dir,
            by="cluster_size_norm",
            norm_order=2.0,
            num_hist_bins=5,
            verbose=False,
        )

        assert len(results) == 2
        for subdir_name in subdirs:
            assert not results[subdir_name].empty
            assert results[subdir_name].index.name == "cluster_size_norm"

    def test_aggregate_data_batch_with_value_range(self):
        """Test aggregate_data_batch with user-specified value range."""
        # Create test subdirectory structure
        subdirs = ["combo_1", "combo_2"]
        self.create_test_subdir_structure(self.data_dir, subdirs)

        results = aggregate_data_batch(
            self.data_dir,
            by="pred_llr",
            value_range=(-5.0, 5.0),
            num_hist_bins=5,
            verbose=False,
        )

        assert len(results) == 2
        for subdir_name in subdirs:
            assert not results[subdir_name].empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
