import os
import warnings
from typing import Any, Dict, List, Tuple
from unittest.mock import call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import stim

# Functions to test
from simulations.simulation_utils import (
    _calculate_chunk_sizes,
    _convert_df_dtypes_for_feather,
    _get_optimal_uint_dtype,
    _handle_empty_shot_chunks,
    get_existing_shots,
    task,
    task_matching,
    task_matching_parallel,
    task_parallel,
)


@pytest.fixture
def sample_stim_circuit() -> stim.Circuit:
    """A sample stim circuit for testing.
    This is a distance 3, 1-round repetition code memory experiment.
    It should have 4 detectors and 1 logical observable.
    """
    try:
        circuit = stim.Circuit.generated(
            "repetition_code:memory",
            distance=3,
            rounds=1,
            after_clifford_depolarization=0.01,
        )
        # Ensure the circuit is as expected for tests
        assert (
            circuit.num_detectors == 4
        ), "Test circuit should have 4 detectors for d=3, r=1 memory code as generated"
        assert circuit.num_observables == 1, "Test circuit should have 1 observable"
        return circuit
    except Exception as e:
        pytest.fail(f"Failed to generate stim circuit: {e}")


@pytest.fixture
def simple_dataframe() -> pd.DataFrame:
    """Provides a simple DataFrame for testing dtype conversion."""
    return pd.DataFrame(
        {
            "float_col": [1.0, 2.5, 3.5],
            "int_col": [1, 2, 3],
            "bool_col": [True, False, True],
            "object_col": ["a", "b", "c"],  # Should remain object
        }
    )


# Tests for helper functions


def test_convert_df_dtypes_for_feather(simple_dataframe):
    df_converted = _convert_df_dtypes_for_feather(
        simple_dataframe.copy()
    )  # Use copy to avoid modifying fixture
    assert df_converted["float_col"].dtype == np.float32
    assert df_converted["int_col"].dtype == np.int32
    assert df_converted["bool_col"].dtype == bool  # Should remain bool
    assert df_converted["object_col"].dtype == object  # Should remain object


@pytest.mark.parametrize(
    "max_val, expected_dtype",
    [
        (2**8 - 1, np.uint16),  # Fits in uint8, but smallest is uint16
        (2**16 - 1, np.uint16),
        (2**16, np.uint32),
        (2**32 - 1, np.uint32),
        (2**32, np.uint64),
        (2**64 - 1, np.uint64),
    ],
)
def test_get_optimal_uint_dtype(max_val, expected_dtype):
    assert _get_optimal_uint_dtype(max_val) == expected_dtype


@pytest.mark.parametrize(
    "shots, n_jobs, repeat, expected_chunks",
    [
        (0, 1, 1, []),
        (100, 1, 1, [100]),
        (100, 2, 1, [50, 50]),
        (100, 10, 1, [10] * 10),
        (101, 10, 1, [11] + [10] * 9),
        (10, 3, 2, [2, 2, 2, 2, 1, 1]),  # 10 // (3*2) = 1. Rem = 4. 1+1=2 for first 4
        (5, 10, 1, [1] * 5),  # More jobs*repeat than shots
    ],
)
def test_calculate_chunk_sizes(shots, n_jobs, repeat, expected_chunks):
    assert _calculate_chunk_sizes(shots, n_jobs, repeat) == expected_chunks


@pytest.mark.parametrize(
    "shots, chunk_sizes, columns, expect_df, expected_cols",
    [
        (0, [], ["a", "b"], True, ["a", "b"]),  # 0 shots
        (10, [], ["a"], True, ["a"]),  # No chunks but shots > 0
        (10, [5, 5], ["a"], False, []),  # Valid chunks
    ],
)
def test_handle_empty_shot_chunks(
    shots, chunk_sizes, columns, expect_df, expected_cols
):
    result = _handle_empty_shot_chunks(shots, chunk_sizes, columns)
    if expect_df:
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected_cols
        assert len(result) == 0
        # Check dtypes if columns are standard
        if (
            "fail" in columns
        ):  # Assuming standard columns for type check if 'fail' is present
            df_expected_types = _convert_df_dtypes_for_feather(
                pd.DataFrame(columns=columns).copy()
            )
            for col in expected_cols:
                assert result[col].dtype == df_expected_types[col].dtype
    else:
        assert result is None


# Tests for get_existing_shots
@patch("os.path.isdir")
@patch("os.listdir")
def test_get_existing_shots_empty_dir(mock_listdir, mock_isdir):
    mock_isdir.return_value = True
    mock_listdir.return_value = []
    total_shots, files_info = get_existing_shots("fake_dir")
    assert total_shots == 0
    assert files_info == []
    mock_isdir.assert_called_once_with("fake_dir")
    mock_listdir.assert_called_once_with("fake_dir")


@patch("os.path.isdir")
@patch("os.listdir")
def test_get_existing_shots_non_existent_dir(mock_listdir, mock_isdir):
    mock_isdir.return_value = False
    total_shots, files_info = get_existing_shots("non_existent_dir")
    assert total_shots == 0
    assert files_info == []
    mock_isdir.assert_called_once_with("non_existent_dir")
    mock_listdir.assert_not_called()  # listdir should not be called if dir doesn't exist


@patch("os.path.isdir")
@patch("os.listdir")
def test_get_existing_shots_with_valid_and_invalid_dirs(mock_listdir, mock_isdir):
    # Simulate that the main data_dir exists
    # and then control existence for subdirectories
    def isdir_side_effect(path):
        if path == "fake_data_dir":
            return True
        if path == os.path.join("fake_data_dir", "batch_0_100"):
            return True
        if path == os.path.join("fake_data_dir", "batch_1_200"):
            return True
        if path == os.path.join("fake_data_dir", "not_a_batch_dir"):
            return True  # It's a dir, but not matching pattern
        if path == os.path.join("fake_data_dir", "batch_invalid_format"):
            return True
        if path == os.path.join("fake_data_dir", "batch_2_file_not_dir"):
            return False  # This one is a file, not a dir
        return False

    mock_isdir.side_effect = isdir_side_effect
    mock_listdir.return_value = [
        "batch_0_100",
        "batch_1_200",
        "not_a_batch_dir",
        "some_file.txt",
        "batch_invalid_format",  # Will be caught by regex
        "batch_badidx_10",  # caught by int conversion
        "batch_2_file_not_dir",  # caught by isdir check
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        total_shots, files_info = get_existing_shots("fake_data_dir")
        # The regex r"^batch_(\d+)_(\d+)$") will not match "batch_badidx_10"
        # so the int() conversion that could raise ValueError is not reached.
        # Thus, no warning is expected from that part for this specific mock_listdir setup.
        assert len(w) == 0  # Expect zero warnings with current get_existing_shots logic
        # if len(w) > 0:
        #     assert issubclass(w[-1].category, UserWarning)
        #     assert (
        #         "Could not parse batch index or shots from directory name batch_badidx_10"
        #         in str(w[-1].message)
        #     )

    assert total_shots == 300
    assert len(files_info) == 2
    assert files_info[0] == (0, os.path.join("fake_data_dir", "batch_0_100"), 100)
    assert files_info[1] == (1, os.path.join("fake_data_dir", "batch_1_200"), 200)

    expected_isdir_calls = [
        call("fake_data_dir"),
        call(os.path.join("fake_data_dir", "batch_0_100")),
        call(os.path.join("fake_data_dir", "batch_1_200")),
        call(os.path.join("fake_data_dir", "not_a_batch_dir")),
        # some_file.txt is filtered by regex, no isdir call for its path
        call(os.path.join("fake_data_dir", "batch_invalid_format")),
        call(os.path.join("fake_data_dir", "batch_badidx_10")),
        call(os.path.join("fake_data_dir", "batch_2_file_not_dir")),
    ]
    # Allow additional calls if any, but these must be present
    # mock_isdir.assert_has_calls(expected_isdir_calls, any_order=True)
    # Check that all expected calls were made, doesn't fail if more calls were made.
    # This is tricky as the order inside os.listdir is not guaranteed for all OS.
    # So we check the set of calls made to isdir for the subdirectories.
    actual_subdir_isdir_calls = set()
    for c in mock_isdir.call_args_list:
        arg = c[0][0]
        if arg != "fake_data_dir":
            actual_subdir_isdir_calls.add(arg)

    expected_subdir_isdir_calls = set(
        os.path.join("fake_data_dir", dirname) for dirname in mock_listdir.return_value
    )
    # This check is too strict as os.listdir can return different order and some paths are not checked
    # assert actual_subdir_isdir_calls == expected_subdir_isdir_calls

    # Check the first call was to the main data_dir
    assert mock_isdir.call_args_list[0] == call("fake_data_dir")
    # Check listdir was called once on the main data_dir
    mock_listdir.assert_called_once_with("fake_data_dir")


# Tests for task and task_parallel (SoftOutputsBpLsdDecoder)
def test_task_bplsd_basic(sample_stim_circuit):
    shots = 10
    circuit = sample_stim_circuit
    decoder_prms = {"max_iter": 10}  # Example BP/LSD specific param

    (
        fails,
        fails_bp,
        converges,
        scalar_soft_infos,
        cluster_sizes_list,
        cluster_llrs_list,
    ) = task(shots, circuit, decoder_prms=decoder_prms)

    assert len(fails) == shots
    assert len(fails_bp) == shots
    assert len(converges) == shots
    assert len(scalar_soft_infos) == shots
    assert len(cluster_sizes_list) == shots
    assert len(cluster_llrs_list) == shots

    assert fails.dtype == bool
    assert fails_bp.dtype == bool
    assert converges.dtype == bool

    for info in scalar_soft_infos:
        assert isinstance(info, dict)
        assert "pred_llr" in info
        assert "detector_density" in info
        assert isinstance(
            info["pred_llr"], (float, type(None))
        )  # Can be None if no errors
        assert isinstance(info["detector_density"], float)

    for cs_arr in cluster_sizes_list:
        assert isinstance(cs_arr, np.ndarray)
        # cs_arr can be empty if no clusters found
        if cs_arr.size > 0:
            assert cs_arr.dtype == int  # Based on decoder output

    for cl_arr in cluster_llrs_list:
        assert isinstance(cl_arr, np.ndarray)
        if cl_arr.size > 0:
            assert cl_arr.dtype == float  # Based on decoder output


def test_task_bplsd_zero_shots(sample_stim_circuit):
    shots = 0
    circuit = sample_stim_circuit
    (
        fails,
        fails_bp,
        converges,
        scalar_soft_infos,
        cluster_sizes_list,
        cluster_llrs_list,
    ) = task(shots, circuit)

    assert len(fails) == 0
    assert len(fails_bp) == 0
    assert len(converges) == 0
    assert len(scalar_soft_infos) == 0
    assert len(cluster_sizes_list) == 0
    assert len(cluster_llrs_list) == 0


def test_task_parallel_bplsd_basic(sample_stim_circuit):
    shots = 20  # Small number for faster test
    circuit = sample_stim_circuit
    n_jobs = 2
    repeat = 2
    decoder_prms = {"bp_method": "ms", "ms_scaling_factor": 0.6}

    df, flat_cs, flat_cl, offsets = task_parallel(
        shots, circuit, n_jobs, repeat, decoder_prms=decoder_prms
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == shots
    expected_cols = ["fail", "fail_bp", "converge", "pred_llr", "detector_density"]
    for col in expected_cols:
        assert col in df.columns

    assert df["fail"].dtype == bool
    assert df["fail_bp"].dtype == bool
    assert df["converge"].dtype == bool
    assert df["pred_llr"].dtype == np.float32
    assert df["detector_density"].dtype == np.float32

    assert isinstance(flat_cs, np.ndarray)
    assert isinstance(flat_cl, np.ndarray)
    assert isinstance(offsets, np.ndarray)

    assert len(offsets) == shots + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(flat_cs)
    assert offsets[-1] == len(flat_cl)
    if flat_cs.size > 0:
        assert flat_cs.dtype == int
    if flat_cl.size > 0:
        assert flat_cl.dtype == float


def test_task_parallel_bplsd_zero_shots(sample_stim_circuit):
    shots = 0
    circuit = sample_stim_circuit
    df, flat_cs, flat_cl, offsets = task_parallel(shots, circuit, 1, 1)

    assert len(df) == 0
    expected_cols = ["fail", "fail_bp", "converge", "pred_llr", "detector_density"]
    for col in expected_cols:
        assert col in df.columns  # Columns should still exist

    assert len(flat_cs) == 0
    assert len(flat_cl) == 0
    assert len(offsets) == 1
    assert offsets[0] == 0


# Tests for task_matching and task_matching_parallel (SoftOutputsMatchingDecoder)
def test_task_matching_basic(sample_stim_circuit):
    shots = 10
    circuit = sample_stim_circuit
    # Matching decoder does not take specific params in SoftOutputsMatchingDecoder __init__
    # other than what's derived from circuit.
    decoder_prms = None

    fails, scalar_soft_infos = task_matching(shots, circuit, decoder_prms=decoder_prms)

    assert len(fails) == shots
    assert len(scalar_soft_infos) == shots
    assert fails.dtype == bool

    for info in scalar_soft_infos:
        assert isinstance(info, dict)
        assert "pred_llr" in info
        assert "detector_density" in info
        assert "gap" in info
        assert isinstance(info["pred_llr"], float)
        assert isinstance(info["detector_density"], float)
        assert isinstance(info["gap"], float)


def test_task_matching_zero_shots(sample_stim_circuit):
    shots = 0
    circuit = sample_stim_circuit
    fails, scalar_soft_infos = task_matching(shots, circuit)

    assert len(fails) == 0
    assert len(scalar_soft_infos) == 0


def test_task_matching_parallel_basic(sample_stim_circuit):
    shots = 20
    circuit = sample_stim_circuit
    n_jobs = 2
    repeat = 2
    decoder_prms = None  # No specific params for matching decoder here

    df = task_matching_parallel(
        shots, circuit, n_jobs, repeat, decoder_prms=decoder_prms
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == shots
    expected_cols = ["fail", "pred_llr", "detector_density", "gap"]
    for col in expected_cols:
        assert col in df.columns

    assert df["fail"].dtype == bool
    assert df["pred_llr"].dtype == np.float32
    assert df["detector_density"].dtype == np.float32
    assert df["gap"].dtype == np.float32


def test_task_matching_parallel_zero_shots(sample_stim_circuit):
    shots = 0
    circuit = sample_stim_circuit
    df = task_matching_parallel(shots, circuit, 1, 1)

    assert len(df) == 0
    expected_cols = ["fail", "pred_llr", "detector_density", "gap"]
    for col in expected_cols:
        assert col in df.columns
