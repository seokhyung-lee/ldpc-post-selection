import numpy as np
import pytest
import stim
from scipy.sparse import csc_matrix

from ldpc_post_selection.decoder import (
    SoftOutputsBpLsdDecoder,
    SoftOutputsDecoder,
    SoftOutputsMatchingDecoder,
)
from ldpc_post_selection.stim_tools import dem_to_parity_check


@pytest.fixture
def circuit_data():
    """Fixture providing circuit and related data for tests."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    dem = circuit.detector_error_model()
    H, obs_matrix, p = dem_to_parity_check(dem)
    num_detectors = H.shape[0]
    num_bits = H.shape[1]
    num_observables = obs_matrix.shape[0]

    # Generate random syndromes
    syndrome = np.random.randint(0, 2, size=num_detectors, dtype=bool)
    syndromes_batch = np.random.randint(0, 2, size=(5, num_detectors), dtype=bool)

    return {
        "circuit": circuit,
        "dem": dem,
        "H": H,
        "obs_matrix": obs_matrix,
        "p": p,
        "num_detectors": num_detectors,
        "num_bits": num_bits,
        "num_observables": num_observables,
        "syndrome": syndrome,
        "syndromes_batch": syndromes_batch,
    }


class TestSoftOutputsDecoder:
    def test_initialization_H_p_obs(self, circuit_data):
        decoder = SoftOutputsDecoder(
            H=circuit_data["H"],
            p=circuit_data["p"],
            obs_matrix=circuit_data["obs_matrix"],
        )
        assert isinstance(decoder.H, csc_matrix)
        assert decoder.H.shape == circuit_data["H"].shape
        assert np.array_equal(decoder.priors, circuit_data["p"])
        assert isinstance(decoder.obs_matrix, csc_matrix)
        assert decoder.obs_matrix.shape == circuit_data["obs_matrix"].shape
        assert decoder.circuit is None

    def test_initialization_circuit(self, circuit_data):
        decoder = SoftOutputsDecoder(circuit=circuit_data["circuit"])
        assert isinstance(decoder.H, csc_matrix)
        assert decoder.H.shape == (
            circuit_data["num_detectors"],
            circuit_data["num_bits"],
        )
        assert isinstance(decoder.priors, np.ndarray)
        assert len(decoder.priors) == circuit_data["num_bits"]
        assert isinstance(decoder.obs_matrix, csc_matrix)
        assert decoder.obs_matrix.shape == (
            circuit_data["num_observables"],
            circuit_data["num_bits"],
        )
        assert decoder.circuit is not None
        assert not decoder.decompose_errors

    def test_initialization_circuit_decompose_errors(self, circuit_data):
        decoder = SoftOutputsDecoder(
            circuit=circuit_data["circuit"], decompose_errors=True
        )
        assert decoder.decompose_errors
        dem_decomposed = circuit_data["circuit"].detector_error_model(
            decompose_errors=True
        )
        H_decomposed, _, _ = dem_to_parity_check(dem_decomposed)
        assert decoder.H.shape[1] == H_decomposed.shape[1]

    def test_decode_not_implemented(self, circuit_data):
        decoder = SoftOutputsDecoder(circuit=circuit_data["circuit"])
        with pytest.raises(NotImplementedError):
            decoder.decode(circuit_data["syndrome"])


class TestSoftOutputsBpLsdDecoder:
    def test_initialization(self, circuit_data):
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit_data["circuit"])
        assert decoder._bplsd is not None
        assert not decoder.decompose_errors

    def test_decode_single_sample(self, circuit_data):
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit_data["circuit"])
        pred, pred_bp, converge, soft_outputs = decoder.decode(circuit_data["syndrome"])

        assert isinstance(pred, np.ndarray)
        assert pred.dtype == bool
        assert pred.shape == (circuit_data["num_bits"],)

        assert isinstance(pred_bp, np.ndarray)
        assert pred_bp.dtype == bool
        assert pred_bp.shape == (circuit_data["num_bits"],)

        assert isinstance(converge, bool)

        assert isinstance(soft_outputs, dict)
        assert "pred_llr" in soft_outputs
        assert isinstance(soft_outputs["pred_llr"], float)
        assert not np.isnan(soft_outputs["pred_llr"])

        assert "detector_density" in soft_outputs
        assert isinstance(soft_outputs["detector_density"], float)
        assert not np.isnan(soft_outputs["detector_density"])
        assert 0 <= soft_outputs["detector_density"] <= 1

        assert "cluster_sizes" in soft_outputs
        assert isinstance(soft_outputs["cluster_sizes"], np.ndarray)
        assert np.issubdtype(soft_outputs["cluster_sizes"].dtype, np.integer)
        assert not np.any(np.isnan(soft_outputs["cluster_sizes"]))

        assert "cluster_llrs" in soft_outputs
        assert isinstance(soft_outputs["cluster_llrs"], np.ndarray)
        assert np.issubdtype(soft_outputs["cluster_llrs"].dtype, np.floating)
        assert not np.any(np.isnan(soft_outputs["cluster_llrs"]))

        assert len(soft_outputs["cluster_sizes"]) == len(soft_outputs["cluster_llrs"])

    def test_logical_gap_proxy_computation(self, circuit_data):
        """Test logical gap proxy computation functionality."""
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit_data["circuit"])

        # Test with gap proxy computation enabled
        pred, pred_bp, converge, soft_outputs = decoder.decode(
            circuit_data["syndrome"],
            compute_logical_gap_proxy=True,
            explore_only_nearby_logical_classes=True,
        )

        # Check that gap_proxy is included in soft outputs
        assert "gap_proxy" in soft_outputs
        assert isinstance(soft_outputs["gap_proxy"], float)
        assert not np.isnan(soft_outputs["gap_proxy"])

        # Test with explore_only_nearby_logical_classes=False
        pred2, pred_bp2, converge2, soft_outputs2 = decoder.decode(
            circuit_data["syndrome"],
            compute_logical_gap_proxy=True,
            explore_only_nearby_logical_classes=False,
        )

        assert "gap_proxy" in soft_outputs2
        assert isinstance(soft_outputs2["gap_proxy"], float)
        assert not np.isnan(soft_outputs2["gap_proxy"])

    def test_logical_gap_proxy_disabled(self, circuit_data):
        """Test that gap_proxy is not computed when disabled."""
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit_data["circuit"])

        pred, pred_bp, converge, soft_outputs = decoder.decode(
            circuit_data["syndrome"], compute_logical_gap_proxy=False
        )

        # gap_proxy should not be in soft_outputs when disabled
        assert "gap_proxy" not in soft_outputs

    def test_logical_gap_proxy_no_observables(self):
        """Test logical gap proxy when there are no observables."""
        # Create a simple case with no observables
        H_simple = csc_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=bool))
        p_simple = np.array([0.1, 0.1, 0.1])

        decoder = SoftOutputsBpLsdDecoder(H=H_simple, p=p_simple, obs_matrix=None)
        syndrome_simple = np.array([True, False], dtype=bool)

        pred, pred_bp, converge, soft_outputs = decoder.decode(
            syndrome_simple, compute_logical_gap_proxy=True
        )

        # Should return 0.0 when no observables
        assert "gap_proxy" in soft_outputs
        assert soft_outputs["gap_proxy"] == 0.0

    def test_logical_gap_proxy_helper_methods(self, circuit_data):
        """Test the helper methods for logical gap proxy computation."""
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit_data["circuit"])

        # Test _get_logical_classes_to_explore
        predicted_logical_class = np.array([True, False], dtype=bool)

        # Test nearby classes only
        nearby_classes = decoder._get_logical_classes_to_explore(
            predicted_logical_class, explore_only_nearby_logical_classes=True
        )
        assert (
            len(nearby_classes) == 2
        )  # Should have 2 nearby classes for 2 observables

        # Test all classes
        all_classes = decoder._get_logical_classes_to_explore(
            predicted_logical_class, explore_only_nearby_logical_classes=False
        )
        assert len(all_classes) == 3  # 2^2 - 1 = 3 (excluding predicted class)

    def test_compute_cluster_stats_false(self, circuit_data):
        """Test that cluster stats are not computed when disabled."""
        decoder = SoftOutputsBpLsdDecoder(circuit=circuit_data["circuit"])

        pred, pred_bp, converge, soft_outputs = decoder.decode(
            circuit_data["syndrome"], include_cluster_stats=False
        )

        # Cluster-related outputs should not be present
        assert "clusters" not in soft_outputs
        assert "cluster_sizes" not in soft_outputs
        assert "cluster_llrs" not in soft_outputs


class TestSoftOutputsMatchingDecoder:
    def test_initialization(self, circuit_data):
        decoder = SoftOutputsMatchingDecoder(circuit=circuit_data["circuit"])
        assert decoder._matching is not None
        assert decoder.decompose_errors
        assert decoder.obs_matrix.shape[0] >= 1

    def test_initialization_raises_error_no_observables(self):
        rep_circuit = stim.Circuit()
        rep_circuit.append_operation("R", [0, 1, 2])
        rep_circuit.append_operation("MR", [0, 1, 2])
        rep_circuit.append_operation(
            "DETECTOR", [stim.target_rec(-2), stim.target_rec(-3)]
        )
        rep_circuit.append_operation(
            "DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)]
        )
        H_simple = csc_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=bool))
        p_simple = np.array([0.1, 0.1, 0.1])
        obs_zero = csc_matrix(np.empty((0, H_simple.shape[1]), dtype=bool))

        with pytest.raises(
            ValueError,
            match="SoftOutputsMatchingDecoder requires at least one observable",
        ):
            SoftOutputsMatchingDecoder(H=H_simple, p=p_simple, obs_matrix=obs_zero)

        circuit_0_obs = stim.Circuit.generated(
            "repetition_code:memory", distance=3, rounds=3
        )
        dem_0_obs = circuit_0_obs.detector_error_model(decompose_errors=True)
        _, obs_m_0, _ = dem_to_parity_check(dem_0_obs)
        if obs_m_0 is None or obs_m_0.shape[0] == 0:
            with pytest.raises(
                ValueError,
                match="SoftOutputsMatchingDecoder requires at least one observable",
            ):
                SoftOutputsMatchingDecoder(circuit=circuit_0_obs)

    def test_decode_single_sample(self, circuit_data):
        decoder = SoftOutputsMatchingDecoder(circuit=circuit_data["circuit"])
        current_num_bits = decoder.H.shape[1]

        if decoder.obs_matrix.shape[0] < 2:
            original_obs_matrix = decoder.obs_matrix.toarray()
            num_dummy_obs_to_add = 2 - original_obs_matrix.shape[0]
            dummy_obs = np.zeros((num_dummy_obs_to_add, current_num_bits), dtype=bool)
            for i in range(num_dummy_obs_to_add):
                dummy_obs[i, i % current_num_bits] = True
            if original_obs_matrix.shape[0] > 0:
                new_obs_matrix_arr = np.vstack([original_obs_matrix, dummy_obs])
            else:
                new_obs_matrix_arr = dummy_obs

            decoder = SoftOutputsMatchingDecoder(
                H=decoder.H, p=decoder.priors, obs_matrix=csc_matrix(new_obs_matrix_arr)
            )

        syndrome_for_test = np.random.randint(0, 2, size=decoder.H.shape[0], dtype=bool)

        pred, soft_outputs = decoder.decode(syndrome_for_test)

        assert isinstance(pred, np.ndarray)
        assert pred.dtype == bool
        assert pred.shape == (decoder.H.shape[1],)

        assert isinstance(soft_outputs, dict)
        assert "pred_llr" in soft_outputs
        assert isinstance(soft_outputs["pred_llr"], float)
        assert not np.isnan(soft_outputs["pred_llr"])

        assert "detector_density" in soft_outputs
        assert isinstance(soft_outputs["detector_density"], float)
        assert not np.isnan(soft_outputs["detector_density"])
        assert 0 <= soft_outputs["detector_density"] <= 1

        assert "gap" in soft_outputs
        assert isinstance(soft_outputs["gap"], float)
        assert not np.isnan(soft_outputs["gap"])

    def test_decode_batch(self, circuit_data):
        decoder = SoftOutputsMatchingDecoder(circuit=circuit_data["circuit"])
        current_num_bits = decoder.H.shape[1]
        num_detectors_current = decoder.H.shape[0]

        if decoder.obs_matrix.shape[0] < 2:
            original_obs_matrix = decoder.obs_matrix.toarray()
            num_dummy_obs_to_add = 2 - original_obs_matrix.shape[0]
            dummy_obs = np.zeros((num_dummy_obs_to_add, current_num_bits), dtype=bool)
            for i in range(num_dummy_obs_to_add):
                dummy_obs[i, i % current_num_bits] = True
            if original_obs_matrix.shape[0] > 0:
                new_obs_matrix_arr = np.vstack([original_obs_matrix, dummy_obs])
            else:
                new_obs_matrix_arr = dummy_obs
            decoder = SoftOutputsMatchingDecoder(
                H=decoder.H, p=decoder.priors, obs_matrix=csc_matrix(new_obs_matrix_arr)
            )

        syndromes_batch_current = np.random.randint(
            0, 2, size=(5, num_detectors_current), dtype=bool
        )

        num_samples = syndromes_batch_current.shape[0]
        preds, soft_outputs = decoder.decode_batch(syndromes_batch_current)

        assert isinstance(preds, np.ndarray)
        assert preds.dtype == bool
        assert preds.shape == (num_samples, decoder.H.shape[1])

        assert isinstance(soft_outputs, dict)

        assert "pred_llr" in soft_outputs
        assert isinstance(soft_outputs["pred_llr"], np.ndarray)
        assert soft_outputs["pred_llr"].dtype == float
        assert soft_outputs["pred_llr"].shape == (num_samples,)
        assert not np.any(np.isnan(soft_outputs["pred_llr"]))

        assert "detector_density" in soft_outputs
        assert isinstance(soft_outputs["detector_density"], np.ndarray)
        assert soft_outputs["detector_density"].dtype == float
        assert soft_outputs["detector_density"].shape == (num_samples,)
        assert not np.any(np.isnan(soft_outputs["detector_density"]))
        assert np.all(
            (0 <= soft_outputs["detector_density"])
            & (soft_outputs["detector_density"] <= 1)
        )

        assert "gap" in soft_outputs
        assert isinstance(soft_outputs["gap"], np.ndarray)
        assert soft_outputs["gap"].dtype == float
        assert soft_outputs["gap"].shape == (num_samples,)
        assert not np.any(np.isnan(soft_outputs["gap"]))

    def test_decode_batch_gap_with_one_observable(self, circuit_data):
        decoder = SoftOutputsMatchingDecoder(circuit=circuit_data["circuit"])
        num_detectors_current = decoder.H.shape[0]
        syndromes_for_test = np.random.randint(
            0, 2, size=(5, num_detectors_current), dtype=bool
        )

        assert decoder.obs_matrix.shape[0] == 1
        num_obs_patterns = 2 ** decoder.obs_matrix.shape[0]
        assert num_obs_patterns == 2

        preds, soft_outputs = decoder.decode_batch(syndromes_for_test)
        assert not np.any(np.isnan(soft_outputs["gap"]))


if __name__ == "__main__":
    pytest.main([__file__])
