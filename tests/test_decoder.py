import unittest

import numpy as np
import stim
from scipy.sparse import csc_matrix

from src.ldpc_post_selection.decoder import (
    SoftOutputsBpLsdDecoder,
    SoftOutputsDecoder,
    SoftOutputsMatchingDecoder,
)
from src.ldpc_post_selection.stim_tools import dem_to_parity_check


class TestSoftOutputsDecoderBase(unittest.TestCase):
    def setUp(self):
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01,
            after_reset_flip_probability=0.01,
        )
        self.dem = self.circuit.detector_error_model()
        self.H, self.obs_matrix, self.p = dem_to_parity_check(self.dem)
        self.num_detectors = self.H.shape[0]
        self.num_bits = self.H.shape[1]
        self.num_observables = self.obs_matrix.shape[0]

        # Generate a random syndrome
        self.syndrome = np.random.randint(0, 2, size=self.num_detectors, dtype=bool)
        self.syndromes_batch = np.random.randint(
            0, 2, size=(5, self.num_detectors), dtype=bool
        )


class TestSoftOutputsDecoder(TestSoftOutputsDecoderBase):
    def test_initialization_H_p_obs(self):
        decoder = SoftOutputsDecoder(H=self.H, p=self.p, obs_matrix=self.obs_matrix)
        self.assertIsInstance(decoder.H, csc_matrix)
        self.assertEqual(decoder.H.shape, self.H.shape)
        self.assertTrue(np.array_equal(decoder.p, self.p))
        self.assertIsInstance(decoder.obs_matrix, csc_matrix)
        self.assertEqual(decoder.obs_matrix.shape, self.obs_matrix.shape)
        self.assertIsNone(decoder.circuit)

    def test_initialization_circuit(self):
        decoder = SoftOutputsDecoder(circuit=self.circuit)
        self.assertIsInstance(decoder.H, csc_matrix)
        self.assertEqual(decoder.H.shape, (self.num_detectors, self.num_bits))
        self.assertIsInstance(decoder.p, np.ndarray)
        self.assertEqual(len(decoder.p), self.num_bits)
        self.assertIsInstance(decoder.obs_matrix, csc_matrix)
        self.assertEqual(
            decoder.obs_matrix.shape, (self.num_observables, self.num_bits)
        )
        self.assertIsNotNone(decoder.circuit)
        self.assertFalse(decoder.decompose_errors)

    def test_initialization_circuit_decompose_errors(self):
        decoder = SoftOutputsDecoder(circuit=self.circuit, decompose_errors=True)
        self.assertTrue(decoder.decompose_errors)
        dem_decomposed = self.circuit.detector_error_model(decompose_errors=True)
        H_decomposed, _, _ = dem_to_parity_check(dem_decomposed)
        self.assertEqual(decoder.H.shape[1], H_decomposed.shape[1])

    def test_decode_not_implemented(self):
        decoder = SoftOutputsDecoder(circuit=self.circuit)
        with self.assertRaises(NotImplementedError):  # Base class decode is a pass
            decoder.decode(self.syndrome)


class TestSoftOutputsBpLsdDecoder(TestSoftOutputsDecoderBase):
    def test_initialization(self):
        decoder = SoftOutputsBpLsdDecoder(circuit=self.circuit)
        self.assertIsNotNone(decoder._bplsd)
        self.assertFalse(decoder.decompose_errors)

    def test_decode_single_sample(self):
        decoder = SoftOutputsBpLsdDecoder(circuit=self.circuit)
        pred, pred_bp, converge, soft_outputs = decoder.decode(self.syndrome)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.dtype, bool)
        self.assertEqual(pred.shape, (self.num_bits,))

        self.assertIsInstance(pred_bp, np.ndarray)
        self.assertEqual(pred_bp.dtype, bool)
        self.assertEqual(pred_bp.shape, (self.num_bits,))

        self.assertIsInstance(converge, bool)

        self.assertIsInstance(soft_outputs, dict)
        self.assertIn("pred_llr", soft_outputs)
        self.assertIsInstance(soft_outputs["pred_llr"], float)
        self.assertFalse(np.isnan(soft_outputs["pred_llr"]))

        self.assertIn("detector_density", soft_outputs)
        self.assertIsInstance(soft_outputs["detector_density"], float)
        self.assertFalse(np.isnan(soft_outputs["detector_density"]))
        self.assertTrue(0 <= soft_outputs["detector_density"] <= 1)

        self.assertIn("cluster_sizes", soft_outputs)
        self.assertIsInstance(soft_outputs["cluster_sizes"], np.ndarray)
        self.assertTrue(np.issubdtype(soft_outputs["cluster_sizes"].dtype, np.integer))
        self.assertFalse(np.any(np.isnan(soft_outputs["cluster_sizes"])))

        self.assertIn("cluster_llrs", soft_outputs)
        self.assertIsInstance(soft_outputs["cluster_llrs"], np.ndarray)
        self.assertTrue(np.issubdtype(soft_outputs["cluster_llrs"].dtype, np.floating))
        self.assertFalse(np.any(np.isnan(soft_outputs["cluster_llrs"])))

        self.assertEqual(
            len(soft_outputs["cluster_sizes"]), len(soft_outputs["cluster_llrs"])
        )


class TestSoftOutputsMatchingDecoder(TestSoftOutputsDecoderBase):
    def test_initialization(self):
        decoder = SoftOutputsMatchingDecoder(circuit=self.circuit)
        self.assertIsNotNone(decoder._matching)
        self.assertTrue(decoder.decompose_errors)
        self.assertGreaterEqual(decoder.obs_matrix.shape[0], 1)

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

        with self.assertRaisesRegex(
            ValueError, "SoftOutputsMatchingDecoder requires at least one observable"
        ):
            SoftOutputsMatchingDecoder(H=H_simple, p=p_simple, obs_matrix=obs_zero)

        circuit_0_obs = stim.Circuit.generated(
            "repetition_code:memory", distance=3, rounds=3
        )
        dem_0_obs = circuit_0_obs.detector_error_model(decompose_errors=True)
        _, obs_m_0, _ = dem_to_parity_check(dem_0_obs)
        if obs_m_0 is None or obs_m_0.shape[0] == 0:
            with self.assertRaisesRegex(
                ValueError,
                "SoftOutputsMatchingDecoder requires at least one observable",
            ):
                SoftOutputsMatchingDecoder(circuit=circuit_0_obs)

    def test_decode_single_sample(self):
        decoder = SoftOutputsMatchingDecoder(circuit=self.circuit)
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
                H=decoder.H, p=decoder.p, obs_matrix=csc_matrix(new_obs_matrix_arr)
            )

        syndrome_for_test = np.random.randint(0, 2, size=decoder.H.shape[0], dtype=bool)

        pred, soft_outputs = decoder.decode(syndrome_for_test)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.dtype, bool)
        self.assertEqual(pred.shape, (decoder.H.shape[1],))

        self.assertIsInstance(soft_outputs, dict)
        self.assertIn("pred_llr", soft_outputs)
        self.assertIsInstance(soft_outputs["pred_llr"], float)
        self.assertFalse(np.isnan(soft_outputs["pred_llr"]))

        self.assertIn("detector_density", soft_outputs)
        self.assertIsInstance(soft_outputs["detector_density"], float)
        self.assertFalse(np.isnan(soft_outputs["detector_density"]))
        self.assertTrue(0 <= soft_outputs["detector_density"] <= 1)

        self.assertIn("gap", soft_outputs)
        self.assertIsInstance(soft_outputs["gap"], float)
        self.assertFalse(np.isnan(soft_outputs["gap"]))

    def test_decode_batch(self):
        decoder = SoftOutputsMatchingDecoder(circuit=self.circuit)
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
                H=decoder.H, p=decoder.p, obs_matrix=csc_matrix(new_obs_matrix_arr)
            )

        syndromes_batch_current = np.random.randint(
            0, 2, size=(5, num_detectors_current), dtype=bool
        )

        num_samples = syndromes_batch_current.shape[0]
        preds, soft_outputs = decoder.decode_batch(syndromes_batch_current)

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.dtype, bool)
        self.assertEqual(preds.shape, (num_samples, decoder.H.shape[1]))

        self.assertIsInstance(soft_outputs, dict)

        self.assertIn("pred_llr", soft_outputs)
        self.assertIsInstance(soft_outputs["pred_llr"], np.ndarray)
        self.assertEqual(soft_outputs["pred_llr"].dtype, float)
        self.assertEqual(soft_outputs["pred_llr"].shape, (num_samples,))
        self.assertFalse(np.any(np.isnan(soft_outputs["pred_llr"])))

        self.assertIn("detector_density", soft_outputs)
        self.assertIsInstance(soft_outputs["detector_density"], np.ndarray)
        self.assertEqual(soft_outputs["detector_density"].dtype, float)
        self.assertEqual(soft_outputs["detector_density"].shape, (num_samples,))
        self.assertFalse(np.any(np.isnan(soft_outputs["detector_density"])))
        self.assertTrue(
            np.all(
                (0 <= soft_outputs["detector_density"])
                & (soft_outputs["detector_density"] <= 1)
            )
        )

        self.assertIn("gap", soft_outputs)
        self.assertIsInstance(soft_outputs["gap"], np.ndarray)
        self.assertEqual(soft_outputs["gap"].dtype, float)
        self.assertEqual(soft_outputs["gap"].shape, (num_samples,))
        self.assertFalse(np.any(np.isnan(soft_outputs["gap"])))

    def test_decode_batch_gap_with_one_observable(self):
        decoder = SoftOutputsMatchingDecoder(circuit=self.circuit)
        num_detectors_current = decoder.H.shape[0]
        syndromes_for_test = np.random.randint(
            0, 2, size=(5, num_detectors_current), dtype=bool
        )

        self.assertEqual(decoder.obs_matrix.shape[0], 1)
        num_obs_patterns = 2 ** decoder.obs_matrix.shape[0]
        self.assertEqual(num_obs_patterns, 2)

        preds, soft_outputs = decoder.decode_batch(syndromes_for_test)
        self.assertFalse(np.any(np.isnan(soft_outputs["gap"])))


if __name__ == "__main__":
    unittest.main()
