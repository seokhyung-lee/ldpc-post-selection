from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim
from pymatching import Matching
from scipy.sparse import csc_matrix, vstack

from .base import SoftOutputsDecoder


class SoftOutputsMatchingDecoder(SoftOutputsDecoder):
    """
    PyMatching decoder with additional soft outputs for quantifying decoding confidence.
    """

    _matching: Matching

    def __init__(
        self,
        H: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        *,
        p: Optional[np.ndarray | List[float]] = None,
        obs_matrix: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        circuit: Optional[stim.Circuit] = None,
    ):
        """
        PyMatching decoder with additional soft outputs

        Parameters
        ----------
        H : 2D array-like of bool/int, including scipy csc matrix
            Parity check matrix. Internally stored as a scipy csc matrix of uint8.
        p : 1D array-like of float
            Error probabilities.
        obs_matrix : 2D array-like of bool/int, including scipy csc matrix
            Observable matrix. Internally stored as a scipy csc matrix of uint8.
        circuit : stim.Circuit, optional
            Circuit.
        """
        super().__init__(
            H=H, p=p, obs_matrix=obs_matrix, circuit=circuit, decompose_errors=True
        )

        if self.obs_matrix is None or self.obs_matrix.shape[0] == 0:
            raise ValueError(
                "SoftOutputsMatchingDecoder requires at least one observable. "
                "Please provide an obs_matrix or a circuit that defines at least one observable."
            )

        H_obss_as_dets = vstack([self.H, self.obs_matrix], format="csc", dtype="uint8")
        weights = np.log((1 - self.priors) / self.priors)
        self._matching = Matching(H_obss_as_dets, weights=weights)

    def decode(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode the detector measurement outcomes for a single sample.

        Parameters
        ----------
        detector_outcomes : 1D numpy array or list of bool/int
            Detector measurement outcomes for a single sample.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        pred : 1D numpy array of bool
            Predicted error pattern.
        soft_outputs : Dict[str, float]
            Dictionary of soft outputs:
            - "pred_llr" (float): LLR of the predicted error pattern.
            - "detector_density" (float): Fraction of violated detector outcomes.
            - "gap" (float): Difference between the LLRs of the best and the second best
                predictions in different logical classes.
        """
        if verbose:
            print("Starting PyMatching decoding...")

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        if verbose:
            print(f"Detector outcomes shape: {detector_outcomes.shape}")
            print(f"Number of violated detectors: {detector_outcomes.sum()}")

        all_obs_values = product([False, True], repeat=self.obs_matrix.shape[0])
        all_obs_values = np.array(list(all_obs_values), dtype=bool)

        if verbose:
            print(
                f"Number of observable patterns to explore: {all_obs_values.shape[0]}"
            )

        repeated_detector_outcomes = np.tile(
            detector_outcomes, (all_obs_values.shape[0], 1)
        )
        det_outcomes_with_obs = np.concatenate(
            [repeated_detector_outcomes, all_obs_values], axis=1
        )

        matching = self._matching

        if verbose:
            print("Running PyMatching decode_batch...")

        preds_all_obs, weights_all_obs = matching.decode_batch(
            det_outcomes_with_obs, return_weights=True
        )

        preds_all_obs = preds_all_obs.astype(bool)
        weights_all_obs = weights_all_obs.astype("float64")

        obs_inds_sorted = np.argsort(weights_all_obs)
        pred = preds_all_obs[obs_inds_sorted[0]]
        pred_llr = weights_all_obs[obs_inds_sorted[0]]

        # With __init__ ensuring at least 1 observable, num_obs_patterns (all_obs_values.shape[0]) >= 2.
        # So, obs_inds_sorted will always have at least two elements.
        gap = weights_all_obs[obs_inds_sorted[1]] - pred_llr

        soft_outputs = {
            "pred_llr": pred_llr,
            "detector_density": detector_outcomes.sum() / len(detector_outcomes),
            "gap": float(gap),
        }

        if verbose:
            print(f"Best prediction LLR: {pred_llr:.4f}")
            print(f"Gap (second best - best): {gap:.4f}")
            print(f"Predicted error weight: {pred.sum()}")
            print("PyMatching decoding completed!")

        return pred, soft_outputs

    def decode_batch(
        self,
        detector_outcomes: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode a batch of detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 2D numpy array of bool/int
            Detector measurement outcomes. Each row is a sample.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        preds : 2D numpy array of bool
            Predicted error patterns. Each row corresponds to a sample in detector_outcomes.
        soft_outputs : Dict[str, np.ndarray]
            See `decode` for details.
        """
        if verbose:
            print("Starting PyMatching batch decoding...")

        if detector_outcomes.ndim != 2:
            raise ValueError("Detector outcomes must be a 2D array for batch decoding.")

        num_samples, num_detectors = detector_outcomes.shape
        if num_detectors != self.H.shape[0]:
            raise ValueError(
                f"Number of detectors in outcomes ({num_detectors}) does not match "
                f"H matrix ({self.H.shape[0]})."
            )

        if verbose:
            print(f"Batch size: {num_samples}")
            print(f"Number of detectors: {num_detectors}")

        num_observables = self.obs_matrix.shape[0]
        all_obs_values = np.array(
            list(product([False, True], repeat=num_observables)), dtype=bool
        )
        num_obs_patterns = all_obs_values.shape[0]

        if verbose:
            print(f"Number of observable patterns per sample: {num_obs_patterns}")

        # Repeat each sample for all observable patterns
        repeated_detector_outcomes = np.repeat(
            detector_outcomes, num_obs_patterns, axis=0
        )

        # Tile all observable patterns for all samples
        tiled_obs_values = np.tile(all_obs_values, (num_samples, 1))

        # Combine detector outcomes with observable patterns
        det_outcomes_with_obs_batch = np.concatenate(
            [repeated_detector_outcomes, tiled_obs_values], axis=1
        )

        if verbose:
            print(
                f"Total batch size for decoding: {det_outcomes_with_obs_batch.shape[0]}"
            )
            print("Running PyMatching decode_batch...")

        matching = self._matching
        preds_all_obs_batch, weights_all_obs_batch = matching.decode_batch(
            det_outcomes_with_obs_batch, return_weights=True
        )

        preds_all_obs_batch = preds_all_obs_batch.astype(bool)
        weights_all_obs_batch = weights_all_obs_batch.astype("float64")

        if verbose:
            print("Processing batch results...")

        # Reshape to (num_samples, num_obs_patterns, ...)
        preds_all_obs_reshaped = preds_all_obs_batch.reshape(
            num_samples, num_obs_patterns, -1
        )
        weights_all_obs_reshaped = weights_all_obs_batch.reshape(
            num_samples, num_obs_patterns
        )

        # Find the best prediction for each sample
        obs_inds_sorted_batch = np.argsort(weights_all_obs_reshaped, axis=1)

        min_weight_indices = obs_inds_sorted_batch[:, 0]
        preds_batch = preds_all_obs_reshaped[
            np.arange(num_samples), min_weight_indices, :
        ]
        pred_llrs_batch = weights_all_obs_reshaped[
            np.arange(num_samples), min_weight_indices
        ]

        # Calculate gap
        # With __init__ ensuring at least 1 observable, num_obs_patterns >= 2.
        # So, obs_inds_sorted_batch will always have at least two columns.
        second_min_weight_indices = obs_inds_sorted_batch[:, 1]
        gaps_batch = (
            weights_all_obs_reshaped[np.arange(num_samples), second_min_weight_indices]
            - pred_llrs_batch
        )

        detector_density_batch = detector_outcomes.sum(axis=1) / num_detectors

        soft_outputs = {
            "pred_llr": pred_llrs_batch,
            "detector_density": detector_density_batch,
            "gap": gaps_batch,
        }

        if verbose:
            print(f"Average prediction LLR: {pred_llrs_batch.mean():.4f}")
            print(f"Average gap: {gaps_batch.mean():.4f}")
            print(
                f"Average predicted error weight: {preds_batch.sum(axis=1).mean():.2f}"
            )
            print("PyMatching batch decoding completed!")

        return preds_batch, soft_outputs
