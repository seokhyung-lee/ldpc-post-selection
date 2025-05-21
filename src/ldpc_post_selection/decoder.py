from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from pymatching import Matching
from scipy.sparse import csc_matrix, vstack

from .stim_tools import dem_to_parity_check


class SoftOutputsDecoder:
    """
    Base class for decoders with additional soft outputs.
    """

    H: csc_matrix
    obs_matrix: Optional[csc_matrix]
    p: np.ndarray
    circuit: Optional[stim.Circuit]
    bit_llrs: np.ndarray
    decompose_errors: bool

    def __init__(
        self,
        H: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        *,
        p: Optional[np.ndarray | List[float]] = None,
        obs_matrix: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        circuit: Optional[stim.Circuit] = None,
        decompose_errors: bool = False,
    ):
        """
        Base class for decoders with additional soft outputs.

        Parameters
        ----------
        H : 2D array-like of bool/int, including scipy csc matrix
            Parity check matrix. Internally stored as a scipy csc matrix of uint8.
        p : 1D array-like of float
            Error probabilities.
        obs_matrix : 2D array-like of bool/int, including scipy csc matrix
            Observable matrix. Internally stored as a scipy csc matrix of uint8.
        circuit : stim.Circuit, optional
            Circuit (converted to H, p, and obs internally).
        decompose_errors : bool, optional
            If True and a circuit is provided, the detector error model will be generated
            with `decompose_errors=True`. Defaults to False.
        """
        self.decompose_errors = decompose_errors
        if circuit is not None:
            assert H is None and p is None and obs_matrix is None
            dem = circuit.detector_error_model(decompose_errors=self.decompose_errors)
            H, obs_matrix, p = dem_to_parity_check(dem)
        else:
            assert H is not None

        if not isinstance(H, csc_matrix):
            H = csc_matrix(H)
        H = H.astype("uint8")

        if p is not None:
            p = np.asarray(p, dtype="float64")

        if obs_matrix is not None:
            if not isinstance(obs_matrix, csc_matrix):
                obs_matrix = csc_matrix(obs_matrix)
            obs_matrix = obs_matrix.astype("uint8")

        self.H = H
        self.p = p
        self.bit_llrs = np.log((1 - self.p) / self.p)
        self.obs_matrix = obs_matrix
        self.circuit = circuit

    def decode(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
    ) -> Any:
        """
        Decode the detector measurement outcomes.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class SoftOutputsBpLsdDecoder(SoftOutputsDecoder):
    """
    BP+LSD decoder with additional soft outputs for quantifying decoding confidence.
    """

    _bplsd: BpLsdDecoder

    def __init__(
        self,
        H: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        *,
        p: Optional[np.ndarray | List[float]] = None,
        obs_matrix: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        circuit: Optional[stim.Circuit] = None,
        max_iter: int = 30,
        bp_method: str = "product_sum",
        lsd_method: str = "LSD_0",
        lsd_order: int = 0,
        ms_scaling_factor: float = 1.0,
        **kwargs,
    ):
        """
        BP+LSD decoder with additional soft outputs.

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
        max_iter : int, optional
            Maximum iterations for the BP part of the decoder. Defaults to 30.
        bp_method : str, optional
            Method for BP message updates ('product_sum' or 'minimum_sum'). Defaults to
            "product_sum".
        lsd_method : str, optional
            Method for the LSD part ('LSD_0', 'LSD_E', 'LSD_CS'). Defaults to "LSD_0".
        lsd_order : int, optional
            Order parameter for LSD. Defaults to 0.
        ms_scaling_factor : float, optional
            Scaling factor for min-sum BP. Defaults to 1.0.
        """
        # SoftOutputsBpLsdDecoder will always use decompose_errors=False if a circuit is given
        super().__init__(
            H=H, p=p, obs_matrix=obs_matrix, circuit=circuit, decompose_errors=False
        )

        self._bplsd = BpLsdDecoder(
            self.H,
            error_channel=self.p,
            max_iter=max_iter,
            bp_method=bp_method,
            lsd_method=lsd_method,
            lsd_order=lsd_order,
            ms_scaling_factor=ms_scaling_factor,
            **kwargs,
        )
        self._bplsd.set_do_stats(True)

    def decode(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Decode the detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 1D array-like of bool/int
            Detector measurement outcomes.

        Returns
        -------
        preds : np.ndarray
            Predicted error pattern.
        preds_bp : np.ndarray
            Predicted error pattern from BP. It is valid only if the BP is converged.
        converge : bool
            Whether the BP is converged.
        soft_outputs: Dict[str, Any]
            Soft outputs:
            - pred_llr (float): LLR of the predicted error pattern
            - detector_density (float): Fraction of violated detector outcomes
            - cluster_sizes (numpy array of int): Sizes of clusters and the remaining
            region (cluster_sizes[-1])
            - cluster_llrs (numpy array of float): LLRs of clusters and the remaining
            region (cluster_llrs[-1])
        """
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        bplsd = self._bplsd
        pred, pred_bp = bplsd.decode(detector_outcomes)
        pred: np.ndarray = pred.astype(bool)
        pred_bp: np.ndarray = pred_bp.astype(bool)

        ## Soft information
        stats: Dict[str, Any] = bplsd.statistics
        soft_outputs: Dict[str, float | int] = {}

        # Convergence
        converge = bplsd.converge

        # LLRs
        llrs = self.bit_llrs
        # bp_llrs = np.array(stats["bit_llrs"])
        # bp_llrs_plus = np.clip(bp_llrs, 0.0, None)

        # Prediction LLR
        soft_outputs["pred_llr"] = float(np.sum(llrs[pred]))
        # soft_outputs["pred_bp_llr"] = float(np.sum(bp_llrs[pred]))

        # Detector density
        soft_outputs["detector_density"] = detector_outcomes.sum() / len(
            detector_outcomes
        )

        # Cluster statistics
        individual_cluster_stats_dict: Dict[int, Dict[str, Any]] = stats[
            "individual_cluster_stats"
        ]

        final_cluster_bit_counts = []
        final_cluster_llrs = []
        # final_cluster_bp_llrs = []
        # final_cluster_bp_llrs_plus = []
        # total_boundary_size = 0
        for data in individual_cluster_stats_dict.values():
            if data.get("active", False):  # Assuming "active" key exists
                final_cluster_bit_counts.append(int(data["final_bit_count"]))
                final_bits = data["final_bits"]
                final_cluster_llrs.append(llrs[final_bits].sum())
                # final_cluster_bp_llrs.append(bp_llrs[final_bits].sum())
                # final_cluster_bp_llrs_plus.append(bp_llrs_plus[final_bits].sum())
                # final_bits_vector = np.zeros(self.H.shape[1], dtype=bool)
                # final_bits_vector[final_bits] = True
                # inside_cnt = self.H[:, final_bits_vector].getnnz(axis=1)
                # outside_cnt = self.H[:, ~final_bits_vector].getnnz(axis=1)
                # assert len(inside_cnt) == self.H.shape[0]
                # total_boundary_size += np.sum((inside_cnt > 0) & (outside_cnt > 0))

        outside_bit_counts = self.H.shape[1] - np.sum(final_cluster_bit_counts)
        outside_llrs = llrs.sum() - np.sum(final_cluster_llrs)
        final_cluster_bit_counts.append(outside_bit_counts)
        final_cluster_llrs.append(outside_llrs)

        soft_outputs["cluster_sizes"] = np.array(
            final_cluster_bit_counts, dtype=np.int_
        )
        soft_outputs["cluster_llrs"] = np.array(final_cluster_llrs, dtype=np.float64)

        assert len(final_cluster_bit_counts) == len(final_cluster_llrs)

        # soft_outputs["total_boundary_size"] = total_boundary_size

        # def max_or_zero(x):
        #     return np.max(x) if x else 0

        # # Cluster size sum & max cluster size
        # soft_outputs["total_cluster_size"] = sum(final_cluster_bit_counts)
        # soft_outputs["max_cluster_size"] = max_or_zero(final_cluster_bit_counts)
        # soft_outputs["cluster_num"] = len(final_cluster_bit_counts)

        # # LLR of final clusters
        # soft_outputs["total_cluster_llr"] = np.sum(final_cluster_llrs)
        # # soft_outputs["total_cluster_bp_llr"] = np.sum(final_cluster_bp_llrs)
        # # soft_outputs["total_cluster_bp_llr_plus"] = np.sum(final_cluster_bp_llrs_plus)
        # soft_outputs["max_cluster_llr"] = max_or_zero(final_cluster_llrs)
        # # soft_outputs["max_cluster_bp_llr"] = max_or_zero(final_cluster_bp_llrs)
        # # soft_outputs["max_cluster_bp_llr_plus"] = max_or_zero(
        # #     final_cluster_bp_llrs_plus
        # # )

        # # LLR outside clusters
        # soft_outputs["outside_cluster_llr"] = (
        #     np.sum(llrs) - soft_outputs["total_cluster_llr"]
        # )
        # # soft_outputs["outside_cluster_bp_llr"] = (
        # #     np.sum(bp_llrs) - soft_outputs["total_cluster_bp_llr"]
        # # )
        # # soft_outputs["outside_cluster_bp_llr_plus"] = (
        # #     np.sum(bp_llrs_plus) - soft_outputs["total_cluster_bp_llr_plus"]
        # # )

        # if norm_orders is not None:
        #     # Ensure norm_orders is a list
        #     if isinstance(norm_orders, (int, float)):
        #         norm_orders = [norm_orders]

        #     final_cluster_llrs_arr = np.array(final_cluster_llrs, dtype="float64")
        #     # final_cluster_bp_llrs_plus_arr = np.array(
        #     #     final_cluster_bp_llrs_plus, dtype="float64"
        #     # )

        #     # Calculate all {alpha}-norms of LLRs of final clusters in a vectorized way
        #     norm_vals = np.power(
        #         np.sum(
        #             np.power(
        #                 final_cluster_llrs_arr.reshape(-1, 1),
        #                 np.array(norm_orders).reshape(1, -1),
        #             ),
        #             axis=0,
        #         ),
        #         1 / np.array(norm_orders),
        #     )

        #     # Assign each norm value to its corresponding key in soft_outputs
        #     for i, alpha in enumerate(norm_orders):
        #         soft_outputs[f"cluster_llr_{alpha}_norm"] = norm_vals[i]

        #         # {alpha}-norm of BP+ LLRs of final clusters
        #         # norm_val_bp_plus = np.power(
        #         #     np.sum(np.power(final_cluster_bp_llrs_plus_arr, alpha)), 1 / alpha
        #         # )
        #         # soft_outputs[f"cluster_bp_llr_plus_{alpha}_norm"] = norm_val_bp_plus

        return pred, pred_bp, converge, soft_outputs


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
        weights = np.log((1 - self.p) / self.p)
        self._matching = Matching(H_obss_as_dets, weights=weights)

    def decode(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode the detector measurement outcomes for a single sample.

        Parameters
        ----------
        detector_outcomes : 1D numpy array or list of bool/int
            Detector measurement outcomes for a single sample.

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
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        all_obs_values = product([False, True], repeat=self.obs_matrix.shape[0])
        all_obs_values = np.array(list(all_obs_values), dtype=bool)

        repeated_detector_outcomes = np.tile(
            detector_outcomes, (all_obs_values.shape[0], 1)
        )
        det_outcomes_with_obs = np.concatenate(
            [repeated_detector_outcomes, all_obs_values], axis=1
        )

        matching = self._matching

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

        return pred, soft_outputs

    def decode_batch(
        self,
        detector_outcomes: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode a batch of detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 2D numpy array of bool/int
            Detector measurement outcomes. Each row is a sample.

        Returns
        -------
        preds : 2D numpy array of bool
            Predicted error patterns. Each row corresponds to a sample in detector_outcomes.
        soft_outputs : Dict[str, np.ndarray]
            See `decode` for details.
        """
        if detector_outcomes.ndim != 2:
            raise ValueError("Detector outcomes must be a 2D array for batch decoding.")

        num_samples, num_detectors = detector_outcomes.shape
        if num_detectors != self.H.shape[0]:
            raise ValueError(
                f"Number of detectors in outcomes ({num_detectors}) does not match "
                f"H matrix ({self.H.shape[0]})."
            )

        num_observables = self.obs_matrix.shape[0]
        all_obs_values = np.array(
            list(product([False, True], repeat=num_observables)), dtype=bool
        )
        num_obs_patterns = all_obs_values.shape[0]

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

        matching = self._matching
        preds_all_obs_batch, weights_all_obs_batch = matching.decode_batch(
            det_outcomes_with_obs_batch, return_weights=True
        )

        preds_all_obs_batch = preds_all_obs_batch.astype(bool)
        weights_all_obs_batch = weights_all_obs_batch.astype("float64")

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

        return preds_batch, soft_outputs
