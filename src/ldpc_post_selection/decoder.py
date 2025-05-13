from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix

from .stim_tools import dem_to_parity_check


class SoftOutputsBpLsdDecoder:
    """
    BP+LSD decoder with additional soft outputs for quantifying decoding confidence.
    """

    _bplsd: BpLsdDecoder
    H: csc_matrix
    obs_matrix: Optional[csc_matrix]
    p: np.ndarray
    circuit: Optional[stim.Circuit]
    bit_llrs: np.ndarray

    def __init__(
        self,
        H: Optional[csc_matrix | np.ndarray | List[List[bool | int]]] = None,
        *,
        p: Optional[np.ndarray | List[float]] = None,
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
            Parity check matrix. Internally stored as a scipy csc matrix of bool.
        p : 1D array-like of float
            Error probabilities.
        max_iter : int, optional
            Maximum iterations for the BP part of the decoder. Defaults to 30.
        bp_method : str, optional
            Method for BP message updates ('product_sum' or 'minimum_sum'). Defaults to
            "product_sum".
        lsd_method : str, optional
            Method for the LSD part ('LSD_0', 'LSD_E', 'LSD_CS'). Defaults to "LSD_0".
        lsd_order : int, optional
            Order parameter for LSD. Defaults to 0.
        """
        if H is None:
            if circuit is None:
                raise ValueError("Either H or circuit must be provided")
            H, obs, p = dem_to_parity_check(circuit.detector_error_model())
        else:
            obs = None

        if not isinstance(H, csc_matrix):
            H = csc_matrix(H)
        H = H.astype("uint8")
        p = np.asarray(p, dtype="float64")

        self._bplsd = BpLsdDecoder(
            H,
            error_channel=p,
            max_iter=max_iter,
            bp_method=bp_method,
            lsd_method=lsd_method,
            lsd_order=lsd_order,
            ms_scaling_factor=ms_scaling_factor,
            **kwargs,
        )
        self._bplsd.set_do_stats(True)
        self.H = H
        self.p = p
        self.bit_llrs = np.log((1 - self.p) / self.p)
        self.obs_matrix = obs
        self.circuit = circuit

    def decode(
        self,
        detector_outcomes: np.ndarray | List[List[bool | int]],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
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

        soft_outputs["cluster_sizes"] = np.array(final_cluster_bit_counts)
        soft_outputs["cluster_llrs"] = np.array(final_cluster_llrs)

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
