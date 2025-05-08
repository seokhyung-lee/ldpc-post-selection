from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix

from .stim_tools import dem_to_parity_check


class BpLsdPsDecoder:
    """
    BP+LSD decoder with post-selection.
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
        bp_method: str = "minimum_sum",
        lsd_method: str = "LSD_0",
        lsd_order: int = 0,
        **kwargs,
    ):
        """
        BP+LSD decoder with post-selection.

        Parameters
        ----------
        H : 2D array-like of bool/int, including scipy csc matrix
            Parity check matrix. Internally stored as a scipy csc matrix of bool.
        p : 1D array-like of float
            Error probabilities.
        max_iter : int, optional
            Maximum iterations for the BP part of the decoder. Defaults to 30.
        bp_method : str, optional
            Method for BP message updates ('product_sum' or 'minimum_sum'). Defaults to "product_sum".
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
        norm_order: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Decode the detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 1D array-like of bool/int
            Detector measurement outcomes.
        norm_order : float, optional
            Norm order for cluster LLR. If provided, the {norm_order}-norm of cluster LLRs
            (each of which is the LLR sum of a cluster) will be computed and returned in
            the soft_outputs dict.

        Returns
        -------
        preds : np.ndarray
            Predicted error pattern.
        preds_bp : np.ndarray
            Predicted error pattern from BP. It is valid only if the BP is converged.
        converge : bool
            Whether the BP is converged.
        soft_outputs: Dict[str, float | int]
            Soft outputs. Three types of LLRs are used:
            - llr: LLRs for prior probabilities
            - bp_llr: LLRs obtained from BP
            - bp_llr_plus: LLRs obtained from BP, clipped to be non-negative
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
        soft_info: Dict[str, float | int] = {}

        # Convergence
        converge = bplsd.converge

        # LLRs
        llrs = self.bit_llrs
        bp_llrs = np.array(stats["bit_llrs"])
        bp_llrs_plus = np.clip(bp_llrs, 0.0, None)

        # Prediction LLR
        soft_info["pred_llr"] = float(np.sum(llrs[pred]))
        soft_info["pred_bp_llr"] = float(np.sum(bp_llrs[pred]))

        # Detector density
        soft_info["detector_density"] = detector_outcomes.sum() / len(detector_outcomes)

        # Cluster statistics
        individual_cluster_stats_dict: Dict[int, Dict[str, Any]] = stats[
            "individual_cluster_stats"
        ]

        final_cluster_bit_counts = []
        final_cluster_llrs = []
        final_cluster_bp_llrs = []
        final_cluster_bp_llrs_plus = []
        total_boundary_size = 0
        for data in individual_cluster_stats_dict.values():
            if data.get("active", False):  # Assuming "active" key exists
                final_cluster_bit_counts.append(int(data["final_bit_count"]))
                final_bits = data["final_bits"]
                final_cluster_llrs.append(llrs[final_bits].sum())
                final_cluster_bp_llrs.append(bp_llrs[final_bits].sum())
                final_cluster_bp_llrs_plus.append(bp_llrs_plus[final_bits].sum())
                final_bits_vector = np.zeros(self.H.shape[1], dtype=bool)
                final_bits_vector[final_bits] = True
                inside_cnt = self.H[:, final_bits_vector].getnnz(axis=1)
                outside_cnt = self.H[:, ~final_bits_vector].getnnz(axis=1)
                assert len(inside_cnt) == self.H.shape[0]
                total_boundary_size += np.sum((inside_cnt > 0) & (outside_cnt > 0))

        soft_info["total_boundary_size"] = total_boundary_size

        # Cluster size sum & max cluster size
        soft_info["total_cluster_size"] = sum(final_cluster_bit_counts)
        soft_info["max_cluster_size"] = (
            max(final_cluster_bit_counts) if final_cluster_bit_counts else 0
        )
        soft_info["cluster_num"] = len(final_cluster_bit_counts)

        # LLR of final clusters
        soft_info["total_cluster_llr"] = np.sum(final_cluster_llrs)
        soft_info["total_cluster_bp_llr"] = np.sum(final_cluster_bp_llrs)
        soft_info["total_cluster_bp_llr_plus"] = np.sum(final_cluster_bp_llrs_plus)
        soft_info["max_cluster_llr"] = max(final_cluster_llrs)
        soft_info["max_cluster_bp_llr"] = max(final_cluster_bp_llrs)
        soft_info["max_cluster_bp_llr_plus"] = max(final_cluster_bp_llrs_plus)

        # LLR outside clusters
        soft_info["outside_cluster_llr"] = np.sum(llrs) - soft_info["total_cluster_llr"]
        soft_info["outside_cluster_bp_llr"] = (
            np.sum(bp_llrs) - soft_info["total_cluster_bp_llr"]
        )
        soft_info["outside_cluster_bp_llr_plus"] = (
            np.sum(bp_llrs_plus) - soft_info["total_cluster_bp_llr_plus"]
        )

        if norm_order is not None:
            # {norm_order}-norm of LLRs of final clusters
            def norm(x):
                return np.power(np.sum(np.power(x, norm_order)), 1 / norm_order)

            soft_info["cluster_llr_norm"] = norm(final_cluster_llrs)
            # soft_info["cluster_bp_llr_norm"] = norm(final_cluster_bp_llrs)
            soft_info["cluster_bp_llr_plus_norm"] = norm(final_cluster_bp_llrs_plus)

        return pred, pred_bp, converge, soft_info
