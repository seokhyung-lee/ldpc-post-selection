from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix

from src.ldpc_post_selection.stim_tools import dem_to_parity_check


class BpLsdPsDecoder:
    """
    BP+LSD decoder with post-selection.
    """

    _bplsd: BpLsdDecoder
    H: csc_matrix
    obs_matrix: Optional[csc_matrix]
    p: np.ndarray
    circuit: Optional[stim.Circuit]
    soft_info_dtypes: Dict[str, Any]

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
        self.obs_matrix = obs
        self.circuit = circuit
        self.soft_info_dtypes = {
            "converge": "bool",
            "cluster_size_sum": "int32",
            "cluster_num": "int32",
            "pred_bp_llr": "float64",
            "cluster_bp_llr_sum": "float64",
            "outside_cluster_bp_llr": "float64",
            "pred_llr": "float64",
            "cluster_llr_sum": "float64",
            "outside_cluster_llr": "float64",
        }

    def decode(
        self, detector_outcomes: np.ndarray | List[List[bool | int]]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Decode the detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 1D array-like of bool/int
            Detector measurement outcomes.

        Returns
        -------
        preds : np.ndarray
            Predicted codeword.
        cluster_frac : float
            Cluster fraction.
        total_cluster_size : float
            Total cluster size.
        """
        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        bplsd = self._bplsd
        pred: np.ndarray = bplsd.decode(detector_outcomes).astype(bool)

        ## Soft information
        stats: Dict[str, Any] = bplsd.statistics

        # Convergence
        converge = bplsd.converge

        # Prediction LLR
        bit_bp_llrs = np.array(stats["bit_llrs"])
        pred_bp_llr = float(np.sum(bit_bp_llrs[pred]))

        # Prior LLR
        bit_llrs = np.log(self.p / (1 - self.p))
        pred_llr = float(np.sum(bit_llrs[pred]))

        # Cluster size sum
        cluster_stats = stats["individual_cluster_stats"]
        cluster_sizes = [
            data["final_bit_count"]
            for _, data in cluster_stats.items()
            if data["active"]
        ]
        cluster_size_sum = sum(cluster_sizes)

        # Cluster number
        cluster_num = len(cluster_stats)

        # Cluster LLR sum & prior-LLR sum
        cluster_bits = []
        for _, data in cluster_stats.items():
            if data["active"]:
                cluster_bits.extend(data["final_bits"])
        cluster_bp_llr_sum = float(np.sum(bit_bp_llrs[cluster_bits]))
        cluster_llr_sum = float(np.sum(bit_llrs[cluster_bits]))

        total_bp_llr = float(np.sum(bit_bp_llrs))
        outside_cluster_bp_llr = total_bp_llr - cluster_bp_llr_sum

        total_llr = float(np.sum(bit_llrs))
        outside_cluster_llr = total_llr - cluster_llr_sum

        soft_info = {
            "converge": converge,
            "cluster_size_sum": cluster_size_sum,
            "cluster_num": cluster_num,
            "pred_bp_llr": pred_bp_llr,
            "cluster_bp_llr_sum": cluster_bp_llr_sum,
            "outside_cluster_bp_llr": outside_cluster_bp_llr,
            "pred_llr": pred_llr,
            "cluster_llr_sum": cluster_llr_sum,
            "outside_cluster_llr": outside_cluster_llr,
        }

        return pred, soft_info
