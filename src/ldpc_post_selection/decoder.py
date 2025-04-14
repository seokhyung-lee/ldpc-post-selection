from typing import List, Tuple

import numpy as np
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix


class BpLsdPsDecoder:
    def __init__(
        self,
        H: csc_matrix | np.ndarray | List[List[bool | int]],
        *,
        p: np.ndarray | List[float],
        **kwargs,
    ):
        """
        BP-LSD decoder with post-selection.

        Parameters
        ----------
        H : 2D array-like of bool/int, including scipy csc matrix
            Parity check matrix. Internally stored as a scipy csc matrix of bool.
        p : 1D array-like of float
            Error probabilities.
        kwargs : dict
            Additional keyword arguments for `ldpc.bplsd_decoder.BpLsdDecoder`.
        """
        if not isinstance(H, csc_matrix):
            H = csc_matrix(H, dtype=bool)
        p = np.asarray(p, dtype="float64")

        self._bplsd = BpLsdDecoder(H, error_channel=p, **kwargs)
        self._bplsd.set_do_stats(True)
        self.H = H
        self.p = p

    def decode(
        self, detector_outcomes: np.ndarray | List[List[bool | int]]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Decode the detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 2D array-like of bool/int
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

        bplsd = self._bplsd
        preds = bplsd.decode(detector_outcomes).astype(bool)

        # Calculate the cluster fraction and total cluster size
        stats = bplsd.statistics["individual_cluster_stats"]
        cluster_sizes = [
            data["final_bit_count"] for _, data in stats.items() if data["active"]
        ]
        cluster_total_size = sum(cluster_sizes)
        cluster_frac = cluster_total_size / self.H.shape[1]

        return preds, cluster_frac, cluster_total_size
