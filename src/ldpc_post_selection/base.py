from typing import Any, List, Optional

import numpy as np
import stim
from scipy.sparse import csc_matrix

from .stim_tools import dem_to_parity_check


class SoftOutputsDecoder:
    """
    Base class for decoders with additional soft outputs.
    """

    H: csc_matrix
    obs_matrix: Optional[csc_matrix]
    priors: np.ndarray
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
        self.priors = p
        self.bit_llrs = np.log((1 - self.priors) / self.priors)
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