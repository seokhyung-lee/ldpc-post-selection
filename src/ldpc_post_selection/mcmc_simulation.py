import pickle
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Self, Tuple

import arviz as az
import numpy as np
import stim
from scipy.sparse import csc_matrix
from statsmodels.stats.proportion import proportion_confint

from .decoder import BpLsdPsDecoder
from .stim_tools import dem_to_parity_check


class BpLsdPsMCMCSimulator:
    c: float
    H: csc_matrix
    obs: csc_matrix
    p: np.ndarray
    bplsd_prms: Dict[str, Any]
    decoder: BpLsdPsDecoder
    e0: Optional[np.ndarray]
    last_error: Optional[np.ndarray]
    _fails: List[Optional[bool]]
    shots_rejected: int
    shots_accepted: int
    shots_decoded: int
    last_n_flip: Optional[int]
    last_shots_accepted: Optional[int]
    last_shots_decoded: Optional[int]
    time: List[Tuple[int, float]]

    def __init__(
        self,
        *,
        c: float,
        H: Optional[csc_matrix] = None,
        obs: Optional[csc_matrix] = None,
        p: Optional[List[float] | np.ndarray] = None,
        circuit: Optional[stim.Circuit] = None,
        e0: Optional[np.ndarray] = None,
        e0_shots_per_batch: int = 100,
        e0_prob_scale: Optional[float] = None,
        max_iter: int = 30,
        bp_method: str = "product_sum",
        lsd_method: str = "LSD_CS",
        lsd_order: int = 0,
        _force_no_e0: bool = False,
    ):
        """
        Initialize the Markov Chain Monte Carlo simulation for decoding LDPC codes
        with post-selection.

        Parameters
        ----------
        c : float
            The post-selection parameter. Only samples with cluster fraction <= (1-c)
            are accepted.
        H : csc_matrix of bool, optional
            The parity check matrix of the code. Required if `circuit` is not provided.
        obs : csc_matrix of bool, optional
            The logical observable matrix. Required if `circuit` is not provided.
        p : 1D array-like of float, optional
            The physical error rates for each qubit. Required if `circuit` is not provided.
        circuit : stim.Circuit, optional
            A stim circuit object containing the detector error model. If provided,
            `H`, `obs`, and `p` are derived from this circuit.
        e0 : 1D array-like of bool, optional
            An initial error vector satisfying the post-selection condition. If None,
            it will be found using the `find_e0` method.
        e0_shots_per_batch : int, default 100
            Number of shots to try in each batch when searching for `e0`.
        e0_prob_scale : float, optional
            A factor to scale the error probabilities when searching for `e0`.
        decoder : str, default "bplsd"
            The type of decoder to use. Currently, only "bplsd" is supported.
        max_iter : int, default 30
            Maximum iterations for the BP part of the decoder.
        bp_method : str, default "product_sum"
            Method for BP message updates ('product_sum' or 'minimum_sum').
        lsd_method : str, default "LSD_CS"
            Method for the LSD part ('LSD_0', 'LSD_E', 'LSD_CS').
        lsd_order : int, default 0
            Order parameter for LSD.
        _force_no_e0 : bool, default False
            If True, skip the search for `e0`.
        """
        if circuit is None:
            assert all(arg is not None for arg in [H, obs, p])
            if not isinstance(H, csc_matrix):
                H = csc_matrix(H)
            H = H.astype(bool)
            if not isinstance(obs, csc_matrix):
                obs = csc_matrix(obs)
            obs = obs.astype(bool)

        else:
            dem = circuit.detector_error_model()
            H, obs, p = dem_to_parity_check(dem)

        self.c = c
        self.H = H
        self.obs = obs
        self.p = np.asanyarray(p, dtype="float64")
        self.bplsd_prms = {
            "max_iter": max_iter,
            "bp_method": bp_method,
            "lsd_method": lsd_method,
            "lsd_order": lsd_order,
        }

        self.decoder = BpLsdPsDecoder(H, p, **self.bplsd_prms)

        if _force_no_e0:
            e0 = fail = None
        elif e0 is None:
            e0, fail = self.find_e0(e0_shots_per_batch, e0_prob_scale)
        else:
            syndrome = (e0.astype("uint8") @ H.T) % 2
            pred, cluster_frac, _ = self.decoder.decode(syndrome)
            if cluster_frac > 1 - c:
                raise ValueError(
                    f"Initial decoding aborted (cluster_frac = {cluster_frac}). Wrong e0 = {e0}."
                )
            residue = e0 ^ pred
            fail = ((residue.astype("uint8") @ obs.T) % 2).any()

        if e0 is None:
            self.e0 = self.last_error = None
        else:
            self.e0 = self.last_error = np.asanyarray(e0, dtype="bool")
        self._fails = [fail]

        self.shots_rejected = 0
        self.shots_accepted = self.shots_decoded = 1
        self.last_n_flip = None
        self.last_shots_accepted = self.last_shots_decoded = None

        self.time = []

    def initialize(self):
        """
        Reinitialize the simulation state to its starting point using the initial error `e0`.

        Raises
        ------
        ValueError
            If `e0` was not provided or found during initialization.
        """
        if self.e0 is None:
            raise ValueError("Unable to initialize since e0 is not given.")
        self.shots_loaded = None
        self.last_error = self.e0
        self._fails = [self._fails[0]]
        self.shots_accepted = self.shots_decoded = 1
        self.shots_rejected = 0
        self.last_n_flip = None
        self.last_shots_accepted = self.last_shots_decoded = None
        self.time = []

    @property
    def shots(self) -> int:
        """
        Return the total number of MCMC samples generated (including rejected).
        """
        return len(self._fails)

    @property
    def fails(self) -> np.ndarray:
        """
        Return a boolean array indicating whether each accepted sample resulted in a logical failure.
        """
        return np.array(self._fails, dtype="bool")

    @property
    def pfail(self) -> float:
        """
        Calculate the logical failure rate based on the accepted samples.
        """
        return self.fails.sum() / self.shots

    @property
    def pfails_history(self) -> np.ndarray:
        """
        Return the history of the logical failure rate calculated cumulatively over the samples.
        """
        return self.fails.cumsum() / np.arange(1, self.shots + 1)

    @property
    def acceptance_rate(self) -> float:
        """
        Calculate the overall acceptance rate of the Metropolis-Hastings proposals.
        This is the ratio of accepted samples to the total number of proposed samples (accepted + rejected).
        """
        return self.shots_accepted / (self.shots_rejected + self.shots_accepted)

    @property
    def decoding_rate(self) -> float:
        """
        Calculate the rate at which proposed samples were decoded (regardless of acceptance).
        This is the ratio of decoded samples to the total number of proposed samples.
        """
        return self.shots_decoded / (self.shots_rejected + self.shots_accepted)

    @property
    def effective_acceptance_rate(self) -> float:
        """
        Calculate the effective acceptance rate among the samples that were decoded.
        This is the ratio of accepted samples to decoded samples.
        """
        return self.shots_accepted / self.shots_decoded

    @property
    def sampling_rate(self) -> float:
        """
        Calculate the sampling rate in shots per second.
        """
        total_time = sum(t[1] for t in self.time)
        shots = sum(t[0] for t in self.time)
        if total_time == 0:
            return np.inf
        freq = shots / total_time
        return freq

    @property
    def ess(self) -> float:
        """
        Calculate the Effective Sample Size (ESS) of the failure indicators using arviz.
        """
        idata = az.convert_to_inference_data(self.fails)
        ess = az.ess(idata)["x"].values
        return ess

    def copy(self) -> Self:
        """
        Create a deep copy of the simulation object.
        """
        return deepcopy(self)

    def find_e0(
        self, shots_per_batch: int = 100, prob_scale: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Find an initial error vector `e0` that satisfies the post-selection condition.

        Parameters
        ----------
        shots_per_batch : int, default=100
            Number of error samples to generate and test in each batch.
        prob_scale : float, optional
            Factor to scale the error probabilities `p`. Useful for finding low-weight errors.

        Returns
        -------
        `e0` : numpy array of bool
            The found initial error vector.
        `fail` : bool
            Whether the initial error leads to a logical failure.
        """
        decoder = self.decoder
        c = self.c
        H = self.H
        p = self.p
        obs = self.obs

        if prob_scale is not None:
            p = p.copy() * prob_scale

        e0 = None
        fail = None
        while e0 is None:
            errors = np.random.uniform(size=(shots_per_batch, H.shape[1]))
            errors = errors < p.reshape(1, -1)
            dets = (errors.astype("uint8") @ H.T) % 2
            for errors_sng, dets_sng in zip(errors, dets):
                pred, cluster_frac = decoder.decode(dets_sng)
                if cluster_frac <= 1 - c:
                    e0 = errors_sng
                    residue = e0 ^ pred
                    fail = ((residue.astype("uint8") @ obs.T) % 2).any()
                    break
        return e0, fail

    def run(
        self,
        shots: int,
        *,
        flip_single_qubit: bool = True,
        n_flip: int = 1,
        adaptive_n_flip: bool = True,
        target_acc_rate: float = 0.234,
        alpha=0.05,
    ) -> np.ndarray:
        """
        Run the MCMC simulation to generate samples.

        Parameters
        ----------
        shots : int
            The number of MCMC samples to generate in this run.
        flip_single_qubit : bool, default=True
            If True, propose new states by flipping a single random qubit.
            If False, propose new states by flipping `n_flip` qubits chosen approximately
            uniformly at random.
        n_flip : int, default=1
            The number of qubits to flip when `flip_single_qubit` is False.
            Ignored if `adaptive_n_flip` is True after the first call.
        adaptive_n_flip : bool, default=True
            If True, adaptively adjust `n_flip` (when `flip_single_qubit` is False)
            or switch between single/multi-qubit flips to target a specific
            effective acceptance rate.
        target_acc_rate : float, default=0.234
            The target effective acceptance rate for the adaptive algorithm.
        alpha : float, default=0.05
            The significance level for the confidence interval used in the adaptive algorithm.

        Returns
        -------
        fails : 1D numpy array of bool
            Failure outcomes (True for logical failure, False otherwise) for the newly
            generated samples in this run.
        """
        t0 = time.time()
        c = self.c
        H = self.H
        obs = self.obs
        p = self.p
        decoder = self.decoder
        fails = self._fails

        log_p = np.log(p)
        log_one_minus_p = np.log(1 - p)

        num_error_locs = H.shape[1]

        if adaptive_n_flip:
            if self.last_n_flip is None:
                self.last_n_flip = n_flip
                self.last_shots_accepted = self.last_shots_decoded = 0
            else:
                n_flip = self.last_n_flip
                assert self.last_shots_accepted is not None
                assert self.last_shots_decoded is not None
        else:
            self.last_n_flip = self.last_shots_accepted = self.last_shots_decoded = None

        for i in range(shots):
            # Adaptively choose n_flip depending on the effective acceptance rate
            if adaptive_n_flip and self.last_shots_decoded:
                eff_acc_rate_low, eff_acc_rate_high = proportion_confint(
                    self.last_shots_accepted,
                    self.last_shots_decoded,
                    method="wilson",
                    alpha=alpha,
                )
                if eff_acc_rate_low > target_acc_rate:
                    n_flip = min(n_flip + 1, num_error_locs)
                    self.last_n_flip = n_flip
                    self.last_shots_accepted = self.last_shots_decoded = 0
                    # print(i, n_flip, eff_acc_rate_low, eff_acc_rate_high)
                elif eff_acc_rate_high < target_acc_rate:
                    n_flip = max(n_flip - 1, 1)
                    self.last_n_flip = n_flip
                    self.last_shots_accepted = self.last_shots_decoded = 0
                    # print(i, n_flip, eff_acc_rate_low, eff_acc_rate_high)

            if flip_single_qubit:
                error_diff = np.zeros(num_error_locs, dtype="bool")
                error_diff[np.random.randint(0, num_error_locs)] = True

            else:
                p_flip = n_flip / num_error_locs
                while True:
                    error_diff = np.random.uniform(size=num_error_locs) < p_flip
                    if error_diff.sum():
                        break
            error_cand = self.last_error ^ error_diff

            only_in_last_error = self.last_error & ~error_cand
            only_in_error_cand = ~self.last_error & error_cand
            log_q = -np.sum(log_p[only_in_last_error])
            log_q += np.sum(log_one_minus_p[only_in_last_error])
            log_q -= np.sum(log_one_minus_p[only_in_error_cand])
            log_q += np.sum(log_p[only_in_error_cand])
            q = min(1, np.exp(log_q))

            if np.random.uniform() < q:
                syndrome = (error_cand @ H.T) % 2
                pred, cluster_frac, _ = decoder.decode(syndrome)
                self.shots_decoded += 1
                if adaptive_n_flip:
                    self.last_shots_decoded += 1

                if cluster_frac <= 1 - c:
                    residue = error_cand ^ pred
                    fail = ((residue @ obs.T) % 2).any()
                    fails.append(fail)
                    self.shots_accepted += 1
                    if adaptive_n_flip:
                        self.last_shots_accepted += 1
                    self.last_error = error_cand
                    continue

            # If not accepted, use the last error again
            fails.append(fails[-1])
            self.shots_rejected += 1

        self.time.append((shots, time.time() - t0))

        return np.array(fails[-shots:], dtype="bool")

    def __getstate__(self):
        """
        Prepare the object's state for pickling. Excludes the decoder object.
        """
        state = self.__dict__.copy()
        if "decoder" in state:
            del state["decoder"]
        return state

    def __setstate__(self, state):
        """
        Restore the object's state from a pickled state with the decoder reinitialized.
        """
        if "fails" in state:
            state["_fails"] = state["fails"]
            del state["fails"]
        if "shots_recorded" in state:
            state["shots_rej"] = state["shots_recorded"]
            del state["shots_recorded"]
        if "shots_rej" in state:
            state["shots_rejected"] = state["shots_rej"]
            del state["shots_rej"]
        if "shots_acc" in state:
            state["shots_accepted"] = state["shots_acc"]
            del state["shots_acc"]
        if "shots_dec" in state:
            state["shots_decoded"] = state["shots_dec"]
            del state["shots_dec"]
        if "last_shots_acc" in state:
            state["last_shots_accepted"] = state["last_shots_acc"]
            del state["last_shots_acc"]
        if "last_shots_dec" in state:
            state["last_shots_decoded"] = state["last_shots_dec"]
            del state["last_shots_dec"]
        if "time" not in state:
            state["time"] = []
        self.__dict__.update(state)
        self.decoder = BpLsdPsDecoder(self.H, self.p, **self.bplsd_prms)

    def save(self, path: str):
        """
        Save the current state of the simulation object to a file using pickle.

        Parameters
        ----------
        path : str
            The file path where the object should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> Self:
        """
        Load a simulation object from a pickle file.

        Parameters
        ----------
        path : str
            The file path from which to load the object.

        Returns
        -------
        MarkovChainMonteCarloDecoderSimulation
            The loaded simulation object.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)

        assert isinstance(obj, cls)
        return obj
