from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from pymatching import Matching
from scipy.sparse import csc_matrix, vstack
from scipy import sparse

from .stim_tools import dem_to_parity_check


def compute_cluster_stats(
    clusters: np.ndarray, llrs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cluster statistics from cluster assignments using numpy native functions.

    Parameters
    ----------
    clusters : 1D numpy array of int
        Cluster assignments for each bit (0 = outside cluster, 1+ = cluster ID).
    llrs : 1D numpy array of float
        Log-likelihood ratios for each bit.

    Returns
    -------
    cluster_sizes : 1D numpy array of int
        Size of each cluster (index corresponds to cluster ID).
    cluster_llrs : 1D numpy array of float
        Sum of LLRs for each cluster (index corresponds to cluster ID).
    """
    max_cluster_id = clusters.max()

    # Use bincount to efficiently compute cluster sizes and LLR sums
    cluster_sizes = np.bincount(clusters, minlength=max_cluster_id + 1).astype(np.int_)
    cluster_llrs = np.bincount(clusters, weights=llrs, minlength=max_cluster_id + 1)

    return cluster_sizes, cluster_llrs


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
            error_channel=self.priors,
            max_iter=max_iter,
            bp_method=bp_method,
            lsd_method=lsd_method,
            lsd_order=lsd_order,
            ms_scaling_factor=ms_scaling_factor,
            **kwargs,
        )
        self._bplsd.set_do_stats(True)

    def _get_logical_classes_to_explore(
        self,
        predicted_logical_class: np.ndarray,
        explore_only_nearby_logical_classes: bool,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Determine logical classes to explore for gap proxy computation.

        Parameters
        ----------
        predicted_logical_class : 1D numpy array of bool
            Predicted logical class.
        explore_only_nearby_logical_classes : bool
            If True, only explore adjacent logical classes (flip one bit).
            If False, explore all possible logical classes except the predicted one.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        logical_classes_to_explore : list of 1D numpy array of bool
            List of logical classes to explore.
        """
        num_observables = len(predicted_logical_class)
        logical_classes_to_explore = []

        if verbose:
            print(
                f"  Getting logical classes to explore (num_observables={num_observables})"
            )

        if explore_only_nearby_logical_classes:
            # Only adjacent logical classes (flip one bit at a time)
            for i in range(num_observables):
                nearby_logical_class = predicted_logical_class.copy()
                nearby_logical_class[i] = not nearby_logical_class[i]
                logical_classes_to_explore.append(nearby_logical_class)
            if verbose:
                print(
                    f"  Exploring {len(logical_classes_to_explore)} nearby logical classes"
                )
        else:
            # All possible logical classes except the predicted one
            all_logical_classes = product([False, True], repeat=num_observables)
            for logical_class in all_logical_classes:
                logical_class_array = np.array(logical_class, dtype=bool)
                if not np.array_equal(logical_class_array, predicted_logical_class):
                    logical_classes_to_explore.append(logical_class_array)
            if verbose:
                print(
                    f"  Exploring {len(logical_classes_to_explore)} total logical classes"
                )

        return logical_classes_to_explore

    def _perform_fixed_logical_class_decoding(
        self,
        detector_outcomes: np.ndarray,
        fixed_logical_class: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """
        Perform fixed-logical-class decoding for a given logical class.

        Parameters
        ----------
        detector_outcomes : 1D numpy array of bool
            Detector measurement outcomes.
        fixed_logical_class : 1D numpy array of bool
            Fixed logical class to decode with.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        pred_llr : float
            Prediction LLR for the fixed logical class decoding.
        pred : 1D numpy array of bool
            Predicted error pattern for the fixed logical class decoding.
        """
        if verbose:
            print(
                f"    Performing fixed-logical-class decoding for class {fixed_logical_class}"
            )

        # Construct H_obs_appended by vertically stacking H and obs_matrix
        H_obs_appended = vstack([self.H, self.obs_matrix], format="csc", dtype="uint8")

        # Create new decoder with appended matrix
        # Use same parameters as the original decoder, but no obs_matrix to avoid recursion
        decoder_fixed = SoftOutputsBpLsdDecoder(
            H=H_obs_appended,
            p=self.priors,
            obs_matrix=None,  # No observables for the fixed decoder to prevent recursion
            max_iter=self._bplsd.max_iter,
            bp_method=self._bplsd.bp_method,
            lsd_method=self._bplsd.lsd_method,
            lsd_order=self._bplsd.lsd_order,
            ms_scaling_factor=self._bplsd.ms_scaling_factor,
        )

        # Construct detector_outcomes_obs_appended
        detector_outcomes_obs_appended = np.concatenate(
            [detector_outcomes, fixed_logical_class]
        )

        # Perform decoding without cluster stats computation for efficiency
        pred_fixed, _, _, soft_outputs_fixed = decoder_fixed.decode(
            detector_outcomes_obs_appended,
            include_cluster_stats=False,
            compute_logical_gap_proxy=False,  # Avoid recursion
            verbose=False,  # Don't cascade verbose to avoid excessive output
        )

        if verbose:
            print(
                f"    Fixed-class decoding completed, pred_llr={soft_outputs_fixed['pred_llr']:.4f}"
            )

        return soft_outputs_fixed["pred_llr"], pred_fixed

    def _compute_logical_gap_proxy(
        self,
        detector_outcomes: np.ndarray,
        pred: np.ndarray,
        original_pred_llr: float,
        explore_only_nearby_logical_classes: bool,
        verbose: bool = False,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Compute logical gap proxy by exploring different logical classes iteratively.

        Parameters
        ----------
        detector_outcomes : 1D numpy array of bool
            Detector measurement outcomes.
        pred : 1D numpy array of bool
            Original predicted error pattern.
        original_pred_llr : float
            Original prediction LLR.
        explore_only_nearby_logical_classes : bool
            Whether to explore only nearby logical classes.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

        Returns
        -------
        gap_proxy : float
            Logical gap proxy (difference between minimum and second minimum pred_llr).
        best_pred : 1D numpy array of bool
            Best predicted error pattern (corresponding to minimum pred_llr).
        best_pred_llr : float
            Best prediction LLR (minimum among all explored).
        """
        if verbose:
            print("  Computing logical gap proxy...")

        if self.obs_matrix is None:
            if verbose:
                print("  No observables, returning original results")
            return 0.0, pred, original_pred_llr

        # Calculate original logical class
        original_logical_class = (
            (pred.astype("uint8") @ self.obs_matrix.T) % 2
        ).astype(bool)

        if verbose:
            print(f"  Original logical class: {original_logical_class}")

        # Store all explored logical classes and their results
        explored_classes = {}  # logical_class tuple -> (pred_llr, pred_pattern)
        explored_classes[tuple(original_logical_class)] = (original_pred_llr, pred)

        # Queue for iterative exploration
        to_explore = [original_logical_class]
        explored_set = {tuple(original_logical_class)}

        if not explore_only_nearby_logical_classes:
            # If exploring all classes, generate them all at once
            num_observables = len(original_logical_class)
            all_logical_classes = list(product([False, True], repeat=num_observables))

            if verbose:
                print(f"  Exploring all {len(all_logical_classes)} logical classes")

            for logical_class_tuple in all_logical_classes:
                logical_class = np.array(logical_class_tuple, dtype=bool)
                if tuple(logical_class) not in explored_set:
                    if verbose:
                        print(f"  Processing logical class {logical_class}")
                    pred_llr, pred_pattern = self._perform_fixed_logical_class_decoding(
                        detector_outcomes, logical_class, verbose=verbose
                    )
                    explored_classes[tuple(logical_class)] = (pred_llr, pred_pattern)
        else:
            # Iterative exploration for nearby classes only
            iteration = 0
            while to_explore:
                iteration += 1
                if verbose:
                    print(
                        f"  Iteration {iteration}: {len(to_explore)} classes to explore"
                    )

                current_class = to_explore.pop(0)

                # Get nearby logical classes
                nearby_classes = self._get_logical_classes_to_explore(
                    current_class, True, verbose=False
                )

                new_best_found = False
                current_best_llr = min(llr for llr, _ in explored_classes.values())

                for logical_class in nearby_classes:
                    logical_class_tuple = tuple(logical_class)
                    if logical_class_tuple not in explored_set:
                        if verbose:
                            print(f"    Processing nearby class {logical_class}")

                        pred_llr, pred_pattern = (
                            self._perform_fixed_logical_class_decoding(
                                detector_outcomes, logical_class, verbose=verbose
                            )
                        )
                        explored_classes[logical_class_tuple] = (pred_llr, pred_pattern)
                        explored_set.add(logical_class_tuple)

                        # If this is better than current best, add it to exploration queue
                        if pred_llr < current_best_llr:
                            to_explore.append(logical_class)
                            new_best_found = True
                            current_best_llr = pred_llr
                            if verbose:
                                print(f"    New best found: {pred_llr:.4f}")

                if verbose:
                    print(
                        f"  Iteration {iteration} completed, new best found: {new_best_found}"
                    )

        # Find best and second best
        all_pred_llrs = [llr for llr, _ in explored_classes.values()]
        all_pred_llrs.sort()

        best_pred_llr = all_pred_llrs[0]
        second_best_pred_llr = (
            all_pred_llrs[1] if len(all_pred_llrs) > 1 else best_pred_llr
        )

        # Find the logical class corresponding to best pred_llr
        best_logical_class_tuple = None
        for logical_class_tuple, (pred_llr, _) in explored_classes.items():
            if pred_llr == best_pred_llr:
                best_logical_class_tuple = logical_class_tuple
                break

        best_pred = explored_classes[best_logical_class_tuple][1]
        gap_proxy = second_best_pred_llr - best_pred_llr

        if verbose:
            print(f"  Total logical classes explored: {len(explored_classes)}")
            print(f"  Best pred_llr: {best_pred_llr:.4f}")
            print(f"  Second best pred_llr: {second_best_pred_llr:.4f}")
            print(f"  Gap proxy: {gap_proxy:.4f}")
            print(f"  Best logical class: {np.array(best_logical_class_tuple)}")

        return gap_proxy, best_pred, best_pred_llr

    def decode(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
        include_cluster_stats: bool = True,
        compute_logical_gap_proxy: bool = False,
        explore_only_nearby_logical_classes: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Decode the detector measurement outcomes.

        Parameters
        ----------
        detector_outcomes : 1D array-like of bool/int
            Detector measurement outcomes.
        include_cluster_stats : bool
            Whether to compute soft outputs related to cluster statistics.
            Defaults to True.
            Automatically set to False when compute_logical_gap_proxy is True.
        compute_logical_gap_proxy : bool
            Whether to compute logical gap proxy. Defaults to False.
        explore_only_nearby_logical_classes : bool
            If True, only explore adjacent logical classes for gap proxy computation.
            If False, explore all possible logical classes. Only used when
            compute_logical_gap_proxy is True. Defaults to True.
        verbose : bool, optional
            If True, print progress information. Defaults to False.

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
            - clusters (1D numpy array of int): Cluster assignments of each bit (0 = outside clusters)
            - cluster_sizes (1D numpy array of int): Sizes of clusters and the remaining
            region (cluster_sizes[-1])
            - cluster_llrs (1D numpy array of float): LLRs of clusters and the remaining
            region (cluster_llrs[-1])
            - gap_proxy (float): Logical gap proxy (only if compute_logical_gap_proxy=True)
        """
        if verbose:
            print("Starting BP+LSD decoding...")

        # Prevent simultaneous use of both options to simplify the implementation
        if compute_logical_gap_proxy:
            include_cluster_stats = False

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        if verbose:
            print(f"Detector outcomes shape: {detector_outcomes.shape}")
            print(f"Number of violated detectors: {detector_outcomes.sum()}")

        bplsd = self._bplsd
        pred, pred_bp = bplsd.decode(detector_outcomes, custom=True)
        pred: np.ndarray = pred.astype(bool)
        pred_bp: np.ndarray = pred_bp.astype(bool)

        if verbose:
            print("BP+LSD decoding completed")
            print(f"Predicted error weight: {pred.sum()}")

        ## Soft information
        stats: Dict[str, Any] = bplsd.statistics
        soft_outputs: Dict[str, float | int] = {}

        # Convergence
        converge = bplsd.converge
        if verbose:
            print(f"BP convergence: {converge}")

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

        if verbose:
            print(f"Prediction LLR: {soft_outputs['pred_llr']:.4f}")
            print(f"Detector density: {soft_outputs['detector_density']:.4f}")

        if include_cluster_stats:
            if verbose:
                print("Computing cluster statistics...")

            # Build cluster assignments
            individual_cluster_stats_dict: Dict[int, Dict[str, Any]] = stats[
                "individual_cluster_stats"
            ]
            clusters = np.zeros(self.H.shape[1], dtype=np.int_)
            cluster_id = 1
            for data in individual_cluster_stats_dict.values():
                if data.get("active", False):  # Assuming "active" key exists
                    final_bits = data["final_bits"]
                    clusters[final_bits] = cluster_id
                    cluster_id += 1

            # Calculate cluster statistics
            cluster_sizes, cluster_llrs = compute_cluster_stats(clusters, llrs)

            soft_outputs["clusters"] = clusters  # 1D array of int
            soft_outputs["cluster_sizes"] = cluster_sizes
            soft_outputs["cluster_llrs"] = cluster_llrs

            if verbose:
                print(f"Number of active clusters: {cluster_id - 1}")
                print(f"Cluster sizes: {cluster_sizes}")

        if compute_logical_gap_proxy:
            if verbose:
                print("Computing logical gap proxy...")
            gap_proxy, best_pred, best_pred_llr = self._compute_logical_gap_proxy(
                detector_outcomes,
                pred,
                soft_outputs["pred_llr"],
                explore_only_nearby_logical_classes,
                verbose=verbose,
            )
            soft_outputs["gap_proxy"] = gap_proxy

            # Update prediction and related soft outputs if a better one was found
            if best_pred_llr < soft_outputs["pred_llr"]:
                if verbose:
                    print(
                        f"  Updating prediction: {soft_outputs['pred_llr']:.4f} -> {best_pred_llr:.4f}"
                    )
                pred = best_pred
                soft_outputs["pred_llr"] = best_pred_llr

        if verbose:
            print("BP+LSD decoding process completed!")

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
