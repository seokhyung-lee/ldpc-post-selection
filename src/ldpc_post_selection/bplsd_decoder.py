from itertools import product
import random
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Self

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix, vstack

from .base import SoftOutputsDecoder
from .cluster_tools import compute_cluster_stats


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
        detector_time_coords: int | Sequence[int] = -1,
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
        max_iter : int
            Maximum iterations for the BP part of the decoder. Defaults to 30.
        bp_method : str
            Method for BP message updates ('product_sum' or 'minimum_sum'). Defaults to
            "product_sum".
        lsd_method : str
            Method for the LSD part ('LSD_0', 'LSD_E', 'LSD_CS'). Defaults to "LSD_0".
        lsd_order : int
            Order parameter for LSD. Defaults to 0.
        ms_scaling_factor : float
            Scaling factor for min-sum BP. Defaults to 1.0.
        detector_time_coords : int or sequence of int, defaults to -1
            Time coordinates of the detectors for sliding window decoding.
            If not given, the last element of coordinates is used for each detector.
            If a single integer, it indicates which element of the coordinates to use.
            If a sequence of integers, it explicitly specifies the time coordinates of
            the detectors, so its length must be the same as the number of detectors.
            If `circuit` is not given, this must be a sequence of integers.
        """
        # SoftOutputsBpLsdDecoder will always use decompose_errors=False if a circuit is given
        super().__init__(
            H=H, p=p, obs_matrix=obs_matrix, circuit=circuit, decompose_errors=False
        )

        bplsd_kwargs = {
            "max_iter": max_iter,
            "bp_method": bp_method,
            "lsd_method": lsd_method,
            "lsd_order": lsd_order,
            "ms_scaling_factor": ms_scaling_factor,
        }
        bplsd_kwargs.update(kwargs)

        self._bplsd_kwargs = bplsd_kwargs

        self._bplsd = BpLsdDecoder(
            self.H,
            error_channel=self.priors,
            always_run_lsd=True,
            **bplsd_kwargs,
        )
        self._bplsd.set_do_stats(True)

        try:
            if len(detector_time_coords) == self.H.shape[0]:
                self._detector_time_coords = np.array(detector_time_coords, dtype=int)
                self._det_time_coord_index = None
            else:
                raise ValueError(
                    "detector_time_coords must be a sequence of integers with the same length as the number of detectors"
                )
        except TypeError:
            self._detector_time_coords = None
            self._det_time_coord_index = detector_time_coords

        # Initialize caches for sliding window decoding
        self._window_structure_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._decoder_cache: Dict[str, SoftOutputsBpLsdDecoder] = {}

        # Precompute adjacency matrix for efficient cluster labeling
        self._adjacency_matrix = (self.H.T @ self.H == 1).astype(bool)

    @property
    def detector_time_coords(self) -> np.ndarray:
        if self._detector_time_coords is not None:
            return self._detector_time_coords.copy()

        else:
            if self.circuit is None:
                raise ValueError(
                    "detector_time_coords must be a sequence of integers if circuit is not given"
                )

            det_coords_dict = self.circuit.get_detector_coordinates()
            det_indices = sorted(det_coords_dict.keys())
            det_time_coords = [
                det_coords_dict[i][self._det_time_coord_index] for i in det_indices
            ]
            det_time_coords = np.array(det_time_coords, dtype=int)
            self._detector_time_coords = det_time_coords
            return det_time_coords.copy()

    def _check_detector_time_coords_validity(self):
        time_coords = self.detector_time_coords
        if min(time_coords) != 0:
            raise ValueError("Detector time coordinates must start from 0")

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

    def _hash_matrix_and_priors(self, H_matrix: csc_matrix, priors: np.ndarray) -> str:
        """
        Generate a hash key for a matrix and priors combination.

        Parameters
        ----------
        H_matrix : scipy csc_matrix
            The parity check matrix.
        priors : 1D numpy array of float
            The error probabilities.

        Returns
        -------
        hash_key : str
            Hash string representing the matrix and priors configuration.
        """
        # Create a hash based on matrix structure and priors
        hasher = hashlib.md5()

        # Hash matrix shape
        hasher.update(str(H_matrix.shape).encode())

        # Hash matrix data (indices and indptr are sufficient for structure)
        hasher.update(H_matrix.indices.tobytes())
        hasher.update(H_matrix.indptr.tobytes())

        # Hash priors
        hasher.update(priors.tobytes())

        return hasher.hexdigest()

    def _get_or_create_window_structure(
        self, window_size: int, commit_size: int, window_position: int
    ) -> Dict[str, Any]:
        """
        Get or create cached window structure for a given window configuration.

        Parameters
        ----------
        window_size : int
            Number of rounds in each window.
        commit_size : int
            Number of rounds for each commitment.
        window_position : int
            The window index (w).

        Returns
        -------
        window_structure : dict
            Cached window structure containing:
            - window_detector_mask: boolean mask for detectors in this window
            - H_window_base: H matrix rows for this window (before fault filtering)
            - window_start: start time of window
            - window_end: end time of window
        """
        cache_key = (window_size, commit_size, window_position)

        if cache_key in self._window_structure_cache:
            return self._window_structure_cache[cache_key]

        # Compute window structure
        detector_times = self.detector_time_coords
        window_start = window_position * commit_size
        window_end = window_position * commit_size + window_size - 1

        # Extract detectors within window time range
        window_detector_mask = (detector_times >= window_start) & (
            detector_times <= window_end
        )

        # Extract corresponding rows from H matrix
        H_window_base = self.H[window_detector_mask, :]

        window_structure = {
            "window_detector_mask": window_detector_mask,
            "H_window_base": H_window_base,
            "window_start": window_start,
            "window_end": window_end,
        }

        # Cache the structure
        self._window_structure_cache[cache_key] = window_structure
        return window_structure

    def _get_or_create_window_decoder(
        self, H_window: csc_matrix, p_window: np.ndarray
    ) -> Self:
        """
        Get or create a cached decoder for the given H matrix and priors.

        Parameters
        ----------
        H_window : scipy csc_matrix
            The window parity check matrix.
        p_window : 1D numpy array of float
            The window error probabilities.

        Returns
        -------
        decoder : SoftOutputsBpLsdDecoder
            Cached or newly created decoder.
        """
        # Generate hash key for this configuration
        hash_key = self._hash_matrix_and_priors(H_window, p_window)

        if hash_key in self._decoder_cache:
            return self._decoder_cache[hash_key]

        # Create new decoder
        window_decoder = SoftOutputsBpLsdDecoder(
            H=H_window,
            p=p_window,
            obs_matrix=None,
            **self._bplsd_kwargs,
        )

        # Cache the decoder
        self._decoder_cache[hash_key] = window_decoder
        return window_decoder

    def clear_caches(self) -> None:
        """
        Clear all caches for window structures and decoders.

        This can be useful to free memory or when the decoder configuration changes.
        """
        self._window_structure_cache.clear()
        self._decoder_cache.clear()

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cache usage.

        Returns
        -------
        cache_info : dict
            Dictionary containing cache sizes:
            - window_structures: number of cached window structures
            - decoders: number of cached decoders
        """
        return {
            "window_structures": len(self._window_structure_cache),
            "decoders": len(self._decoder_cache),
        }

    def decode(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
        include_cluster_stats: bool = True,
        compute_logical_gap_proxy: bool = False,
        explore_only_nearby_logical_classes: bool = True,
        verbose: bool = False,
        _benchmarking: bool = False,
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
        _benchmarking : bool
            If True, measure elapsed time for each step and print the outcomes in real time.
            Defaults to False.

        Returns
        -------
        pred : np.ndarray
            Predicted error pattern.
        pred_bp : np.ndarray
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
            - cluster_size_norm_frac_{order} (float): Norm fraction of cluster sizes for each order
            - cluster_llr_norm_frac_{order} (float): Norm fraction of cluster LLRs for each order
        """
        if verbose:
            print("Starting BP+LSD decoding...")

        if _benchmarking:
            start_time = time.time()
            step_start = time.time()

        # Prevent simultaneous use of both options to simplify the implementation
        if compute_logical_gap_proxy:
            include_cluster_stats = False

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        if _benchmarking:
            print(f"[Benchmarking] Input processing: {time.time() - step_start:.6f}s")
            step_start = time.time()

        if verbose:
            print(f"Detector outcomes shape: {detector_outcomes.shape}")
            print(f"Number of violated detectors: {detector_outcomes.sum()}")

        bplsd = self._bplsd
        pred, pred_bp = bplsd.decode(detector_outcomes, return_bp_correction=True)
        pred: np.ndarray = pred.astype(bool)
        pred_bp: np.ndarray = pred_bp.astype(bool)

        if _benchmarking:
            print(f"[Benchmarking] BP+LSD decoding: {time.time() - step_start:.6f}s")
            step_start = time.time()

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

        if _benchmarking:
            soft_info_start = time.time()

        # LLRs
        llrs = self.bit_llrs
        # bp_llrs = np.array(stats["bit_llrs"])
        # bp_llrs_plus = np.clip(bp_llrs, 0.0, None)

        if _benchmarking:
            print(
                f"[Benchmarking] LLRs extraction: {time.time() - soft_info_start:.6f}s"
            )
            soft_info_start = time.time()

        # Prediction LLR
        soft_outputs["pred_llr"] = float(np.sum(llrs[pred]))
        # soft_outputs["pred_bp_llr"] = float(np.sum(bp_llrs[pred]))

        # Detector density
        soft_outputs["detector_density"] = detector_outcomes.sum() / len(
            detector_outcomes
        )

        if _benchmarking:
            print(
                f"[Benchmarking] Basic soft outputs (pred_llr, detector_density): {time.time() - soft_info_start:.6f}s"
            )
            step_start = time.time()

        if verbose:
            print(f"Prediction LLR: {soft_outputs['pred_llr']:.4f}")
            print(f"Detector density: {soft_outputs['detector_density']:.4f}")

        if include_cluster_stats:
            if verbose:
                print("Computing cluster statistics...")

            if _benchmarking:
                cluster_start = time.time()

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

            if _benchmarking:
                print(
                    f"[Benchmarking] Build cluster assignments: {time.time() - cluster_start:.6f}s"
                )
                cluster_start = time.time()

            # Calculate cluster statistics
            cluster_sizes, cluster_llrs = compute_cluster_stats(clusters, llrs)

            if _benchmarking:
                print(
                    f"[Benchmarking] Compute cluster statistics: {time.time() - cluster_start:.6f}s"
                )
                cluster_start = time.time()

            soft_outputs["clusters"] = clusters  # 1D array of int
            soft_outputs["cluster_sizes"] = cluster_sizes
            soft_outputs["cluster_llrs"] = cluster_llrs

            if _benchmarking:
                print(
                    f"[Benchmarking] Store cluster outputs: {time.time() - cluster_start:.6f}s"
                )

            if _benchmarking:
                step_start = time.time()

            if verbose:
                print(f"Number of active clusters: {cluster_id - 1}")
                print(f"Cluster sizes: {cluster_sizes}")

        if compute_logical_gap_proxy:
            if verbose:
                print("Computing logical gap proxy...")

            if _benchmarking:
                gap_start = time.time()

            gap_proxy, best_pred, best_pred_llr = self._compute_logical_gap_proxy(
                detector_outcomes,
                pred,
                soft_outputs["pred_llr"],
                explore_only_nearby_logical_classes,
                verbose=verbose,
            )

            if _benchmarking:
                print(
                    f"[Benchmarking] Logical gap proxy computation: {time.time() - gap_start:.6f}s"
                )
                gap_start = time.time()

            soft_outputs["gap_proxy"] = gap_proxy

            # Update prediction and related soft outputs if a better one was found
            if best_pred_llr < soft_outputs["pred_llr"]:
                if verbose:
                    print(
                        f"  Updating prediction: {soft_outputs['pred_llr']:.4f} -> {best_pred_llr:.4f}"
                    )
                pred = best_pred
                soft_outputs["pred_llr"] = best_pred_llr

            if _benchmarking:
                print(
                    f"[Benchmarking] Update prediction from gap proxy: {time.time() - gap_start:.6f}s"
                )
                step_start = time.time()

        if verbose:
            print("BP+LSD decoding process completed!")

        if _benchmarking:
            print(f"[Benchmarking] Total decode time: {time.time() - start_time:.6f}s")

        return pred, pred_bp, converge, soft_outputs

    def decode_sliding_window(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
        window_size: int,
        commit_size: int,
        verbose: bool = False,
        _benchmarking: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode detector outcomes using (window_size, commit_size)-sliding window method.

        Parameters
        ----------
        detector_outcomes : 1D array-like of bool/int
            Detector measurement outcomes.
        window_size : int
            Number of rounds in each window.
        commit_size : int
            Number of rounds for each commitment.
        verbose : bool, optional
            If True, print progress information. Defaults to False.
        _benchmarking : bool
            If True, measure elapsed time for each step and print the outcomes in real time.
            Defaults to False.

        Returns
        -------
        pred : 1D numpy array of bool
            Predicted error pattern.
        soft_outputs : dict
            Aggregated soft outputs from all windows containing:
            - all_clusters: list of cluster assignments for each window
            - committed_clusters: list of boolean arrays after each window.
            True if fault is committed and in a cluster, False otherwise.
            - committed_faults: list of boolean arrays after each window.
            True if fault is committed, False otherwise.
        """

        self._check_detector_time_coords_validity()

        if window_size <= commit_size:
            raise ValueError("W must be greater than F")

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        if verbose:
            print(
                f"Starting sliding window decoding with W={window_size}, F={commit_size}"
            )

        if _benchmarking:
            start_time = time.time()
            step_start = time.time()
            # Dictionary to track benchmarking times for each step
            benchmark_times = {
                "extract_detector_mask": [],
                "extract_h_matrix_rows": [],
                "find_active_faults": [],
                "extract_submatrices": [],
                "create_decoder": [],
                "decode": [],
                "convert_to_full_size": [],
                "determine_commits": [],
                "update_masks": [],
                "update_detectors_prediction": [],
                "window_total": [],
            }

        # Initialize prediction array
        pred = np.zeros(self.H.shape[1], dtype=bool)

        # Get detector time coordinates
        detector_times = self.detector_time_coords
        max_time = detector_times.max()

        # Storage for aggregated soft outputs
        window_clusters = []

        # Storage for window-wise committed clusters (boolean: True if committed AND in cluster)
        window_committed_clusters = []

        # Storage for window-wise committed faults (boolean: True if committed)
        window_committed_faults = []

        if _benchmarking:
            print(f"[Benchmarking] Initialization: {time.time() - step_start:.6f}s")
            step_start = time.time()

        if verbose:
            print(f"Max detector time: {max_time}")
            print(f"Total detectors: {len(detector_times)}")

        w = 0
        while True:
            if _benchmarking:
                window_start_time = time.time()

            window_start = w * commit_size
            window_end = w * commit_size + window_size - 1

            if verbose:
                print(f"\nWindow {w}: time range [{window_start}, {window_end}]")

            # Check if this is the final window
            is_final_window = window_end >= max_time

            if _benchmarking:
                step_time = time.time()

            # Get cached window structure or compute it
            window_structure = self._get_or_create_window_structure(
                window_size, commit_size, w
            )
            window_detector_mask = window_structure["window_detector_mask"]
            H_window = window_structure["H_window_base"]

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["extract_detector_mask"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Get window structure (cached): {elapsed:.6f}s"
                )
                step_time = time.time()

            if not np.any(window_detector_mask):
                if verbose:
                    print(f"No detectors in window {w}, stopping")
                break

            # Extract detector outcomes for this window
            det_outcomes_window = detector_outcomes[window_detector_mask]

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["extract_h_matrix_rows"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Extract detector outcomes: {elapsed:.6f}s"
                )
                step_time = time.time()

            # Find columns (faults) that have at least one nonzero element
            # Exclude already-committed faults from this window
            fault_mask = np.asarray(H_window.sum(axis=0) > 0).ravel()
            if window_committed_faults:
                # Compute committed faults mask from previous windows
                committed_faults_mask = np.any(window_committed_faults, axis=0)
                fault_mask = fault_mask & ~committed_faults_mask

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["find_active_faults"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Find active faults: {elapsed:.6f}s")
                step_time = time.time()

            if not np.any(fault_mask):
                if verbose:
                    print(f"No active faults in window {w}, skipping")
                w += 1
                continue

            # Extract submatrices
            H_window = H_window[:, fault_mask]
            p_window = self.priors[fault_mask]

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["extract_submatrices"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Extract submatrices: {elapsed:.6f}s"
                )
                step_time = time.time()

            if verbose:
                print(f"Window matrix shape: {H_window.shape}")
                print(f"Active faults: {fault_mask.sum()}")
                print(f"Violated detectors: {det_outcomes_window.sum()}")

            # Get cached decoder or create new one
            window_decoder = self._get_or_create_window_decoder(H_window, p_window)

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["create_decoder"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Get/create decoder (cached): {elapsed:.6f}s"
                )
                step_time = time.time()

            # Decode window
            pred_window_small, _, _, soft_outputs_window = window_decoder.decode(
                det_outcomes_window,
                include_cluster_stats=True,
                compute_logical_gap_proxy=False,
                verbose=False,
            )

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["decode"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Decode: {elapsed:.6f}s")
                step_time = time.time()

            # Convert window prediction to full size
            pred_window = np.zeros(self.H.shape[1], dtype=bool)
            pred_window[fault_mask] = pred_window_small

            # Convert clusters to full size
            clusters_window = np.zeros(self.H.shape[1], dtype=int)
            clusters_window[fault_mask] = soft_outputs_window["clusters"]

            # Store window soft outputs
            window_clusters.append(clusters_window)

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["convert_to_full_size"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Convert to full size: {elapsed:.6f}s"
                )
                step_time = time.time()

            # Determine which faults to commit
            if is_final_window:
                # Final window: commit all faults
                pred_to_commit = pred_window
                commit_mask = fault_mask
                if verbose:
                    print("Final window: committing all faults")
            else:
                # Regular window: commit only faults involved in detectors within [w*F, w*F+F-1]
                commit_start = w * commit_size
                commit_end = w * commit_size + commit_size - 1
                commit_detector_mask = (detector_times >= commit_start) & (
                    detector_times <= commit_end
                )

                if np.any(commit_detector_mask):
                    # Find faults involved in commit region detectors
                    H_commit_rows = self.H[commit_detector_mask, :]
                    commit_mask = np.asarray(H_commit_rows.sum(axis=0) > 0).ravel()
                    # Exclude already committed faults from commit region
                    if window_committed_faults:
                        committed_faults_mask = np.any(window_committed_faults, axis=0)
                        commit_mask &= ~committed_faults_mask

                    pred_to_commit = pred_window.copy()
                    pred_to_commit[~commit_mask] = False
                else:
                    pred_to_commit = np.zeros(self.H.shape[1], dtype=bool)
                    commit_mask = np.zeros(self.H.shape[1], dtype=bool)

                if verbose:
                    print(f"Commit region: [{commit_start}, {commit_end}]")
                    print(f"Committing {pred_to_commit.sum()} faults")

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["determine_commits"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Determine commits: {elapsed:.6f}s")
                step_time = time.time()

            # No need to maintain separate committed faults mask - computed from window_committed_faults when needed

            # Track committed clusters (boolean: True if committed AND in cluster)
            committed_clusters_current_window = commit_mask & (clusters_window > 0)

            # Track committed faults (boolean: True if committed)
            committed_faults_current_window = commit_mask.copy()

            # Store both arrays
            window_committed_clusters.append(committed_clusters_current_window)
            window_committed_faults.append(committed_faults_current_window)

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["update_masks"].append(elapsed)
                print(f"[Benchmarking] Window {w} - Update masks: {elapsed:.6f}s")
                step_time = time.time()

            # Update detector outcomes and prediction
            detector_update = ((pred_to_commit.astype(np.uint8) @ self.H.T) % 2).astype(
                bool
            )
            detector_outcomes ^= detector_update
            pred ^= pred_to_commit

            if _benchmarking:
                elapsed = time.time() - step_time
                benchmark_times["update_detectors_prediction"].append(elapsed)
                print(
                    f"[Benchmarking] Window {w} - Update detectors/prediction: {elapsed:.6f}s"
                )

            if verbose:
                print(f"Updated {detector_update.sum()} detector outcomes")
                print(f"Total prediction weight: {pred.sum()}")
                print(f"Remaining violated detectors: {detector_outcomes.sum()}")

            if _benchmarking:
                window_elapsed = time.time() - window_start_time
                benchmark_times["window_total"].append(window_elapsed)
                print(f"[Benchmarking] Window {w} total: {window_elapsed:.6f}s")

            # Break if final window
            if is_final_window:
                break

            w += 1

        # Create aggregated soft outputs
        soft_outputs = {
            "all_clusters": window_clusters,
            "committed_clusters": window_committed_clusters,
            "committed_faults": window_committed_faults,
        }

        if verbose:
            print(f"\nSliding window decoding completed!")
            print(f"Total windows processed: {len(window_clusters)}")
            print(f"Final prediction weight: {pred.sum()}")

        if _benchmarking:
            total_time = time.time() - start_time
            print(f"[Benchmarking] Total sliding window decode time: {total_time:.6f}s")

            # Print aggregated benchmark summary
            print("\n[Benchmarking] ========== SUMMARY ACROSS ALL WINDOWS ==========")
            print(f"Total windows processed: {len(benchmark_times['window_total'])}")
            print("\nStep-by-step breakdown (mean ± std) [total]:\n")

            for step_name, times in benchmark_times.items():
                if times:  # Only show steps that were actually executed
                    times_array = np.array(times)
                    mean_time = np.mean(times_array)
                    std_time = np.std(times_array)
                    total_step_time = np.sum(times_array)
                    print(
                        f"  {step_name:<30}: {mean_time:.6f} ± {std_time:.6f}s [total: {total_step_time:.6f}s]"
                    )

            print("\n[Benchmarking] ===============================================")

        return pred, soft_outputs

    def simulate_single(self, sliding_window=False, seed=None, **kwargs):
        if seed is not None:
            rng = np.random.default_rng(seed)
            errors = rng.random(self.H.shape[1], dtype=np.float64) < self.priors
        else:
            errors = np.random.random(self.H.shape[1]) < self.priors

        det_outcomes = (errors @ self.H.T % 2).astype(bool)

        if sliding_window:
            pred, soft_outputs = self.decode_sliding_window(det_outcomes, **kwargs)
        else:
            pred, _, _, soft_outputs = self.decode(det_outcomes, **kwargs)

        residual = pred ^ errors
        valid = not bool(np.any(residual @ self.H.T % 2))
        if not valid:
            raise ValueError("Decoding outcome invalid")

        fail = bool(np.any((residual @ self.obs_matrix.T) % 2))

        return fail, soft_outputs
