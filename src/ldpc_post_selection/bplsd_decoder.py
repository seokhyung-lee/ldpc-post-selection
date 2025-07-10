from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix, vstack

from .base import SoftOutputsDecoder
from .utils import compute_cluster_stats, label_clusters


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

    def decode_sliding_window(
        self,
        detector_outcomes: np.ndarray | List[bool | int],
        window_size: int,
        commit_size: int,
        verbose: bool = False,
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

        Returns
        -------
        pred : 1D numpy array of bool
            Predicted error pattern.
        soft_outputs : dict
            Aggregated soft outputs from all windows containing:
            - clusters: list of cluster assignments for each window
            - cluster_sizes: list of cluster sizes for each window
            - cluster_llrs: list of cluster LLRs for each window
            - committed_clusters: 1D numpy array of int with cluster labels for committed faults (0 = not in cluster, 1+ = cluster ID)
            - committed_cluster_sizes: 1D numpy array of int with sizes of committed clusters
            - committed_cluster_llrs: 1D numpy array of float with LLR sums of committed clusters
        """
        if window_size <= commit_size:
            raise ValueError("W must be greater than F")

        detector_outcomes = np.asarray(detector_outcomes, dtype=bool)
        if detector_outcomes.ndim > 1:
            raise ValueError("Detector outcomes must be a 1D array")

        if verbose:
            print(
                f"Starting sliding window decoding with W={window_size}, F={commit_size}"
            )

        # Initialize prediction array
        pred = np.zeros(self.H.shape[1], dtype=bool)

        # Get detector time coordinates
        detector_times = self.detector_time_coords
        max_time = detector_times.max()

        # Storage for aggregated soft outputs
        window_clusters = []
        window_cluster_sizes = []
        window_cluster_llrs = []
        committed_clusters_mask = np.zeros(self.H.shape[1], dtype=bool)

        if verbose:
            print(f"Max detector time: {max_time}")
            print(f"Total detectors: {len(detector_times)}")

        w = 0
        while True:
            window_start = w * commit_size
            window_end = w * commit_size + window_size - 1

            if verbose:
                print(f"\nWindow {w}: time range [{window_start}, {window_end}]")

            # Check if this is the final window
            is_final_window = window_end >= max_time

            # Extract detectors within window time range
            window_detector_mask = (detector_times >= window_start) & (
                detector_times <= window_end
            )

            if not np.any(window_detector_mask):
                if verbose:
                    print(f"No detectors in window {w}, stopping")
                break

            # Extract detector outcomes for this window
            det_outcomes_window = detector_outcomes[window_detector_mask]

            # Extract corresponding rows from H matrix
            H_window_rows: csc_matrix = self.H[window_detector_mask, :]

            # Find columns (faults) that have at least one nonzero element
            fault_mask = np.asarray(H_window_rows.sum(axis=0) > 0).flatten()

            if not np.any(fault_mask):
                if verbose:
                    print(f"No active faults in window {w}, skipping")
                w += 1
                continue

            # Extract submatrices
            H_window = H_window_rows[:, fault_mask]
            p_window = self.priors[fault_mask]

            if verbose:
                print(f"Window matrix shape: {H_window.shape}")
                print(f"Active faults: {fault_mask.sum()}")
                print(f"Violated detectors: {det_outcomes_window.sum()}")

            # Create new decoder for this window
            window_decoder = SoftOutputsBpLsdDecoder(
                H=H_window,
                p=p_window,
                obs_matrix=None,
                **self._bplsd_kwargs,
            )

            # Decode window
            pred_window_small, _, _, soft_outputs_window = window_decoder.decode(
                det_outcomes_window,
                include_cluster_stats=True,
                compute_logical_gap_proxy=False,
                verbose=False,
            )

            # Convert window prediction to full size
            pred_window = np.zeros(self.H.shape[1], dtype=bool)
            pred_window[fault_mask] = pred_window_small

            # Convert clusters to full size
            clusters_window = np.zeros(self.H.shape[1], dtype=int)
            clusters_window[fault_mask] = soft_outputs_window["clusters"]

            # Store window soft outputs
            window_clusters.append(clusters_window)
            window_cluster_sizes.append(soft_outputs_window["cluster_sizes"])
            window_cluster_llrs.append(soft_outputs_window["cluster_llrs"])

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
                    commit_fault_mask = np.asarray(
                        H_commit_rows.sum(axis=0) > 0
                    ).flatten()

                    pred_to_commit = pred_window.copy()
                    pred_to_commit[~commit_fault_mask] = False
                    commit_mask = commit_fault_mask
                else:
                    pred_to_commit = np.zeros(self.H.shape[1], dtype=bool)
                    commit_mask = np.zeros(self.H.shape[1], dtype=bool)

                if verbose:
                    print(f"Commit region: [{commit_start}, {commit_end}]")
                    print(f"Committing {pred_to_commit.sum()} faults")

            # Update committed clusters mask
            committed_clusters_window = clusters_window.copy()
            committed_clusters_window[~commit_mask] = 0
            committed_clusters_mask[committed_clusters_window > 0] = True

            # Update detector outcomes and prediction
            detector_update = ((pred_to_commit.astype(np.uint8) @ self.H.T) % 2).astype(
                bool
            )
            detector_outcomes ^= detector_update
            pred ^= pred_to_commit

            if verbose:
                print(f"Updated {detector_update.sum()} detector outcomes")
                print(f"Total prediction weight: {pred.sum()}")

            # Break if final window
            if is_final_window:
                break

            w += 1

        # Convert committed clusters mask to cluster index array using label_clusters
        adj_matrix = (self.H.T @ self.H == 1).astype(bool)
        vertices_inside_clusters = np.where(committed_clusters_mask)[0]
        committed_clusters_idx = label_clusters(adj_matrix, vertices_inside_clusters)

        # Compute committed cluster statistics
        committed_cluster_sizes, committed_cluster_llrs = compute_cluster_stats(
            committed_clusters_idx, self.bit_llrs
        )

        # Create aggregated soft outputs
        soft_outputs = {
            "clusters": window_clusters,
            "cluster_sizes": window_cluster_sizes,
            "cluster_llrs": window_cluster_llrs,
            "committed_clusters": committed_clusters_idx,
            "committed_cluster_sizes": committed_cluster_sizes,
            "committed_cluster_llrs": committed_cluster_llrs,
        }

        if verbose:
            print(f"\nSliding window decoding completed!")
            print(f"Total windows processed: {len(window_clusters)}")
            print(f"Final prediction weight: {pred.sum()}")
            print(
                f"Committed clusters: {(committed_clusters_idx > 0).sum()} faults in {committed_clusters_idx.max()} clusters"
            )
            print(f"Committed cluster sizes: {committed_cluster_sizes}")
            print(f"Committed cluster LLRs: {committed_cluster_llrs}")

        return pred, soft_outputs

    def simulate_single(self, sliding_window=False, seed=None, **kwargs):
        rng = np.random.default_rng(seed)
        errors = rng.random(self.H.shape[1], dtype="float64") < self.priors
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
