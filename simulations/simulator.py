import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymatching as pm
import stim
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix, vstack
from statsmodels.stats.proportion import proportion_confint

from src.ldpc_post_selection.decoder import BpLsdPsDecoder
from src.ldpc_post_selection.stim_tools import dem_to_parity_check


def set_df_dtypes(df: pd.DataFrame, dtypes: List[Any]):
    """
    Set the data types for columns in a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    dtypes : list
        A list of data types to apply to the columns of df in order.
    """
    for col, dtype in zip(df.columns, dtypes):
        df[col] = df[col].astype(dtype)


class BpLsdPsSimulator:
    """
    Runs Monte Carlo (OMC) simulations for BpLsdPsDecoder and optionally MWPM.

    This class encapsulates the logic for simulating the BpLsdPsDecoder, using standard
    Monte Carlo sampling. It handles circuit initialization, data sampling, parallelized
    decoding, result aggregation, and post-selection performance analysis.
    """

    circuit: Optional[stim.Circuit]
    H: csc_matrix
    obs: csc_matrix
    p: np.ndarray
    dem: stim.DetectorErrorModel
    bplsd_prms: Dict[str, Any]
    aggregate_results: bool
    # Results storage
    # results: Optional[pd.DataFrame] = None  # Stores individual shot results
    results: Optional[pd.DataFrame] = None  # Stores results
    _last_run_params: Dict[str, Any] = {}  # Stores params of the last run

    def __init__(
        self,
        *,
        circuit: Optional[stim.Circuit] = None,
        H: Optional[csc_matrix | np.ndarray | List[List[int | bool]]] = None,
        obs: Optional[csc_matrix | np.ndarray | List[List[int | bool]]] = None,
        p: Optional[np.ndarray] = None,
        max_iter: int = 30,
        bp_method: str = "product_sum",
        lsd_method: str = "LSD_CS",
        lsd_order: int = 0,
        aggregate_results: bool = True,
        verbose: int = 0,
    ):
        """
        Initialize the Monte Carlo Simulator.

        Parameters
        ----------
        circuit : stim.Circuit, optional
            The quantum circuit model. If provided, H, obs, and p_priors are derived from it.
            Overrides H, obs, p_priors if given.
        H : csc_matrix, optional
            Parity check matrix. Required if circuit is None.
        obs : csc_matrix, optional
            Logical observables matrix. Required if circuit is None.
        p_priors : np.ndarray, optional
            Prior error probabilities for each fault. Required if circuit is None.
        max_iter : int, optional
            Maximum iterations for the BP part of the decoder. Defaults to 30.
        bp_method : str, optional
            Method for BP message updates ('product_sum' or 'minimum_sum'). Defaults to "minimum_sum".
        lsd_method : str, optional
            Method for the LSD part ('LSD_0', 'LSD_E', 'LSD_CS'). Defaults to "LSD_0".
        lsd_order : int, optional
            Order parameter for LSD. Defaults to 0.
        verbose : int, optional
            Verbosity level for initialization. Defaults to 0.

        Raises
        ------
        ValueError
            If neither circuit nor all of H, obs, p_priors are provided.
        """
        self.circuit = circuit
        if circuit is not None:
            dem = circuit.detector_error_model()
            self.H, self.obs, self.p = dem_to_parity_check(dem)
        elif H is not None and obs is not None and p is not None:
            if not isinstance(H, csc_matrix):
                H = csc_matrix(H)
            if not isinstance(obs, csc_matrix):
                obs = csc_matrix(obs)
            self.H = H.astype(bool)
            self.obs = obs.astype(bool)
            self.p = np.asanyarray(p, dtype="float64")

        else:
            raise ValueError(
                "Either 'circuit' or 'H', 'obs', and 'p_priors' must be provided."
            )

        self.bplsd_prms = {
            "max_iter": max_iter,
            "bp_method": bp_method,
            "lsd_method": lsd_method,
            "lsd_order": lsd_order,
        }

        self.results = pd.DataFrame(dtype="float64")
        self.aggregate_results = aggregate_results

        if verbose:
            print("MonteCarloSimulator Initialized:")
            print("  Number of checks:", self.H.shape[0])
            print("  Number of faults:", self.H.shape[1])
            print("  Number of observables:", self.obs.shape[0])

    def _sample(
        self, shots: int, seed: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample detection events and observable flips from the detector error model.

        Parameters
        ----------
        shots : int
            Number of samples to generate.
        seed : int | None, optional
            Seed for the random number generator. Defaults to None.

        Returns
        -------
        tuple
            Contains (det_data, obs_data, det_density_all):
            det_data : np.ndarray
                Detection event data for each shot.
            obs_data : np.ndarray
                Observable flip data for each shot.
            det_density_all : np.ndarray
                Detector density for each shot.
        """
        if self.circuit is None:
            if seed is not None:
                np.random.seed(seed)
            error = np.random.rand(shots, self.H.shape[1]) < self.p.reshape(1, -1)
            det_data = (error.astype("uint8") @ self.H.T) % 2
            obs_data = (error.astype("uint8") @ self.obs.T) % 2

        else:
            sampler = self.circuit.compile_detector_sampler()
            det_data, obs_data = sampler.sample(shots, separate_observables=True)

        det_density_all = (
            det_data.sum(axis=1) / det_data.shape[1]
            if det_data.shape[1] > 0
            else np.zeros(det_data.shape[0])
        )
        return det_data, obs_data, det_density_all

    def _prepare_mwpm_decoder(
        self,
    ) -> Tuple[csc_matrix, pm.Matching, np.ndarray, np.ndarray]:
        """
        Prepare the parity check matrix and Matching object for MWPM decoding with boundary.

        Uses the instance attributes `self.H`, `self.obs`, `self.p_priors`.

        Returns
        -------
        tuple
            Contains (H_with_bdry, matching, bdry_errors, weights):
            H_with_bdry : csc_matrix
                Parity check matrix augmented with a boundary check row.
            matching : pm.Matching
                PyMatching object initialized with H_with_bdry and weights.
            bdry_errors : np.ndarray
                Indices of the boundary errors (faults corresponding to the observable).
            weights : np.ndarray
                Edge weights for the matching graph.
        """
        bdry_errors = self.obs.nonzero()[1]
        bdry_check = csc_matrix(
            ([1] * len(bdry_errors), ([0] * len(bdry_errors), bdry_errors)),
            shape=(1, self.H.shape[1]),
        )
        H_with_bdry = vstack([self.H, bdry_check])
        weights = np.log((1 - self.p) / self.p)
        matching = pm.Matching(H_with_bdry, weights=weights)
        return H_with_bdry, matching, bdry_errors, weights

    def _run_mwpm_decoding(
        self,
        matching: pm.Matching,
        det_data_batch: np.ndarray,
        obs_data_batch: np.ndarray,
        bdry_errors: np.ndarray,
        d: int,
        p: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run MWPM decoding for a batch and compute logical gaps.

        Parameters
        ----------
        matching : pm.Matching
            Initialized PyMatching object.
        det_data_batch : np.ndarray
            Detection event data for the batch.
        obs_data_batch : np.ndarray
            Observable flip data for the batch.
        bdry_errors : np.ndarray
            Indices of boundary errors.
        d : int
            Code distance.
        p : float
            Physical error rate.

        Returns
        -------
        tuple
            Contains (fails, logical_gaps):
            fails : np.ndarray
                Boolean array indicating MWPM logical failures for each sample.
            logical_gaps : np.ndarray
                Computed logical gaps normalized by d * log((1-p)/p).
        """
        pred_weights_both = []
        fails_both = []
        for bdry_value in [False, True]:
            det_data_with_bdry = np.hstack(
                [det_data_batch, np.full((det_data_batch.shape[0], 1), bdry_value)]
            )
            preds, pred_weights = matching.decode_batch(
                det_data_with_bdry, return_weights=True
            )
            obs_preds = preds[:, bdry_errors].sum(axis=1) % 2
            batch_fails = (obs_preds ^ obs_data_batch.ravel()).astype("bool")
            pred_weights_both.append(pred_weights)
            fails_both.append(batch_fails)

        logical_gaps = np.abs(pred_weights_both[0] - pred_weights_both[1])
        # Avoid division by zero if p is 0 or 1, although typically p is small
        log_ratio = np.log((1 - p) / p) if 0 < p < 1 else np.inf
        norm_factor = d * log_ratio
        logical_gaps /= (
            norm_factor if norm_factor != 0 else 1.0
        )  # Handle potential zero division

        fails = np.where(
            pred_weights_both[0] <= pred_weights_both[1],
            fails_both[0],
            fails_both[1],
        )
        return fails, logical_gaps

    def _process_batch(
        self,
        det_data_batch: np.ndarray,
        obs_data_batch: np.ndarray,
        det_density_batch: np.ndarray,
        Q_digits: int,
        get_logical_gaps: bool,
        d: int | None,
        p: float | None,
        matching: pm.Matching | None,
        bdry_errors: np.ndarray | None,
    ) -> pd.DataFrame:
        """
        Process a batch of simulation samples using BPLSD and optionally MWPM.

        Uses instance attributes for decoder configuration (H, p_priors, max_iter, etc.).

        Parameters
        ----------
        det_data_batch : np.ndarray
            Detection event data for the batch.
        obs_data_batch : np.ndarray
            Observable flip data for the batch.
        det_density_batch : np.ndarray
            Detector density data for the batch.
        Q_digits : int
            Quantization digits for aggregation.
        get_logical_gaps : bool
            Whether to compute logical gaps with MWPM.
        d : int | None
            Code distance (needed for MWPM).
        p : float | None
            Physical error rate (needed for MWPM).
        matching : pm.Matching | None
            Pre-initialized PyMatching object if get_logical_gaps is True.
        bdry_errors : np.ndarray | None
            Indices of boundary errors if get_logical_gaps is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing simulation results for the batch.
        """
        bp_prms = self.bplsd_prms
        bplsd = BpLsdPsDecoder(
            self.H,
            p=self.p,
            max_iter=bp_prms["max_iter"],
            bp_method=bp_prms["bp_method"],
            lsd_method=bp_prms["lsd_method"],
            lsd_order=bp_prms["lsd_order"],
        )

        get_aggr_data = self.aggregate_results

        # Prepare output DataFrame structure
        if get_aggr_data:
            strategies = ["cf", "dd"]
            if get_logical_gaps:
                raise NotImplementedError(
                    "Aggregated data with logical gaps not implemented yet."
                )
            cols = ["Q"]
            for strat in strategies:
                cols.extend([f"num_samples_{strat}", f"num_fails_{strat}"])
            output = pd.DataFrame(columns=cols, dtype="float64").set_index("Q")
        else:
            soft_info_keys = list(bplsd.soft_info_dtypes.keys()) + ["det_density"]
            soft_info_dtypes = list(bplsd.soft_info_dtypes.values()) + ["float64"]
            output = pd.DataFrame(columns=soft_info_keys)
            set_df_dtypes(output, soft_info_dtypes)

        # Process samples in the batch
        for i, (det_vals, obs_vals, det_density) in enumerate(
            zip(det_data_batch, obs_data_batch, det_density_batch)
        ):
            error_preds, soft_info = bplsd.decode(det_vals)
            # Calculate logical prediction based on the first observable (assuming single observable for now)
            obs_preds = ((error_preds.astype("uint8") @ self.obs.T) % 2).astype(bool)
            fail = np.any(
                obs_vals ^ obs_preds
            )  # Check against all ground truth obs flips

            if get_aggr_data:
                factor = 10**Q_digits
                cluster_frac = soft_info["cluster_size_sum"] / self.H.shape[1]
                cluster_frac_q = np.ceil(cluster_frac * factor) / factor
                det_density_q = np.ceil(det_density * factor) / factor
                Qs = {"cf": cluster_frac_q, "dd": det_density_q}

                for strat, Q in Qs.items():
                    col_num_samples = f"num_samples_{strat}"
                    col_num_fails = f"num_fails_{strat}"

                    # Check if index Q exists, initialize if not
                    if Q not in output.index:
                        output.loc[Q, [col_num_samples, col_num_fails]] = [0, 0]
                    # Increment counts
                    output.loc[Q, col_num_samples] += 1
                    if fail:
                        output.loc[Q, col_num_fails] += 1

            else:
                soft_info["det_density"] = det_density

        if get_aggr_data:
            output = output.fillna(0).astype("uint64")

        # Run MWPM if requested
        if get_logical_gaps:
            if d is None or p is None or matching is None or bdry_errors is None:
                raise ValueError("Internal error: MWPM dependencies not met.")
            mwpm_fails, logical_gaps = self._run_mwpm_decoding(
                matching, det_data_batch, obs_data_batch, bdry_errors, d, p
            )
            if get_aggr_data:
                # Placeholder: Add logic here if/when aggregated gap analysis is implemented
                pass
            else:
                output["fail_mwpm"] = mwpm_fails
                output["logical_gap"] = logical_gaps

        # Set final dtypes for non-aggregated data
        if not get_aggr_data:
            final_dtypes = ["bool", "float64", "uint64"]
            if get_logical_gaps:
                final_dtypes.extend(["bool", "float64"])
            set_df_dtypes(output, final_dtypes)

        return output

    def run(
        self,
        shots: int | float,
        *,
        get_logical_gaps: bool = False,
        d: int | None = None,
        p: float | None = None,
        Q_digits: int = 2,
        n_jobs: int = -1,
        n_batch: int | None = None,
        seed: int | None = None,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """
        Run the Monte Carlo simulation.

        Generates samples, performs decoding (potentially in parallel),
        and aggregates results. Stores results in `self.results` or
        `self.aggregated_results`.

        Parameters
        ----------
        shots : int | float
            The number of simulation shots (samples) to run.
        get_logical_gaps : bool, optional
            If True, computes logical gaps using MWPM (requires d and p).
            Defaults to False.
        d : int | None, optional
            Code distance, required if get_logical_gaps is True. Defaults to None.
        p : float | None, optional
            Physical error rate used for MWPM weights, required if get_logical_gaps is True.
            If None, uses the average of `self.p_priors`. Defaults to None.
        Q_digits : int, optional
            Number of digits for quantization if aggregate_results is True. Defaults to 2.
        n_jobs : int, optional
            Number of parallel jobs (-1 for all cores). Defaults to -1.
        n_batch : int | None, optional
            Number of batches to split the shots into for parallel processing.
            If None, automatically determined based on shots and n_jobs.
            Should be >= n_jobs if specified manually. Defaults to None.
        seed : int | None, optional
            Seed for the random number generator. Defaults to None.
        verbose : int, optional
            Verbosity level for parallel processing. Defaults to 0.

        Returns
        -------
        pd.DataFrame
            The simulation results, either aggregated or per-shot. Also stored
            in `self.aggregated_results` or `self.results`.

        Raises
        ------
        ValueError
            If `get_logical_gaps` is True but `d` is not provided.
            If `self.aggregate_results` is True and `get_logical_gaps` is True (not implemented).
        """
        shots = int(shots)
        self._last_run_params = locals()  # Store parameters for potential later use
        get_aggr_data = self.aggregate_results

        # 1. Sampling
        det_data, obs_data, det_density_all = self._sample(shots, seed=seed)

        # 2. Prepare MWPM decoder if needed
        matching = None
        bdry_errors = None
        if get_logical_gaps:
            if d is None:
                raise ValueError(
                    "Code distance 'd' must be provided when get_logical_gaps=True"
                )
            if p is None:
                # Estimate a single p if not provided, for MWPM weighting consistency
                p = np.mean(self.p)
                if verbose > 0:
                    print(
                        f"Warning: Physical error rate 'p' not provided for MWPM. Using average of priors: {p:.4f}"
                    )
            if get_aggr_data:
                raise NotImplementedError(
                    "Aggregated data with logical gaps not implemented yet."
                )

            _, matching, bdry_errors, _ = self._prepare_mwpm_decoder()

        # 3. Parallel Processing Setup
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        n_jobs = min(n_jobs, shots)  # Cannot use more jobs than shots

        if n_batch is None:
            # Determine a reasonable number of batches
            # Aim for at least 1 batch per job, but avoid tiny batches
            min_batch_size = 100  # Heuristic minimum batch size
            batches_for_jobs = n_jobs
            batches_for_size = (shots + min_batch_size - 1) // min_batch_size
            n_batch = max(batches_for_jobs, batches_for_size)
            n_batch = min(n_batch, shots)  # Cannot have more batches than shots
        elif n_batch < n_jobs:
            if verbose > 0:
                print(
                    f"Warning: n_batch ({n_batch}) is less than n_jobs ({n_jobs}). "
                    f"Reducing n_jobs to {n_batch} for efficiency."
                )
            n_jobs = n_batch  # Can't effectively use more jobs than batches
        n_batch = min(n_batch, shots)  # Ensure n_batch is not greater than shots

        process_args = {
            "Q_digits": Q_digits,
            "get_logical_gaps": get_logical_gaps,
            "d": d,
            "p": p,
            "matching": matching,
            "bdry_errors": bdry_errors,
        }

        # Split data for batches
        det_data_split = np.array_split(det_data, n_batch, axis=0)
        obs_data_split = np.array_split(obs_data, n_batch, axis=0)
        det_density_split = np.array_split(det_density_all, n_batch, axis=0)

        # Run parallel processing
        outputs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self._process_batch)(
                det_data_batch=det_split,
                obs_data_batch=obs_split,
                det_density_batch=density_split,
                **process_args,
            )
            for det_split, obs_split, density_split in zip(
                det_data_split, obs_data_split, det_density_split
            )
        )

        # 4. Aggregate Results
        if get_aggr_data:
            final_results = pd.concat(outputs, axis=0).groupby(level=0).sum()
            final_results.sort_index(inplace=True)
            self.results = final_results
            self.results = None  # Clear individual results if aggregated
        else:
            final_results = pd.concat(outputs, axis=0).reset_index(drop=True)
            final_results["det_density"] = det_density_all  # Add original density back
            self.results = final_results
            self.results = None  # Clear aggregated results

        return final_results

    def compute_ps_performance(
        self,
        clist: np.ndarray,
        strat: str | list[str] = "all",
        alpha: float = 0.05,
        verbose: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Compute post-selection performance metrics from the latest simulation results.

        Analyzes the individual simulation results stored in `self.results`
        to calculate post-selected failure rates and abortion rates for
        various strategies and thresholds. Requires `run` to have been called
        with `get_aggr_data=False`.

        Parameters
        ----------
        clist : np.ndarray
            An array of post-selection thresholds (values between 0 and 1).
            For 'cluster_frac' and 'det_density', samples with values <= (1-c) are kept.
            For 'logical_gap', samples with values >= c are kept.
        strat : str or list[str], optional
            The post-selection strategy/strategies to evaluate. Can be
            'cluster_frac', 'det_density', 'logical_gap', or 'all' to evaluate all
            available strategies in `self.results`. Defaults to "all".
        alpha : float, optional
            Significance level for confidence intervals (e.g., 0.05 for 95% CI). Defaults to 0.05.
        verbose : bool, optional
            If True, prints the current strategy being processed. Defaults to False.

        Returns
        ------
        dict[str, pd.DataFrame]
            A dictionary where keys are the evaluated strategy names and values
            are DataFrames. Each DataFrame is indexed by the threshold 'c' and
            contains columns:
            'fail': Post-selected logical failure rate estimate.
            'delta_fail': Half-width of the confidence interval for 'fail'.
            'abort': Abortion rate estimate.
            'delta_abort': Half-width of the confidence interval for 'abort'.

        Raises
        ------
        ValueError
            If `run` has not been executed yet or was run with `get_aggr_data=True`.
            If a specified strategy is not available in the results or is invalid.
        """
        if self.results is None:
            raise ValueError(
                "No simulation results available. Call run() first with get_aggr_data=False."
            )
        if self.results is not None:
            raise ValueError(
                "Cannot compute post-selection performance on aggregated data. Run simulation with get_aggr_data=False."
            )

        df = self.results  # Use the stored individual results
        clist = np.asanyarray(clist)

        possible_strats = ["cluster_frac", "det_density"]
        if "logical_gap" in df.columns:
            possible_strats.append("logical_gap")

        valid_available_strats = [s for s in possible_strats if s in df.columns]

        if strat == "all":
            eval_strats = valid_available_strats
        else:
            eval_strats = [strat] if isinstance(strat, str) else strat
            for s in eval_strats:
                if s not in valid_available_strats:
                    raise ValueError(
                        f"Strategy '{s}' not available in results or not a valid strategy."
                        f" Available strategies in results: {valid_available_strats}"
                    )

        num_samples = len(df)
        outputs = {}

        for current_strat in eval_strats:
            if verbose:
                print("Processing strategy:", current_strat)

            # Sort based on strategy: ascending for fractions/density, descending for gap
            ascending_sort = current_strat != "logical_gap"
            df_sorted = df.sort_values(
                current_strat, axis=0, ascending=ascending_sort
            ).reset_index(drop=True)

            Q = df_sorted[current_strat].values
            fails_sorted = df_sorted["fail"].values

            # Determine indices corresponding to thresholds
            thresholds_to_search = (1 - clist) if ascending_sort else clist
            inds_c = np.searchsorted(Q, thresholds_to_search, side="right")

            # Calculate cumulative counts
            num_accepted = inds_c
            num_fails_cumsum = fails_sorted.cumsum()

            # Initialize result arrays
            fail_rate = np.full_like(clist, np.nan, dtype=float)
            delta_fail = np.full_like(clist, np.nan, dtype=float)
            abort_rate = np.full_like(clist, np.nan, dtype=float)
            delta_abort = np.full_like(clist, np.nan, dtype=float)

            valid_inds_mask = (
                num_accepted > 0
            )  # Only calculate where samples are accepted
            if np.any(valid_inds_mask):
                num_accepted_valid = num_accepted[valid_inds_mask]
                # Index for cumulative sum access (need index k-1 for sum up to k)
                inds_c_cumsum = np.maximum(0, num_accepted_valid - 1)
                num_fails_at_threshold = num_fails_cumsum[inds_c_cumsum]

                # Calculate confidence intervals for failure rate
                fail_low, fail_upp = proportion_confint(
                    num_fails_at_threshold,
                    num_accepted_valid,
                    alpha=alpha,
                    method="binom_test",
                )
                fail_rate[valid_inds_mask] = (fail_low + fail_upp) / 2
                delta_fail[valid_inds_mask] = fail_upp - fail_rate[valid_inds_mask]

                # Calculate confidence intervals for abortion rate
                num_aborted = num_samples - num_accepted_valid
                abort_low, abort_upp = proportion_confint(
                    num_aborted, num_samples, alpha=alpha, method="binom_test"
                )
                abort_rate[valid_inds_mask] = (abort_low + abort_upp) / 2
                delta_abort[valid_inds_mask] = abort_upp - abort_rate[valid_inds_mask]

            # Handle cases where no samples are accepted (fail=0, abort=1)
            no_accepted_mask = ~valid_inds_mask
            if np.any(no_accepted_mask):
                fail_rate[no_accepted_mask] = 0.0
                delta_fail[no_accepted_mask] = 0.0
                abort_rate[no_accepted_mask] = 1.0
                delta_abort[no_accepted_mask] = 0.0

            df_ps = pd.DataFrame(
                {
                    "c": clist,
                    "fail": fail_rate,
                    "delta_fail": delta_fail,
                    "abort": abort_rate,
                    "delta_abort": delta_abort,
                }
            )
            df_ps = df_ps.set_index("c").sort_index()

            outputs[current_strat] = df_ps

        return outputs

    @property
    def num_samples(self) -> int:
        """Return the number of samples from the last run."""
        if self.results is not None:
            return len(self.results)
        elif self.results is not None:
            # Sum samples across all bins for one strategy (e.g., 'cf')
            sample_cols = [
                col for col in self.results.columns if col.startswith("num_samples_")
            ]
            if sample_cols:
                return self.results[sample_cols[0]].sum()
            else:
                return 0  # Should not happen if run was successful
        else:
            return 0
