import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatching as pm
import seaborn as sns
import stim
from joblib import Parallel, delayed
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csc_matrix, vstack
from statsmodels.stats.proportion import proportion_confint
from typing import Tuple, List, Dict, Any, Self
import time
import pickle
import arviz as az
from datetime import datetime
from copy import deepcopy

from SlidingWindowDecoder.src.build_circuit import build_circuit, \
    dem_to_check_matrices
from SlidingWindowDecoder.src.codes_q import create_bivariate_bicycle_codes


def build_circuit_surface_code(d: int,
                               T: int,
                               *,
                               p_bitflip: float = 0.,
                               p_circuit: float = 0.):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=T,
        after_clifford_depolarization=p_circuit,
        before_round_data_depolarization=p_bitflip * 1.5,
        before_measure_flip_probability=p_circuit,
        after_reset_flip_probability=p_circuit
    )
    return circuit


def build_circuit_BB(n, T, p):
    if n == 72:  # d=6
        args = 6, 6, [3], [1, 2], [1, 2], [3]
    elif n == 90:  # d=10
        args = 15, 3, [9], [1, 2], [2, 7], [0]
    elif n == 108:  # d=10
        args = 9, 6, [3], [1, 2], [1, 2], [3]
    elif n == 144:  # d=12
        args = 12, 6, [3], [1, 2], [1, 2], [3]
    elif n == 288:  # d=18
        args = 12, 12, [3], [2, 7], [1, 2], [3]
    elif n == 360:  # d<=24
        args = 30, 6, [9], [1, 2], [25, 26], [3]
    elif n == 756:  # d<=34
        args = 21, 18, [3], [10, 17], [3, 19], [5]
    else:
        raise ValueError

    code, A_list, B_list = create_bivariate_bicycle_codes(*args)
    circuit = build_circuit(code,
                            A_list,
                            B_list,
                            p=p,
                            # physical error rate
                            num_repeat=T,
                            # usually set to code distance
                            z_basis=True,
                            # whether in the z-basis or x-basis
                            use_both=False,
                            # whether use measurement results in both basis to decode one basis
                            )
    return circuit


def set_df_dtypes(df: pd.DataFrame, dtypes: list):
    for col, dtype in zip(df.columns, dtypes):
        df[col] = df[col].astype(dtype)


def simulate(circuit: stim.Circuit,
             shots: int,
             *,
             max_iter=30,
             bp_method='product_sum',
             lsd_method='lsd_cs',
             lsd_order=0,
             get_logical_gaps=False,
             d=None,
             p=None,
             get_aggr_data=False,
             Q_digits=2,
             n_jobs=-1,
             n_batch=None,
             verbose=0):
    dem = circuit.detector_error_model()
    H, obs, p_priors = dem_to_check_matrices(dem)

    if verbose:
        print("Number of checks:", H.shape[0])
        print("Number of faults:", H.shape[1])
        print("Number of observables:", obs.shape[0])

    dem_sampler = dem.compile_sampler()
    det_data, obs_data, _ = dem_sampler.sample(shots=round(shots),
                                               return_errors=False,
                                               bit_packed=False)

    det_density_all = det_data.sum(axis=1) / det_data.shape[1]

    # Only for surface codes
    if get_logical_gaps:
        assert d is not None
        assert p is not None
        bdry_errors = obs.nonzero()[1]
        bdry_check = csc_matrix(([1] * len(bdry_errors),
                                 ([0] * len(bdry_errors), bdry_errors)),
                                shape=(1, H.shape[1]))
        H_with_bdry = vstack([H, bdry_check])
        weights = np.log((1 - p_priors) / p_priors)

    def task(det_data_batch, obs_data_batch, det_density_batch):
        bplsd = BpLsdDecoder(
            H,
            error_channel=p_priors,
            max_iter=max_iter,
            bp_method=bp_method,
            lsd_method=lsd_method,
            lsd_order=lsd_order,
        )
        bplsd.set_do_stats(True)
        if get_aggr_data:
            strategies = ['cf', 'dd']
            if get_logical_gaps:
                strategies += ['gap']
            cols = ['Q']
            for strat in strategies:
                cols.extend([f'num_samples_{strat}', f'num_fails_{strat}'])
            output = pd.DataFrame(columns=cols, dtype='float64')
            output.set_index('Q', inplace=True)
        else:
            output = pd.DataFrame(columns=['fail', 'cluster_frac',
                                           'cluster_num'])

        for det_vals, obs_vals, det_density in zip(det_data_batch,
                                                   obs_data_batch,
                                                   det_density_batch):
            preds = bplsd.decode(det_vals)
            obss_pred = [
                preds[obs_sng.toarray().astype('bool').ravel()].sum() % 2
                for obs_sng in obs
            ]
            obss_pred = np.array(obss_pred).astype('bool')
            fail = np.any(obs_vals ^ obss_pred)

            stats = bplsd.statistics['individual_cluster_stats']
            cluster_sizes \
                = [data['final_bit_count'] for _, data in stats.items()
                   if data['active']]
            # cluster_size_sum = sum(cluster_sizes)
            cluster_num = len(cluster_sizes)
            # cluster_max_size = max(cluster_sizes) if cluster_sizes else 0
            cluster_frac = sum(cluster_sizes) / H.shape[1]

            if get_aggr_data:
                factor = 10**Q_digits
                cluster_frac = np.ceil(cluster_frac * factor) / factor
                det_density = np.ceil(det_density * factor) / factor
                Qs = {'cf': cluster_frac, 'dd': det_density}

                for strat, Q in Qs.items():
                    col_num_samples = f'num_samples_{strat}'
                    col_num_fails = f'num_fails_{strat}'
                    try:
                        if pd.isnull(output.loc[Q, col_num_samples]):
                            output.loc[Q, col_num_samples] = 0
                            output.loc[Q, col_num_fails] = 0
                        output.loc[Q, col_num_samples] += 1
                        if fail:
                            output.loc[Q, col_num_fails] += 1
                    except KeyError:
                        output.loc[Q, col_num_samples] = 1
                        output.loc[Q, col_num_fails] = int(fail)

            else:
                output.loc[len(output), :] = fail, cluster_frac, cluster_num

        if get_aggr_data:
            output.fillna(0, inplace=True)
            for col in output.columns:
                output[col] = output[col].astype('uint64')

        if get_logical_gaps:
            matching = pm.Matching(H_with_bdry, weights=weights)
            pred_weights_both = []
            fails_both = []
            for bdry_value in [False, True]:
                det_data_with_bdry \
                    = np.hstack([det_data_batch,
                                 np.full((det_data_batch.shape[0], 1),
                                         bdry_value)])
                preds, pred_weights \
                    = matching.decode_batch(det_data_with_bdry,
                                            return_weights=True)
                obs_preds = preds[:, bdry_errors].sum(axis=1) % 2
                fails = (obs_preds ^ obs_data_batch.ravel()).astype('bool')
                pred_weights_both.append(pred_weights)
                fails_both.append(fails)
            logical_gaps \
                = np.abs(pred_weights_both[0] - pred_weights_both[1])
            logical_gaps /= (d * np.log((1 - p) / p))
            fails = np.where(pred_weights_both[0] <= pred_weights_both[1],
                             fails_both[0], fails_both[1])

            if get_aggr_data:
                raise NotImplementedError
            else:
                output['fail_mwpm'] = fails
                output['logical_gap'] = logical_gaps

        if not get_aggr_data:
            set_df_dtypes(output, ['bool', 'float64', 'uint64'])

        return output

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    if n_batch is None:
        outputs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(task)([det], [obs])
            for det, obs in zip(det_data, obs_data))

    elif n_batch >= n_jobs:
        det_data_split = np.array_split(det_data, n_batch, axis=0)
        obs_data_split = np.array_split(obs_data, n_batch, axis=0)
        det_density_split = np.array_split(det_density_all, n_batch, axis=0)
        outputs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(task)(*prms) for prms
            in zip(det_data_split, obs_data_split, det_density_split))
    else:
        raise ValueError

    if get_aggr_data:
        outputs = pd.concat(outputs, axis=0).groupby(level=0).sum()
        outputs.sort_index(inplace=True)

    else:
        outputs = pd.concat(outputs, axis=0)
        outputs['det_density'] = det_density_all

    return outputs


def compute_ps_performance(df: pd.DataFrame,
                           clist: np.ndarray,
                           strat='all',
                           alpha=0.05,
                           verbose=False):
    clist = np.asanyarray(clist)

    possible_strats = ['cluster_frac', 'det_density']
    if 'logical_gap' in df.columns:
        possible_strats.append('logical_gap')
    if strat == 'all':
        strats = possible_strats
    else:
        assert strat in possible_strats
        strats = [strat]

    num_samples = len(df)

    outputs = {}

    for strat in strats:
        if verbose:
            print('strat =', strat)
        df.sort_values(strat,
                       axis=0,
                       ascending=(strat != 'logical_gap'),
                       inplace=True)
        df.reset_index(inplace=True, drop=True)
        Q = df[strat].values
        if strat == 'logical_gap':
            Q = 1 - Q
        num_fails = df['fail'].values.cumsum()
        num_accepted = np.arange(1, num_samples + 1)

        inds_c = np.searchsorted(Q, 1 - clist, side='right') - 1
        mask_valid_inds = inds_c != -1
        inds_c = inds_c[mask_valid_inds]
        clist_valid = clist[mask_valid_inds]
        num_fails = num_fails[inds_c]
        num_accepted = num_accepted[inds_c]

        fail_low, fail_upp = proportion_confint(num_fails,
                                                num_accepted,
                                                alpha=alpha,
                                                method='binom_test')
        fail = (fail_low + fail_upp) / 2
        delta_fail = fail_upp - fail
        abort_low, abort_upp \
            = proportion_confint(num_samples - num_accepted,
                                 num_samples,
                                 alpha=alpha,
                                 method='binom_test')
        abort = (abort_low + abort_upp) / 2
        delta_abort = abort_upp - abort

        df_ps = pd.DataFrame({'c': clist_valid,
                              'fail': fail,
                              'delta_fail': delta_fail,
                              'abort': abort,
                              'delta_abort': delta_abort})
        df_ps = df_ps.set_index('c').sort_index()

        outputs[strat] = df_ps

    return outputs


class BpLsdPsDecoder:
    def __init__(self, H, p, **kwargs):
        self.bplsd = BpLsdDecoder(H, error_channel=p, **kwargs)
        self.bplsd.set_do_stats(True)
        self.H = H
        self.p = p

    def decode(self, dets, return_cluster_size=False):
        bplsd = self.bplsd
        preds = bplsd.decode(dets)

        # Calculate cluster_frac
        stats = bplsd.statistics['individual_cluster_stats']
        cluster_sizes \
            = [data['final_bit_count'] for _, data in stats.items()
               if data['active']]
        cluster_total_size = sum(cluster_sizes)
        cluster_frac = cluster_total_size / self.H.shape[1]

        if return_cluster_size:
            return preds, cluster_frac, cluster_total_size
        else:
            return preds, cluster_frac


def get_datetime():
    formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


def find_smallest_uint_type(n):
    if n < 0:
        return "No unsigned type can store a negative integer."

    if n <= 15:  # 4 bits (2^4 - 1 = 15)
        return "uint4"
    elif n <= 255:  # 8 bits (2^8 - 1 = 255)
        return "uint8"
    elif n <= 65535:  # 16 bits (2^16 - 1 = 65535)
        return "uint16"
    elif n <= 4294967295:  # 32 bits (2^32 - 1 = 4294967295)
        return "uint32"
    elif n <= 18446744073709551615:  # 64 bits (2^64 - 1 = 18446744073709551615)
        return "uint64"
    else:
        return "No available type can store this integer."


class MCMC:
    def __init__(self,
                 *,
                 c: float,
                 H: csc_matrix | None = None,
                 obs: csc_matrix | None = None,
                 p: List[float] | np.ndarray | None = None,
                 circuit: stim.Circuit | None = None,
                 e0: np.ndarray | None = None,
                 e0_shots_per_batch: int = 100,
                 e0_prob_scale: float | None = None,
                 _force_no_e0: bool = False,
                 **kwargs):
        if circuit is None:
            assert all(arg is not None for arg in [H, obs, p])

        self.c = c
        if circuit is not None:
            dem = circuit.detector_error_model()
            H, obs, p = dem_to_check_matrices(dem)
        self.H = H
        self.obs = obs
        self.p = np.asanyarray(p, dtype='float64')
        self.kwargs = kwargs
        self.decoder = BpLsdPsDecoder(H, p, **kwargs)

        if _force_no_e0:
            e0 = fail = None
        elif e0 is None:
            e0, fail = self._find_e0(e0_shots_per_batch, e0_prob_scale)
        else:
            syndrome = (e0 @ H.T) % 2
            pred, cluster_frac, cluster_size \
                = self.decoder.decode(syndrome, return_cluster_size=True)
            if cluster_frac > 1 - c:
                raise ValueError('Initial decoding aborted. Wrong e0.')
            residue = e0 ^ pred
            fail = ((residue @ obs.T) % 2).any()

        if e0 is None:
            self.e0 = self.last_error = None
        else:
            self.e0 = self.last_error = np.asanyarray(e0, dtype='bool')
        self._fails = [fail]

        self.shots_rej = 0
        self.shots_acc = self.shots_dec = 1
        self.last_n_flip = None
        self.last_shots_acc = self.last_shots_dec = None

        self.time = []

    def initialise(self):
        if self.e0 is None:
            raise ValueError("Unable to initialise since e0 is not given.")
        self.shots_loaded = None
        self.last_error = self.e0
        self._fails = [self._fails[0]]
        self.shots_acc = self.shots_dec = 1
        self.shots_rej = 0
        self.last_n_flip = None
        self.last_shots_acc = self.last_shots_dec = None
        self.time = []

    @property
    def shots(self) -> int:
        return len(self._fails)

    @property
    def fails(self) -> np.ndarray:
        return np.array(self._fails, dtype='bool')

    @property
    def pfail(self) -> float:
        return self.fails.sum() / self.shots

    @property
    def pfails_history(self) -> np.ndarray:
        return self.fails.cumsum() / np.arange(1, self.shots + 1)

    @property
    def acc_rate(self) -> float:
        return self.shots_acc / (self.shots_rej + self.shots_acc)

    @property
    def dec_rate(self) -> float:
        return self.shots_dec / (self.shots_rej + self.shots_acc)

    @property
    def eff_acc_rate(self) -> float:
        return self.shots_acc / self.shots_dec

    @property
    def sampling_rate(self) -> float:
        total_time = sum(t[1] for t in self.time)
        shots = sum(t[0] for t in self.time)
        freq = shots / total_time
        return freq

    @property
    def ess(self) -> float:
        idata = az.convert_to_inference_data(self.fails)
        ess = az.ess(idata)['x'].values
        return ess

    def copy(self):
        return deepcopy(self)

    def _find_e0(self,
                 shots_per_batch: int = 100,
                 prob_scale: float | None = None) -> np.ndarray:
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
            dets = (errors @ H.T) % 2
            for errors_sng, dets_sng in zip(errors, dets):
                pred, cluster_frac = decoder.decode(dets_sng)
                if cluster_frac <= 1 - c:
                    e0 = errors_sng
                    residue = e0 ^ pred
                    fail = ((residue @ obs.T) % 2).any()
                    break
        return e0, fail

    def sample(self,
               shots: int,
               *,
               flip_single_qubit: bool = True,
               n_flip: int = 1,
               adaptive_n_flip: bool = True,
               target_acc_rate: float = 0.234,
               alpha=0.05) -> List[bool]:
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
                self.last_shots_acc = self.last_shots_dec = 0
            else:
                n_flip = self.last_n_flip
                assert self.last_shots_acc is not None
                assert self.last_shots_dec is not None
        else:
            self.last_n_flip = self.last_shots_acc = self.last_shots_dec \
                = None

        for i in range(shots):
            # Adaptively choose n_flip depending on the effective acceptance rate
            if adaptive_n_flip and self.last_shots_dec:
                eff_acc_rate_low, eff_acc_rate_high \
                    = proportion_confint(self.last_shots_acc,
                                         self.last_shots_dec,
                                         method='wilson',
                                         alpha=alpha)
                if eff_acc_rate_low > target_acc_rate:
                    n_flip = min(n_flip + 1, num_error_locs)
                    self.last_n_flip = n_flip
                    self.last_shots_acc = self.last_shots_dec = 0
                    # print(i, n_flip, eff_acc_rate_low, eff_acc_rate_high)
                elif eff_acc_rate_high < target_acc_rate:
                    n_flip = max(n_flip - 1, 1)
                    self.last_n_flip = n_flip
                    self.last_shots_acc = self.last_shots_dec = 0
                    # print(i, n_flip, eff_acc_rate_low, eff_acc_rate_high)

            if flip_single_qubit:
                error_diff = np.zeros(num_error_locs, dtype='bool')
                error_diff[np.random.randint(0, num_error_locs)] = True
            else:
                p_flip = n_flip / num_error_locs
                while True:
                    error_diff \
                        = np.random.uniform(size=num_error_locs) < p_flip
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
                pred, cluster_frac, cluster_size \
                    = decoder.decode(syndrome, return_cluster_size=True)
                self.shots_dec += 1
                if adaptive_n_flip:
                    self.last_shots_dec += 1

                if cluster_frac <= 1 - c:
                    residue = error_cand ^ pred
                    fail = ((residue @ obs.T) % 2).any()
                    fails.append(fail)
                    self.shots_acc += 1
                    if adaptive_n_flip:
                        self.last_shots_acc += 1
                    self.last_error = error_cand
                    continue

            # If not accepted, use last error again
            fails.append(fails[-1])
            self.shots_rej += 1

        self.time.append((shots, time.time() - t0))

        return fails[-shots:]

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'decoder' in state:
            del state['decoder']
        return state

    def __setstate__(self, state):
        if 'fails' in state:
            state['_fails'] = state['fails']
            del state['fails']
        if 'shots_recorded' in state:
            state['shots_rej'] = state['shots_recorded']
            del state['shots_recorded']
        if 'time' not in state:
            state['time'] = []
        self.__dict__.update(state)
        self.decoder = BpLsdPsDecoder(self.H, self.p, **self.kwargs)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> Self:
        with open(path, 'rb') as f:
            return pickle.load(f)


# def mcmc_sample(shots: int,
#                 *,
#                 c: float,
#                 dem: stim.DetectorErrorModel | None = None,
#                 H: csc_matrix | None = None,
#                 obs: csc_matrix | None = None,
#                 p: np.ndarray | None = None,
#                 e0: np.ndarray | None = None,
#                 shots_e0: int = 100,
#                 prob_scale_e0: float | None = None,
#                 include_e0: bool = True,
#                 full_output: bool = False,
#                 return_acc_rates: bool = False,
#                 flip_single_qubit: bool = True,
#                 n_flip: int = 1,
#                 adaptive_n_flip: bool = False,
#                 target_acc_rate: float = 0.234,
#                 **kwargs):
#     assert dem is not None or all(prm is not None for prm in [H, obs, p])
#     if dem is not None:
#         H, obs, p = dem_to_check_matrices(dem)
#
#     p = np.asanyarray(p, dtype='float64')
#     decoder = BpLsdPsDecoder(H, p, **kwargs)
#     log_p = np.log(p)
#     log_one_minus_p = np.log(1 - p)
#
#     # Find e0 if it is not explicitly given
#     if e0 is None:
#         assert dem is not None
#         if prob_scale_e0 is None:
#             dem_e0 = dem
#         else:
#             assert prob_scale_e0 > 0
#             dem_e0 = stim.DetectorErrorModel()
#             for inst in dem:
#                 if inst.type == 'error':
#                     dem_e0.append('error',
#                                   [inst.args_copy()[0] * prob_scale_e0],
#                                   inst.targets_copy())
#                 else:
#                     dem_e0.append(inst)
#
#         dem_sampler = dem_e0.compile_sampler()
#         while e0 is None:
#             samples = dem_sampler.sample(shots=shots_e0,
#                                          return_errors=True,
#                                          bit_packed=False)
#             for det_data, obs_data, error_data in zip(*samples):
#                 pred, cluster_frac = decoder.decode(det_data)
#                 if cluster_frac <= 1 - c:
#                     e0 = error_data
#                     break
#
#     error = np.asanyarray(e0, dtype='bool')
#     syndrome = (error @ H.T) % 2
#
#     pred, cluster_frac, cluster_size \
#         = decoder.decode(syndrome, return_cluster_size=True)
#     if cluster_frac > 1 - c:
#         raise ValueError('Initial decoding aborted. Try different e0.')
#     residue = error ^ pred
#     fail = ((residue @ obs.T) % 2).any()
#
#     fails = np.zeros(shots, dtype='bool')
#     if full_output:
#         dtype_error = find_smallest_uint_type(H.shape[1])
#         dtype_syndrome = find_smallest_uint_type(H.shape[0])
#         cluster_sizes = np.zeros(shots, dtype=dtype_error)
#         num_errors = np.zeros(shots, dtype=dtype_error)
#         num_defects = np.zeros(shots, dtype=dtype_syndrome)
#     else:
#         cluster_sizes = num_errors = num_defects = None
#
#     if include_e0:
#         fails[0] = fail
#         if full_output:
#             cluster_sizes[0] = int(cluster_size)
#             num_errors[0] = int(np.sum(error))
#             num_defects[0] = int(np.sum(syndrome))
#
#     acc_shots = 0
#     decoded_shots = 0
#     num_error_locs = H.shape[1]
#
#     p_flip = n_flip / num_error_locs
#     for i in range(int(include_e0), shots):
#         prev_error = error
#         prev_fail = fail
#
#         # Adaptively choose n_flip depending on the effective acceptance rate
#         if adaptive_n_flip:
#             pass
#
#         if flip_single_qubit:
#             error_diff = np.zeros(num_error_locs, dtype='bool')
#             error_diff[np.random.randint(0, num_error_locs)] = True
#         else:
#             while True:
#                 error_diff = np.random.choice([True, False],
#                                               size=num_error_locs,
#                                               p=[p_flip, 1 - p_flip])
#                 if error_diff.sum():
#                     break
#         error = prev_error ^ error_diff
#
#         only_in_prev_error = prev_error & ~error
#         only_in_error = ~prev_error & error
#         log_q = -np.sum(log_p[only_in_prev_error])
#         log_q += np.sum(log_one_minus_p[only_in_prev_error])
#         log_q -= np.sum(log_one_minus_p[only_in_error])
#         log_q += np.sum(log_p[only_in_error])
#         q = min(1, np.exp(log_q))
#         # print(q, end='')
#         if np.random.choice([True, False], p=(q, 1 - q)):
#             syndrome = (error @ H.T) % 2
#             pred, cluster_frac, cluster_size \
#                 = decoder.decode(syndrome, return_cluster_size=True)
#             decoded_shots += 1
#
#             if cluster_frac <= 1 - c:
#                 residue = error ^ pred
#                 fail = ((residue @ obs.T) % 2).any()
#                 fails[i] = fail
#                 acc_shots += 1
#                 if full_output:
#                     cluster_sizes[i] = int(cluster_size)
#                     num_errors[i] = int(np.sum(error))
#                     num_defects[i] = int(np.sum(syndrome))
#                 continue
#         # If not accepted, use previous error again
#         error, fail = prev_error, prev_fail
#         fails[i] = fail
#         if full_output:
#             cluster_sizes[i] = cluster_sizes[i - 1]
#             num_errors[i] = num_errors[i - 1]
#             num_defects[i] = num_defects[i - 1]
#         # print(' N')
#
#     acc_rate = acc_shots / (shots - int(include_e0))
#     decoded_rate = decoded_shots / (shots - int(include_e0))
#
#     additional_output = {}
#     if return_acc_rates:
#         additional_output['acc_rate'] = acc_rate
#         additional_output['decoded_rate'] = decoded_rate
#
#     if full_output:
#         additional_output.update({
#             'cluster_sizes': cluster_sizes,
#             'num_errors': num_errors,
#             'num_defects': num_defects,
#         })
#
#     return fails, error, additional_output


# Gelman-Rubin
# https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains
# def metropolis_simulation(dem: stim.DetectorErrorModel,
#                           *,
#                           c: float,
#                           init_shots: int | None = None,
#                           shots_increase: int = 2,
#                           max_shots: int | None = None,
#                           max_round: int | None = None,
#                           num_chains: int = 6,
#                           R_tol: float = 1.1,
#                           # pfail_rme_tol: float = 0.5,
#                           pfail_rel_diff_tol: float = 0.1,
#                           pfail_ub: float | None = None,
#                           verbose: bool = False,
#                           prev_data: str | dict | None = None,
#                           save_path: str | None = None,
#                           **kwargs,
#                           ):
#     def task(shots, e0, include_e0):
#         fails, next_e0 = mcmc_sample(shots,
#                                      c=c,
#                                      dem=dem,
#                                      e0=e0,
#                                      include_e0=include_e0,
#                                      **kwargs)
#         num_fails = np.sum(fails)
#         return num_fails, next_e0
#
#     if max_shots is None:
#         max_shots = np.inf
#
#     if isinstance(prev_data, str):
#         try:
#             with open(prev_data, 'rb') as f:
#                 prev_data = pickle.load(f)
#         except FileNotFoundError:
#             prev_data = None
#
#     if prev_data is None:
#         assert dem is not None
#         assert c is not None
#         assert init_shots is not None
#         shots_now = init_shots
#         e0_list = [None] * num_chains
#         include_e0 = True
#         i_round = 0
#         num_fails = np.zeros(num_chains, dtype='int32')
#         shots = 0
#         pfail = None
#         Blist = []
#         Wlist = []
#         Vlist = []
#         Rlist = []
#         pfail_rel_diff_list = []
#         converged = False
#     else:
#         assert num_chains == len(prev_data['fails_indv'])
#         assert c == prev_data['c']
#         dem = prev_data['dem']
#         shots_now = min(max_shots,
#                         prev_data['shots_last_round'] * shots_increase)
#         e0_list = prev_data['e0_list']
#         include_e0 = False
#         i_round = prev_data['num_rounds']
#         num_fails = prev_data['fails_indv']
#         shots = prev_data['shots_indv']
#         pfail = prev_data['pfail']
#         Blist = prev_data['Blist']
#         Wlist = prev_data['Wlist']
#         Vlist = prev_data['Vlist']
#         Rlist = prev_data['Rlist']
#         pfail_rel_diff_list = prev_data['pfail_rel_diff_list']
#         R = Rlist[-1]
#         pfail_rel_diff = pfail_rel_diff_list[-1]
#         converged = (R < R_tol and pfail_rel_diff < pfail_rel_diff_tol
#                      and (pfail_ub is None or pfail < pfail_ub))
#         if converged:
#             data = prev_data
#
#     while not converged and (max_round is None or i_round < max_round):
#         if verbose:
#             formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             print(f"Stage {i_round} ({formatted_datetime})")
#         outputs = Parallel(n_jobs=num_chains, backend='loky')(
#             delayed(task)(shots_now, e0, include_e0) for e0 in e0_list)
#         num_fails_now, e0_list = zip(*outputs)
#         num_fails += np.array(num_fails_now)
#         shots += shots_now
#
#         # pfail_low, pfail_upp = proportion_confint(np.sum(num_fails),
#         #                                           shots * num_chains,
#         #                                           alpha=alpha,
#         #                                           method='binom_test')
#         prev_pfail = pfail
#         pfail = np.sum(num_fails) / shots / num_chains
#         # pfail_rme = (pfail_upp - pfail_low) / pfail / 2
#
#         if verbose:
#             print("pfail =", pfail)
#             # print("pfail_rme =", pfail_rme)
#
#         means = num_fails / shots
#         variances = shots / (shots - 1) * means * (1 - means)
#         B = shots * np.var(means, ddof=1)
#         W = np.mean(variances)
#         V = ((shots - 1) / shots * W
#              + (num_chains + 1) / num_chains / shots * B)
#         R = np.sqrt(V / W) if W > 0 else np.inf
#         Blist.append(B)
#         Wlist.append(W)
#         Vlist.append(V)
#         Rlist.append(R)
#
#         if prev_pfail is None:
#             pfail_rel_diff = np.inf
#         elif pfail > 0:
#             pfail_rel_diff = np.abs((pfail - prev_pfail) / pfail)
#         else:
#             pfail_rel_diff = np.inf
#
#         pfail_rel_diff_list.append(pfail_rel_diff)
#
#         if verbose:
#             print("R =", R)
#             print("pfail_rel_diff =", pfail_rel_diff)
#
#         converged = (R < R_tol and pfail_rel_diff < pfail_rel_diff_tol
#                      and (pfail_ub is None or pfail < pfail_ub))
#
#         data = {
#             'dem': dem,
#             'c': c,
#             'fails': np.sum(num_fails),
#             'shots': shots * num_chains,
#             'pfail': pfail,
#             'B': B,
#             'W': W,
#             'V': V,
#             'R': R,
#             'Blist': Blist,
#             'Wlist': Wlist,
#             'Vlist': Vlist,
#             'Rlist': Rlist,
#             'pfail_rel_diff': pfail_rel_diff,
#             'pfail_rel_diff_list': pfail_rel_diff_list,
#             'num_rounds': i_round + 1,
#             'fails_indv': num_fails,
#             'shots_indv': shots,
#             'shots_last_round': shots_now,
#             'e0_list': e0_list
#         }
#
#         if save_path:
#             with open(save_path, 'wb') as f:
#                 pickle.dump(data, f)
#
#         shots_now = min(shots_now * shots_increase, max_shots)
#         include_e0 = False
#         i_round += 1
#         if verbose:
#             print()
#
#     return data


def plot_error_band(data=None,
                    *,
                    x=None,
                    y=None,
                    delta_y=None,
                    ax=None,
                    alpha=.2,
                    color=None,
                    **kwargs):
    target = plt if ax is None else ax
    if data is None:
        line, _ = target.plot(x, y, **kwargs)
        target.fill_between(x,
                            y - delta_y,
                            y + delta_y,
                            color=line.get_color(),
                            alpha=.2)
    else:
        ax = sns.lineplot(data, x=x, y=y, ax=ax, **kwargs)
        target.fill_between(data[x],
                            data[y] - data[delta_y],
                            data[y] + data[delta_y],
                            color=ax.get_lines()[-1].get_color(),
                            alpha=alpha)
