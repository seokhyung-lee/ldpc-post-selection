from typing import Tuple

import numpy as np
import stim
from scipy.sparse import csc_matrix


def dem_to_parity_check(
    dem: stim.DetectorErrorModel,
) -> Tuple[csc_matrix, csc_matrix, np.ndarray]:
    """
    Convert a detector error model (DEM) into a parity check matrix, observable matrix,
    and probability vector.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to convert.

    Returns
    -------
    H : csc_matrix
        A boolean matrix of shape (number of detectors, number of errors)
        where H[i, j] = True if detector i is involved in error j.
    obs_matrix : csc_matrix
        A boolean matrix of shape (number of observables, number of errors)
        where obs_matrix[i, j] = True if observable i is involved in error j.
    p : np.ndarray
        A 1D numpy array of probabilities corresponding to errors in the DEM.
    """
    dem = dem.flattened()

    probabilities = []
    det_ids_in_ems = []
    obs_ids_in_ems = []

    for _, instruction in enumerate(dem):
        if instruction.type == "error":
            det_ids = []
            obs_ids = []
            det_ids_in_ems.append(det_ids)
            obs_ids_in_ems.append(obs_ids)

            # Extract probability
            prob = float(instruction.args_copy()[0])
            probabilities.append(prob)

            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    det_ids.append(int(str(target)[1:]))
                elif target.is_logical_observable_id():
                    obs_ids.append(int(str(target)[1:]))
                else:
                    raise ValueError(f"Unknown target type: {target}")

    p = np.array(probabilities)

    # Create the parity check matrix H
    if det_ids_in_ems:
        num_detectors = dem.num_detectors
        num_errors = len(det_ids_in_ems)

        # Prepare data for CSC matrix construction
        row_indices = []
        col_indices = []
        data = []

        for error_idx, det_ids in enumerate(det_ids_in_ems):
            for det_id in det_ids:
                row_indices.append(det_id)
                col_indices.append(error_idx)
                data.append(True)

        H = csc_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_detectors, num_errors),
            dtype=bool,
        )
    else:
        H = csc_matrix((0, 0), dtype=bool)

    # Create the observable matrix
    if obs_ids_in_ems:
        # Find the maximum observable ID
        num_observables = dem.num_observables
        num_errors = len(obs_ids_in_ems)

        # Prepare data for CSC matrix construction
        row_indices = []
        col_indices = []
        data = []

        for error_idx, obs_ids in enumerate(obs_ids_in_ems):
            for obs_id in obs_ids:
                row_indices.append(obs_id)
                col_indices.append(error_idx)
                data.append(True)

        obs_matrix = csc_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_observables, num_errors),
            dtype=bool,
        )
    else:
        obs_matrix = csc_matrix((0, 0), dtype=bool)

    return H, obs_matrix, p
