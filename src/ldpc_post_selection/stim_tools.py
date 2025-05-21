from typing import Sequence, Tuple

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

    final_probabilities = []
    final_det_id_lists = []
    final_obs_id_lists = []

    for _, instruction in enumerate(dem):
        instruction: stim.DemInstruction
        if instruction.type == "error":
            error_prob = float(instruction.args_copy()[0])

            current_segment_dets = []
            current_segment_obs = []

            targets = instruction.targets_copy()
            if (
                not targets
            ):  # Handle error instructions with no targets, if they are possible
                # This case might mean an error with a probability but no effect on dets/obs.
                # Depending on desired behavior, one might add an empty mechanism or skip.
                # For now, let's assume error instructions always have targets if they are meaningful.
                # If an error op has a probability but no targets, it won\'t form a column.
                pass

            for i, target in enumerate(targets):
                if target.is_relative_detector_id():
                    current_segment_dets.append(int(str(target)[1:]))
                elif target.is_logical_observable_id():
                    current_segment_obs.append(int(str(target)[1:]))
                elif target.is_separator():
                    if (
                        current_segment_dets or current_segment_obs
                    ):  # Add segment if not empty
                        final_det_id_lists.append(list(current_segment_dets))
                        final_obs_id_lists.append(list(current_segment_obs))
                        final_probabilities.append(error_prob)
                    current_segment_dets.clear()  # Reset for next segment
                    current_segment_obs.clear()
                else:
                    raise ValueError(f"Unknown target type: {target}")

            # After loop, add the last segment
            if current_segment_dets or current_segment_obs:
                final_det_id_lists.append(list(current_segment_dets))
                final_obs_id_lists.append(list(current_segment_obs))
                final_probabilities.append(error_prob)

    p = np.array(final_probabilities, dtype=float)
    num_decomposed_errors = len(final_probabilities)

    # Create the parity check matrix H
    if num_decomposed_errors > 0:
        num_detectors = dem.num_detectors
        row_indices_h = []
        col_indices_h = []
        data_h = []

        for error_idx, det_ids in enumerate(final_det_id_lists):
            for det_id in det_ids:
                row_indices_h.append(det_id)
                col_indices_h.append(error_idx)
                data_h.append(True)

        H = csc_matrix(
            (data_h, (row_indices_h, col_indices_h)),
            shape=(num_detectors, num_decomposed_errors),
            dtype=bool,
        )
    else:
        # If dem.num_detectors is 0, H should be (0,0)
        # If num_decomposed_errors is 0 but dem.num_detectors > 0, H should be (dem.num_detectors, 0)
        H = csc_matrix((dem.num_detectors, 0), dtype=bool)

    # Create the observable matrix
    if num_decomposed_errors > 0:
        num_observables = dem.num_observables
        row_indices_obs = []
        col_indices_obs = []
        data_obs = []

        for error_idx, obs_ids in enumerate(final_obs_id_lists):
            for obs_id in obs_ids:
                row_indices_obs.append(obs_id)
                col_indices_obs.append(error_idx)
                data_obs.append(True)

        obs_matrix = csc_matrix(
            (data_obs, (row_indices_obs, col_indices_obs)),
            shape=(num_observables, num_decomposed_errors),
            dtype=bool,
        )
    else:
        # If dem.num_observables is 0, obs_matrix should be (0,0)
        # If num_decomposed_errors is 0 but dem.num_observables > 0, obs_matrix should be (dem.num_observables, 0)
        obs_matrix = csc_matrix((dem.num_observables, 0), dtype=bool)

    return H, obs_matrix, p


def remove_detectors_from_circuit(
    circuit: stim.Circuit,
    detector_ids_to_remove: Sequence[int],
) -> stim.Circuit:
    """
    Removes specified DETECTOR instructions from a stim circuit.

    Parameters
    ----------
    circuit : stim.Circuit
        The input stim circuit.
    detector_ids_to_remove : Sequence[int]
        A sequence of 0-indexed original detector IDs to remove. These correspond
        to the order of DETECTOR instructions in the circuit.

    Returns
    -------
    stim.Circuit
        A new stim circuit with the specified DETECTOR instructions removed.
    """
    circuit = circuit.flattened()
    detector_ids_to_remove_set = set(detector_ids_to_remove)
    new_circuit = stim.Circuit()

    original_detector_idx_counter = 0

    for instruction in circuit:
        if instruction.name == "DETECTOR":
            if original_detector_idx_counter not in detector_ids_to_remove_set:
                new_circuit.append(instruction)
            original_detector_idx_counter += 1
        else:
            new_circuit.append(instruction)

    return new_circuit
