import stim

from color_code_stim import ColorCode

from ext.SlidingWindowDecoder.src.build_circuit import (
    build_circuit as build_BB_circuit_swd,
)
from ext.SlidingWindowDecoder.src.codes_q import (
    create_bivariate_bicycle_codes,
)
from ldpc_post_selection.stim_tools import remove_detectors_from_circuit


def build_surface_code_circuit(
    *,
    d: int,
    T: int,
    p: float = 0.0,
    noise: str = "circuit-level",
    only_z_detectors: bool = False,
) -> stim.Circuit:
    """
    Build a stim circuit for a rotated surface code.

    Parameters
    ----------
    d : int
        The code distance.
    T : int
        The number of measurement rounds.
    p : float, default=0.0
        The physical error rate.
    noise : str, default="circuit-level"
        The noise model type: ["circuit-level", "code-capacity", "phenom"].
    only_z_detectors : bool, default=False
        Whether to only include Z-type detectors.

    Returns
    -------
    stim.Circuit
        The generated stim circuit.
    """
    if noise in {"circuit-level", "circuit_level"}:
        p_clifford = p_meas = p_reset = p
        p_depol = 0
    elif noise in {"code-capacity", "code_capacity"}:
        p_clifford = p_meas = p_reset = 0
        p_depol = p
    elif noise in {"phenom", "phenomenological"}:
        p_clifford = p_reset = 0
        p_depol = p_meas = p

    else:
        raise ValueError(f"Invalid noise type: {noise}")

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=T,
        after_clifford_depolarization=p_clifford,
        before_round_data_depolarization=p_depol,
        before_measure_flip_probability=p_meas,
        after_reset_flip_probability=p_reset,
    )
    if only_z_detectors:
        detector_coords = circuit.get_detector_coordinates()
        det_ids_to_remove = []  # X-type detectors
        for det_id, (x, y, z) in detector_coords.items():
            if (round(x) + round(y)) % 4 == 2:
                det_ids_to_remove.append(det_id)
        circuit = remove_detectors_from_circuit(circuit, det_ids_to_remove)

    return circuit


def build_color_code_circuit(
    *,
    d: int,
    T: int,
    p: float = 0.0,
    noise: str = "circuit-level",
    only_z_detectors: bool = False,
    comparative_decoding: bool = False,
) -> stim.Circuit:
    """
    Build a stim circuit for a color code.

    Parameters
    ----------
    d : int
        The code distance.
    T : int
        The number of measurement rounds.
    p : float, default=0.0
        The physical error rate.
    noise : str, default="circuit-level"
        The noise model type. Can be either "circuit-level" or "code-capacity".
    only_z_detectors : bool, default=False
        Whether to only include Z-type detectors.
    comparative_decoding : bool, default=False
        Whether to use comparative decoding.

    Returns
    -------
    stim.Circuit
        The generated stim circuit.
    """
    if noise in {"circuit-level", "circuit_level"}:
        p_circuit = p
        p_depol = 0
    elif noise in {"code-capacity", "code_capacity"}:
        p_circuit = 0
        p_depol = p
    else:
        raise ValueError(f"Invalid noise type: {noise}")
    cc = ColorCode(
        d=d,
        rounds=T,
        p_circuit=p_circuit,
        p_bitflip=p_depol * 2 / 3,
        comparative_decoding=comparative_decoding,
        exclude_non_essential_pauli_detectors=only_z_detectors,
    )
    circuit = cc.circuit

    return circuit


def get_BB_distance(n: int) -> int:
    distance_dict = {
        72: 6,
        90: 10,
        108: 10,
        144: 12,
        288: 18,
        360: 24,
        756: 34,
    }
    if n not in distance_dict:
        raise ValueError(f"Unsupported code size: {n}")

    return distance_dict[n]


def build_BB_circuit(*, n: int, T: int, p: float) -> stim.Circuit:
    """
    Build a stim circuit for a Bivariate Bicycle (BB) code.

    Currently supported values for `n` correspond to specific code parameters:
    - n=72 (d=6)
    - n=90 (d=10)
    - n=108 (d=10)
    - n=144 (d=12)
    - n=288 (d=18)
    - n=360 (d<=24)
    - n=756 (d<=34)

    Parameters
    ----------
    n : int
        The number of physical qubits. Must be one of the supported values.
    T : int
        The number of measurement rounds.
    p : float
        The physical error rate.

    Returns
    -------
    stim.Circuit
        The generated stim circuit for the specified BB code.
    """
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
        raise ValueError(f"Unsupported code size: {n}")

    code, A_list, B_list = create_bivariate_bicycle_codes(*args)
    circuit = build_BB_circuit_swd(
        code,
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
