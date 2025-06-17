import os
from typing import Optional, Tuple
import numpy as np
import stim
import random
from quits.ldpc_utility import (
    generate_ldpc,
    optimize_ldpc,
    has_duplicate_edges,
    binary_matrix_rank,
)
from quits.qldpc_code import HgpCode
from quits.circuit import get_qldpc_mem_circuit
from ldpc.code_util import compute_code_parameters


def generate_check_matrix(
    n: int,
    dv: int,
    dc: int,
    verbose: bool = False,
    initial_rounds: int = 100,
    additional_rounds: int = 10,
    max_depth: int = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an optimized LDPC parity-check matrix.

    This function creates an initial LDPC parity-check matrix and optimizes it
    to ensure no duplicate edges and full rank constraints are satisfied.

    Parameters
    ----------
    n : int
        Number of variable nodes
    dv : int
        Variable node degree
    dc : int
        Check node degree (m = n * dv / dc should be an integer)
    verbose : bool
        Whether to print optimization progress (default: False)
    initial_rounds : int
        Number of optimization rounds for initial optimization (default: 100)
    additional_rounds : int
        Number of optimization rounds for each additional optimization cycle (default: 10)
    max_depth : int
        Maximum depth parameter for the optimization algorithm (default: 10)
    seed : int, optional
        Random seed (default: None)

    Returns
    -------
    H : 2D numpy array of int
        Optimized LDPC parity-check matrix with shape (m, n) where m = n * dv / dc
    """
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate an initial parity-check matrix
    H = generate_ldpc(n, dv, dc)
    if verbose:
        print("Original parity-check matrix H:")
        print(H)

    # Perform initial optimization
    if verbose:
        print(f"\nOptimizing for {initial_rounds} rounds...")
    H = optimize_ldpc(H, initial_rounds, max_depth=max_depth)

    # Continue optimizing until constraints are satisfied
    while has_duplicate_edges(H) or binary_matrix_rank(H) < H.shape[0]:
        # Additional optimization rounds
        if verbose:
            print(f"\nOptimizing for another {additional_rounds} rounds...")
        H = optimize_ldpc(H, additional_rounds, max_depth=max_depth)

    if verbose:
        print(f"\nOptimized parity-check matrix H (rank {binary_matrix_rank(H)}):")
        print(H)

    # Reset random seed
    if seed is not None:
        random.seed(None)
        np.random.seed(None)

    return H


def compute_hgp_code_parameters(
    H1: np.ndarray,
    H2: np.ndarray,
    timeout_seconds: int = 1,
) -> Tuple[HgpCode, Tuple[int, int, int]]:
    """
    Compute code parameters for hypergraph product (HGP) code.

    Parameters
    ----------
    H1 : 2D numpy array of int
        First parity-check matrix for HGP code construction
    H2 : 2D numpy array of int
        Second parity-check matrix for HGP code construction
    timeout_seconds : int
        Timeout for distance computation (default: 1)

    Returns
    -------
    code : HgpCode
        HGP code object with built graph
    code_prms : tuple of ints
        (# data qubits, # logical qubits, estimated code distance)
    """
    # Create HGP code from the two parity-check matrices
    code = HgpCode(H1, H2)

    # Extract number of data qubits and logical qubits
    num_data = code.hz.shape[1]  # Number of data qubits
    num_logical = code.lz.shape[0]  # Number of logical qubits

    # Compute code distance for each classical code
    _, _, d1 = compute_code_parameters(H1, timeout_seconds=timeout_seconds)
    _, _, d2 = compute_code_parameters(H2, timeout_seconds=timeout_seconds)

    # Take minimum distance
    distance = min(d1, d2)

    return code, (num_data, num_logical, distance)


def build_hgp_circuit_from_code(
    code: HgpCode,
    p: float,
    num_rounds: int,
    basis: str = "Z",
    noisy_init: bool = True,
    noisy_meas: bool = False,
    seed: Optional[int] = None,
) -> stim.Circuit:
    """
    Build a hypergraph product (HGP) quantum error correction circuit from HgpCode.

    Parameters
    ----------
    code : HgpCode
        HGP code object with built graph
    p : float
        Physical error rate for all error mechanisms
    num_rounds : int
        Number of measurement rounds (T-1)
    basis : str
        Measurement basis, either 'Z' or 'X' (default: 'Z')
    noisy_init : bool
        Whether to include noise in initialization (default: True)
    noisy_meas : bool
        Whether to include noise in measurements (default: False)
    seed : int, optional
        Random seed for graph construction (default: None)
    Returns
    -------
    stim.Circuit
        Quantum error correction circuit for the HGP code
    """
    # Generate the quantum error correction circuit
    code.build_graph(seed=seed)
    circuit = stim.Circuit(
        get_qldpc_mem_circuit(
            code,
            p,
            p,
            p,
            p,
            num_rounds,
            basis=basis,
            noisy_init=noisy_init,
            noisy_meas=noisy_meas,
        )
    )

    return circuit


if __name__ == "__main__":
    # Define LDPC parameters
    n_cl = 12  # Number of variable nodes
    dv = 3  # Variable node degree
    dc = 4  # Check node degree
    required_code_prms = (225, 9, 6)

    p = 1e-3
    T = 12

    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_dir = os.path.join(current_dir, "data/hgp_prebuilt/circuits")
    check_matrix_dir = os.path.join(current_dir, "data/hgp_prebuilt/check_matrices")
    os.makedirs(circuit_dir, exist_ok=True)
    os.makedirs(check_matrix_dir, exist_ok=True)

    # Generate optimized check matrix
    counts = 0
    seed = 0
    while True:
        print(f"\n====== seed = {seed} (count = {counts}) ======")
        # Create check matrix
        H = generate_check_matrix(n_cl, dv, dc, verbose=False, seed=seed)
        code, code_prms = compute_hgp_code_parameters(H, H)
        n, k, d = code_prms
        print(f"code_prms = {code_prms}")
        if code_prms != required_code_prms:
            print("Skipped")
            seed += 1
            continue

        # Save check matrix
        check_matrix_subdir = os.path.join(
            check_matrix_dir, f"({dv},{dc})_n{n}_k{k}_d{d}"
        )
        os.makedirs(check_matrix_subdir, exist_ok=True)
        np.savetxt(os.path.join(check_matrix_subdir, f"seed{seed}.txt"), H, fmt="%d")

        # Create circuit
        circuit = build_hgp_circuit_from_code(code, p, T)

        # Save circuit
        circuit_subdir = os.path.join(
            circuit_dir, f"({dv},{dc})_n{n}_k{k}_d{d}_T{T}_p{p}"
        )
        os.makedirs(circuit_subdir, exist_ok=True)
        circuit.to_file(os.path.join(circuit_subdir, f"seed{seed}.stim"))

        print(f"Saved")

        counts += 1
        seed += 1
        if counts > 100:
            break
