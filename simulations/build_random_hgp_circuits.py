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


def build_hgp_circuit(
    H1: np.ndarray,
    H2: np.ndarray,
    p: float = 1e-3,
    num_rounds: int = 5,
    basis: str = "Z",
    seed: Optional[int] = None,
    noisy_init: bool = True,
    noisy_meas: bool = False,
    timeout_seconds: int = 1,
) -> Tuple[stim.Circuit, Tuple[int, int, int]]:
    """
    Build a hypergraph product (HGP) quantum error correction circuit.

    This function creates an HGP code from two classical parity-check matrices
    and generates the corresponding quantum error correction circuit using Stim.

    Parameters
    ----------
    H1 : 2D numpy array of int
        First parity-check matrix for HGP code construction
    H2 : 2D numpy array of int
        Second parity-check matrix for HGP code construction
    p : float
        Physical error rate for all error mechanisms (default: 1e-3)
    num_rounds : int
        Number of measurement rounds (T-1) (default: 5)
    basis : str
        Measurement basis, either 'Z' or 'X' (default: 'Z')
    seed : int, optional
        Random seed for graph construction (default: None)
    noisy_init : bool
        Whether to include noise in initialization (default: True)
    noisy_meas : bool
        Whether to include noise in measurements (default: False)
    timeout_seconds : int
        Timeout for distance computation (default: 1)

    Returns
    -------
    circuit : stim.Circuit
        Quantum error correction circuit for the HGP code
    code_prms : tuple of ints
        (# data qubits, # logical qubits, estimated code distance)
    """
    # Create HGP code from the two parity-check matrices
    code = HgpCode(H1, H2)
    code.build_graph(seed=seed)

    # Generate the quantum error correction circuit
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

    # Extract number of data qubits and logical qubits
    num_data = code.hz.shape[1]  # Number of data qubits
    num_logical = code.lz.shape[0]  # Number of logical qubits

    # Compute code distance for each classical code
    _, _, d1 = compute_code_parameters(H1, timeout_seconds=timeout_seconds)
    _, _, d2 = compute_code_parameters(H2, timeout_seconds=timeout_seconds)

    # Take minimum distance
    distance = min(d1, d2)

    code_prms = (num_data, num_logical, distance)

    return circuit, code_prms


if __name__ == "__main__":
    # Define LDPC parameters
    n_cl = 12  # Number of variable nodes
    dv = 3  # Variable node degree
    dc = 4  # Check node degree

    p = 1e-3
    T = 12

    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_dir = os.path.join(current_dir, "data/hgp_circuits")

    # Generate optimized check matrix
    counts = 0
    seed = 0
    while True:
        print(f"\n====== seed = {seed} (count = {counts}) ======")
        H = generate_check_matrix(n_cl, dv, dc, verbose=False, seed=seed)
        circuit, code_prms = build_hgp_circuit(H, H, p=p, num_rounds=T, seed=seed)
        n, k, d = code_prms
        print(f"code_prms = {code_prms}")
        if code_prms != (225, 9, 4):
            print("Skipped")
            seed += 1
            continue

        # Create directory structure: folder_name/seed{seed}.stim
        folder_name = f"({dv},{dc})_n{n}_k{k}_d{d}_T{T}_p{p}"
        circuit_subdir = os.path.join(circuit_dir, folder_name)
        os.makedirs(circuit_subdir, exist_ok=True)

        fname = f"seed{seed}.stim"
        circuit.to_file(os.path.join(circuit_subdir, fname))
        print(f"Saved: {folder_name}/{fname}")

        counts += 1
        seed += 1
        if counts > 100:
            break
