"""
Toric code bit-flip simulation using BP+LSD decoder.
"""

from math import e
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import glob
import re
import shutil

import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from ldpc.codes import ring_code
from bposd.hgp import hgp
from joblib import Parallel, delayed
from tqdm import tqdm

from ldpc_post_selection.bplsd_decoder import SoftOutputsBpLsdDecoder


# Default data directory (absolute path relative to this script)
DEFAULT_DATA_DIR = (
    Path(__file__).parent / "data" / "toric_minsum_iter30_lsd0_bitflip_raw"
)


def get_simulation_file_paths(
    d: int, p: float, data_dir: str | Path = DEFAULT_DATA_DIR
) -> tuple[Path, Path]:
    """
    Generate file paths for simulation data files.

    Parameters
    ----------
    d : int
        Code distance.
    p : float
        Bit-flip error rate.
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to DEFAULT_DATA_DIR.

    Returns
    -------
    fails_path : Path
        Path to the fails data file.
    clusters_path : Path
        Path to the clusters data file.
    """
    data_path = Path(data_dir)
    fails_filename = data_path / f"d{d}_p{p:.3f}_fails.npy"
    clusters_filename = data_path / f"d{d}_p{p:.3f}_clusters.npz"
    return fails_filename, clusters_filename


def generate_toric_code(d: int) -> tuple[csr_matrix, csr_matrix]:
    """
    Generate a toric code for bit-flip error simulation.

    Creates a toric code using the hypergraph product construction with ring codes.
    For bit-flip error simulation, only Z-type stabilizers and logical Z operators
    are relevant.

    Parameters
    ----------
    d : int
        The code distance of the toric code.

    Returns
    -------
    hz : scipy.sparse.csr_matrix
        The Z-type stabilizer check matrix with shape (n_z_checks, n_qubits).
    lz : scipy.sparse.csr_matrix
        The logical Z operator matrix with shape (2, n_qubits).
    """
    # Create ring codes for hypergraph product construction
    h1 = ring_code(d)
    h2 = ring_code(d)

    # Generate the quantum code using hypergraph product
    qcode = hgp(h1, h2, compute_distance=False)

    # Return Z-type check matrix and logical Z operators
    return qcode.hz, qcode.lz


def generate_bitflip_errors(
    n_qubits: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate bit-flip errors according to independent bit-flip noise model.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits.
    p : float
        Bit-flip error probability per qubit.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    errors : 1D numpy array of bool
        Binary error pattern with shape (n_qubits,).
    """
    return rng.random(n_qubits) < p


def compute_detector_outcomes(errors: np.ndarray, hz: csr_matrix) -> np.ndarray:
    """
    Compute detector outcomes (syndrome) from error pattern.

    Parameters
    ----------
    errors : 1D numpy array of bool
        Binary error pattern with shape (n_qubits,).
    hz : scipy.sparse.csr_matrix
        Z-type stabilizer check matrix with shape (n_checks, n_qubits).

    Returns
    -------
    detector_outcomes : 1D numpy array of bool
        Detector measurement outcomes with shape (n_checks,).
    """
    return (hz @ errors.astype(np.uint8)) % 2 == 1


def compute_logical_outcomes(errors: np.ndarray, lz: csr_matrix) -> np.ndarray:
    """
    Compute logical outcomes from error pattern.

    Parameters
    ----------
    errors : 1D numpy array of bool
        Binary error pattern with shape (n_qubits,).
    lz : scipy.sparse.csr_matrix
        Logical Z operator matrix with shape (2, n_qubits).

    Returns
    -------
    logical_outcomes : 1D numpy array of bool
        Logical measurement outcomes with shape (2,).
    """
    return (lz @ errors.astype(np.uint8)) % 2 == 1


def check_existing_data(
    d: int, p: float, data_dir: str | Path = DEFAULT_DATA_DIR
) -> tuple[int, np.ndarray | None, csr_matrix | None]:
    """
    Check for existing simulation data and return current shot count and data.

    Parameters
    ----------
    d : int
        Code distance.
    p : float
        Bit-flip error rate.
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to DEFAULT_DATA_DIR.

    Returns
    -------
    existing_shots : int
        Number of existing shots (0 if no data exists).
    existing_fails : numpy.ndarray or None
        Existing failure data, or None if no data exists.
    existing_clusters : scipy.sparse.csr_matrix or None
        Existing cluster data, or None if no data exists.
    """
    try:
        existing_fails, existing_clusters = load_single_simulation_data(d, p, data_dir)
        return len(existing_fails), existing_fails, existing_clusters
    except FileNotFoundError:
        return 0, None, None


def run_incremental_toric_bitflip_simulation(
    d: int,
    p: float,
    target_shots: int,
    *,
    max_iter: int = 30,
    bp_method: str = "minimum_sum",
    lsd_method: str = "LSD_0",
    lsd_order: int = 0,
    ms_scaling_factor: float = 1.0,
    include_cluster_stats: bool = True,
    save_data: bool = True,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    seed: int | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run incremental toric code bit-flip error simulation using BP+LSD decoder.

    This function checks for existing data and only runs additional shots as needed
    to reach the target number of shots, then combines old and new data.

    Parameters
    ----------
    d : int
        Code distance of the toric code.
    p : float
        Bit-flip error probability per qubit.
    target_shots : int
        Target total number of simulation shots.
    max_iter : int, optional
        Maximum BP iterations. Defaults to 30.
    bp_method : str, optional
        BP method ('product_sum' or 'minimum_sum'). Defaults to "minimum_sum".
    lsd_method : str, optional
        LSD method ('LSD_0', 'LSD_E', 'LSD_CS'). Defaults to "LSD_0".
    lsd_order : int, optional
        LSD order parameter. Defaults to 0.
    ms_scaling_factor : float, optional
        Min-sum scaling factor. Defaults to 1.0.
    include_cluster_stats : bool, optional
        Whether to compute cluster statistics. Defaults to True.
    save_data : bool, optional
        Whether to save simulation data to files. Defaults to True.
    data_dir : str or Path, optional
        Directory to save data. Defaults to DEFAULT_DATA_DIR.
    seed : int, optional
        Random seed. If None, uses random seed.
    verbose : bool, optional
        Whether to print progress information. Defaults to False.

    Returns
    -------
    summary : dict
        A dictionary containing simulation summary statistics.
    """
    # Check for existing data
    existing_shots, existing_fails, existing_clusters = check_existing_data(
        d, p, data_dir
    )

    if existing_shots >= target_shots:
        if verbose:
            print(
                f"d={d}, p={p:.6f}: Already has {existing_shots} shots (>= {target_shots} target). Skipping."
            )
        # Return summary based on existing data
        failure_rate = (
            np.mean(existing_fails[:target_shots])
            if existing_fails is not None
            else 0.0
        )
        return {
            "d": d,
            "p": p,
            "shots": target_shots,
            "failure_rate": failure_rate,
            "existing_shots": existing_shots,
            "new_shots": 0,
        }

    additional_shots = target_shots - existing_shots

    if verbose:
        print(
            f"d={d}, p={p:.6f}: Found {existing_shots} existing shots, running {additional_shots} additional shots"
        )

    # Generate toric code
    hz, lz = generate_toric_code(d)
    n_qubits = hz.shape[1]

    if verbose:
        print(f"Code parameters: n_qubits={n_qubits}, n_checks={hz.shape[0]}")

    # Set up decoder with uniform bit-flip priors
    priors = np.full(n_qubits, p, dtype=float)
    decoder = SoftOutputsBpLsdDecoder(
        H=hz,
        p=priors,
        obs_matrix=lz,
        max_iter=max_iter,
        bp_method=bp_method,
        lsd_method=lsd_method,
        lsd_order=lsd_order,
        ms_scaling_factor=ms_scaling_factor,
    )

    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Initialize result accumulators for new data
    new_fails = []
    new_cluster_matrices = []

    # Use tqdm for progress bar if verbose
    iterable = range(additional_shots)
    if verbose:
        iterable = tqdm(
            iterable, desc=f"Additional shots d={d}, p={p:.3f}", unit="shot"
        )

    # Run additional simulation shots
    for shot in iterable:
        # Generate bit-flip error
        errors = generate_bitflip_errors(n_qubits, p, rng)

        # Compute detector and logical outcomes
        detector_outcomes = compute_detector_outcomes(errors, hz)
        true_logical_outcomes = compute_logical_outcomes(errors, lz)

        # Decode
        pred, pred_bp, converge, soft_outputs = decoder.decode(
            detector_outcomes=detector_outcomes,
            include_cluster_stats=include_cluster_stats,
            verbose=False,
        )

        # Compute predicted logical outcomes
        pred_logical_outcomes = compute_logical_outcomes(pred, lz)

        # Determine if decoding failed (any logical error)
        fail = not np.array_equal(true_logical_outcomes, pred_logical_outcomes)

        # Accumulate results
        new_fails.append(fail)

        if include_cluster_stats and soft_outputs.get("clusters") is not None:
            new_cluster_matrices.append(
                csr_matrix(soft_outputs["clusters"], dtype="uint16")
            )

    # Combine old and new data
    if existing_shots > 0:
        # Combine fails data
        new_fails_array = np.array(new_fails, dtype=bool)
        combined_fails = np.concatenate([existing_fails, new_fails_array])

        # Combine cluster data if it exists
        combined_clusters = None
        if existing_clusters is not None and new_cluster_matrices:
            combined_clusters = vstack([existing_clusters] + new_cluster_matrices)
        elif existing_clusters is not None:
            combined_clusters = existing_clusters
        elif new_cluster_matrices:
            combined_clusters = vstack(new_cluster_matrices)
    else:
        # No existing data, use only new data
        combined_fails = np.array(new_fails, dtype=bool)
        combined_clusters = (
            vstack(new_cluster_matrices) if new_cluster_matrices else None
        )

    failure_rate = np.mean(combined_fails) if len(combined_fails) > 0 else 0.0

    if verbose:
        print(
            f"Simulation completed. Total shots: {len(combined_fails)}, Failure rate: {failure_rate:.6f}"
        )

    # Save combined data if requested
    if save_data:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        # Get file paths using helper function
        fails_filename, clusters_filename = get_simulation_file_paths(d, p, data_dir)

        # Save results
        np.save(fails_filename, combined_fails)
        save_npz(clusters_filename, combined_clusters)

    # Return summary statistics
    summary = {
        "d": d,
        "p": p,
        "shots": len(combined_fails),
        "failure_rate": failure_rate,
        "existing_shots": existing_shots,
        "new_shots": additional_shots,
    }
    return summary


def load_single_simulation_data(
    d: int,
    p: float,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, csr_matrix | None]:
    """
    Load simulation data for a single (d, p) combination.

    Parameters
    ----------
    d : int
        Code distance.
    p : float
        Bit-flip error rate.
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to DEFAULT_DATA_DIR.

    Returns
    -------
    fails : numpy.ndarray
        1D boolean array of failure outcomes.
    clusters : scipy.sparse.csr_matrix or None
        2D sparse matrix of cluster data, or None if not found.

    Raises
    ------
    FileNotFoundError
        If the fails data file for the specified (d, p) is not found.
    """
    # Get file paths using helper function
    fails_filename, clusters_filename = get_simulation_file_paths(d, p, data_dir)

    if not fails_filename.exists():
        raise FileNotFoundError(f"Data file not found: {fails_filename}")

    fails = np.load(fails_filename)
    clusters = None
    if clusters_filename.exists():
        clusters = load_npz(clusters_filename)

    return fails, clusters


def get_existing_simulation_params(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> List[tuple[int, float]]:
    """
    Scans the data directory to find all existing simulation parameter combinations.

    This function looks for files matching the pattern 'd{d}_p{p}_fails.npy'
    to identify which simulations have already been run and saved.

    Parameters
    ----------
    data_dir : str or Path, optional
        The directory where the simulation data is stored.
        Defaults to DEFAULT_DATA_DIR.

    Returns
    -------
    list of tuple
        A sorted list of unique (d, p) tuples corresponding to the
        existing simulation data.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return []

    pattern = str(data_path / "d*_p*_fails.npy")
    files = glob.glob(pattern)

    if not files:
        print(f"No data files found in {data_path}")
        return []

    params = set()
    filename_pattern = r"d(\d+)_p(\d+\.\d{3})_fails\.npy"

    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.match(filename_pattern, filename)
        if match:
            d = int(match.group(1))
            p = float(match.group(2))
            params.add((d, p))

    return sorted(list(params))


def run_single_simulation(d: int, p: float, target_shots: int) -> Dict[str, Any]:
    """
    Run a single incremental simulation for given distance and error rate.

    This function uses the incremental approach to check for existing data
    and only run additional shots as needed.

    Parameters
    ----------
    d : int
        Code distance.
    p : float
        Bit-flip error rate.
    target_shots : int
        Target total number of simulation shots.

    Returns
    -------
    result : dict
        Dictionary containing simulation parameters and summary statistics.
    """
    result = run_incremental_toric_bitflip_simulation(
        d=d,
        p=p,
        target_shots=target_shots,
        max_iter=30,
        bp_method="minimum_sum",
        lsd_method="LSD_0",
        lsd_order=0,
        ms_scaling_factor=1.0,
        include_cluster_stats=True,
        save_data=True,
        verbose=False,  # Disable verbose to reduce output for parallel runs
        seed=None,
    )
    return result


if __name__ == "__main__":
    print("--- Starting incremental toric code simulation ---")

    # Parameter ranges
    distances = np.arange(5, 41, 4)
    # error_rates = np.arange(0.05, 0.101, 0.005)
    error_rates = np.arange(0.01, 0.111, 0.005).round(3)
    target_shots = 1000000

    print(f"Distance values: {distances}")
    print(f"Error rate values: {error_rates}")
    print(f"Target shots per simulation: {target_shots}")
    print(
        f"Total parameter combinations: {len(distances)} × {len(error_rates)} = {len(distances) * len(error_rates)}"
    )

    # Check existing data status
    print("\n--- Checking existing data ---")
    existing_params = get_existing_simulation_params()
    if existing_params:
        print(f"Found existing data for {len(existing_params)} parameter combinations")
        # Show a sample of existing data with shot counts
        sample_size = min(5, len(existing_params))
        for i, (d, p) in enumerate(existing_params[:sample_size]):
            existing_shots, _, _ = check_existing_data(d, p)
            print(f"  d={d}, p={p:.3f}: {existing_shots} shots")
        if len(existing_params) > sample_size:
            print(f"  ... and {len(existing_params) - sample_size} more")
    else:
        print("No existing data found.")

    # Generate all parameter combinations
    param_combinations = [(d, p) for d in distances for p in error_rates]

    print(f"\nRunning simulations with 18 parallel jobs...")

    # Run simulations in parallel
    results = Parallel(n_jobs=18)(
        delayed(run_single_simulation)(d, p, target_shots)
        for d, p in tqdm(param_combinations)
    )

    # Print summary
    print(f"\nAll simulations completed!")
    print(f"Total parameter combinations processed: {len(results)}")

    # Count statistics
    existing_data_count = sum(1 for r in results if r.get("existing_shots", 0) > 0)
    new_shots_total = sum(r.get("new_shots", r["shots"]) for r in results)
    total_shots = sum(r["shots"] for r in results)

    print(f"Parameter combinations with existing data: {existing_data_count}")
    print(
        f"Parameter combinations with new data only: {len(results) - existing_data_count}"
    )
    print(f"Total new shots run: {new_shots_total:,}")
    print(f"Total shots in dataset: {total_shots:,}")

    print("--- Simulation finished ---")
