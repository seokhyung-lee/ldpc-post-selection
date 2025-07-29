import heapq
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, cpu_count, delayed
from scipy import stats
from scipy.ndimage import label


def generate_lattice(d: int, p: float) -> np.ndarray:
    """
    Generates a 3D cubic lattice with occupied sites.

    Parameters
    ----------
    d : int
        The dimension of the cubic lattice (d x d x d).
    p : float
        The probability that a site is occupied.

    Returns
    -------
    lattice : 3D numpy array of bool with shape (d, d, d)
        The lattice where True represents an occupied site and False a vacant site.
    """
    return np.random.rand(d, d, d) < p


def calculate_x_alpha(lattice: np.ndarray, alpha: float) -> float:
    """
    Calculates the X_alpha metric for a given lattice.

    X_alpha = (sum_i |C_i|^alpha)^(1/alpha), where C_i are clusters.
    If alpha is np.inf, X_alpha is defined as the size of the largest cluster.

    Parameters
    ----------
    lattice : 3D numpy array of bool with shape (d, d, d)
        The lattice with occupied (True) and vacant (False) sites.
    alpha : float
        The exponent alpha > 0, or np.inf.

    Returns
    -------
    x_alpha : float
        The calculated X_alpha value. Returns 0 if no clusters are found.
    """
    if not lattice.any():  # No occupied sites
        return 0.0

    # Use 3x3x3 connectivity
    labeled_lattice, num_clusters = label(
        lattice, structure=np.ones((3, 3, 3), dtype=bool)
    )

    if num_clusters == 0:
        return 0.0

    cluster_sizes = np.bincount(
        labeled_lattice.ravel()
    )  # includes background (label 0)
    occupied_cluster_sizes = cluster_sizes[1:]  # Exclude background

    if occupied_cluster_sizes.size == 0:
        return 0.0

    if alpha == np.inf:
        # X_inf is the size of the largest cluster
        x_alpha = np.max(occupied_cluster_sizes)
    elif alpha <= 0:
        # As per the problem description, alpha > 0. Handle invalid alpha.
        # Alternatively, could raise ValueError.
        # For now, returning NaN or a specific error indicator might be appropriate.
        # Let's return NaN for consistency if an invalid alpha is somehow passed, though
        # the problem defines alpha > 0 or alpha = inf (handled above).
        print(
            f"Warning: alpha = {alpha} is not > 0. X_alpha calculation may be ill-defined."
        )
        return np.nan
    else:
        # Calculate sum(|C_i|^alpha)
        sum_sizes_alpha = np.sum(occupied_cluster_sizes**alpha)

        # Calculate (sum(|C_i|^alpha))^(1/alpha)
        x_alpha = sum_sizes_alpha ** (1.0 / alpha)

    return float(x_alpha)  # Ensure it's a float, max can return int


def calculate_y(lattice: np.ndarray) -> int | float:
    """
    Calculates the minimum number of vacant sites on a spanning path.

    Finds the minimum cost path from the face z=0 to the face z=d-1,
    where cost = number of vacant sites (occupied=0, vacant=1).
    Uses Dijkstra's algorithm on the grid.

    Parameters
    ----------
    lattice : 3D numpy array of bool with shape (d, d, d)
        The lattice with occupied (True) and vacant (False) sites.

    Returns
    -------
    min_vacant_sites : int or None
        The minimum number of vacant sites on any spanning path. Returns
        infinity if no spanning path exists.
    """
    d = lattice.shape[0]
    if d == 0:
        return float("inf")  # Or raise error, depends on desired behavior

    # Priority queue stores tuples: (cost, x, y, z)
    pq: List[Tuple[int, int, int, int]] = []

    # Dictionary to store the minimum cost found so far to reach (x, y, z)
    min_costs: Dict[Tuple[int, int, int], float] = {}

    # Initialize the search from the z=0 face
    for x in range(d):
        for y in range(d):
            z = 0
            cost = 0 if lattice[x, y, z] else 1  # Cost is 1 if vacant, 0 if occupied
            heapq.heappush(pq, (cost, x, y, z))
            min_costs[(x, y, z)] = cost

    min_spanning_cost = float("inf")

    while pq:
        cost, x, y, z = heapq.heappop(pq)

        # If we found a shorter path already, skip
        if cost > min_costs.get((x, y, z), float("inf")):
            continue

        # If we reached the target face (z=d-1)
        if z == d - 1:
            min_spanning_cost = min(min_spanning_cost, cost)
            # Don't stop early, need the minimum across all paths to the face
            # Continue processing neighbors in case a path through here leads
            # to an even cheaper path ending elsewhere on the z=d-1 face later.

        # Explore neighbors (6 neighbors in 3D)
        for dx, dy, dz in [
            (0, 0, 1),
            (0, 0, -1),
            (0, 1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (-1, 0, 0),
        ]:
            nx, ny, nz = x + dx, y + dy, z + dz

            # Check bounds
            if 0 <= nx < d and 0 <= ny < d and 0 <= nz < d:
                neighbor_cost = 0 if lattice[nx, ny, nz] else 1
                new_cost = cost + neighbor_cost

                # If this path is cheaper than any known path to the neighbor
                if new_cost < min_costs.get((nx, ny, nz), float("inf")):
                    min_costs[(nx, ny, nz)] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny, nz))

    return min_spanning_cost


def _run_single_simulation(
    d: int, p: float, alpha_values: List[float]
) -> Tuple[float, Dict[float, float], Dict[float, float]]:
    """
    Performs a single run of the simulation: generates one lattice,
    calculates Y, all X_alpha values, and all X_gap_alpha values.

    Parameters
    ----------
    d : int
        Lattice dimension.
    p : float
        Occupation probability.
    alpha_values : list of float
        Alpha values to compute X_alpha and X_gap_alpha for.

    Returns
    -------
    y_val : float
        The calculated Y value for this lattice.
    x_results : dict
        Dictionary mapping alpha (float or np.inf) to its calculated X_alpha value.
    x_gap_results : dict
        Dictionary mapping alpha (float or np.inf) to its calculated X_gap_alpha value.
    """
    lattice = generate_lattice(d, p)
    y_val = calculate_y(lattice)

    total_sites = d * d * d
    # vacant_sites = total_sites - np.sum(lattice) # np.sum(lattice) is num_occupied
    # More direct: vacant_sites is number of False values
    vacant_sites = np.sum(~lattice)

    x_results = {}
    x_gap_results = {}
    for alpha in alpha_values:
        x_alpha_val = calculate_x_alpha(lattice, alpha)
        x_results[alpha] = x_alpha_val
        x_gap_results[alpha] = (
            float(vacant_sites) - x_alpha_val
        )  # Ensure float arithmetic

    return y_val, x_results, x_gap_results


def run_simulation(
    d: int, p: float, alpha_values: List[float], num_simulations: int
) -> None:
    """
    Runs the percolation simulation multiple times in parallel using joblib
    and saves the raw results (Y, X_alpha, X_gap_alpha) and metadata to a file.

    Parameters
    ----------
    d : int
        The dimension of the cubic lattice (d x d x d).
    p : float
        The probability that a site is occupied.
    alpha_values : list of float
        A list of alpha values to test for X_alpha.
    num_simulations : int
        The number of simulation runs to perform.

    Returns
    -------
    None
    """
    # Initialize results dictionaries
    results_x_alpha: Dict[str, np.ndarray] = {
        (str(alpha) if alpha != np.inf else "inf"): np.zeros(num_simulations)
        for alpha in alpha_values
    }
    results_x_gap_alpha: Dict[str, np.ndarray] = {  # For X_gap_alpha
        (str(alpha) if alpha != np.inf else "inf"): np.zeros(num_simulations)
        for alpha in alpha_values
    }
    results_y: np.ndarray = np.zeros(num_simulations)

    n_jobs = -1
    print(
        f"Running {num_simulations} simulations in parallel using {cpu_count()} cores (d={d}, p={p:.3f})..."
    )

    parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_simulation)(d, p, alpha_values)
        for _ in range(num_simulations)
    )

    print("\nParallel simulations complete. Processing results...")
    for i, (y_val, x_results_dict, x_gap_results_dict) in enumerate(parallel_results):
        results_y[i] = y_val
        for alpha, x_val in x_results_dict.items():
            alpha_key = str(alpha) if alpha != np.inf else "inf"
            results_x_alpha[alpha_key][i] = x_val
        for alpha, x_gap_val in x_gap_results_dict.items():  # Process X_gap
            alpha_key = str(alpha) if alpha != np.inf else "inf"
            results_x_gap_alpha[alpha_key][i] = x_gap_val

    print("Saving results...")
    script_dir = os.path.dirname(__file__)
    save_dir = os.path.join(script_dir, "data", "simple_lattice_percolation_simulation")
    filename = f"percolation_d{d}_p{p:.4f}_runs{num_simulations}.npz"
    save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    alpha_keys_for_saving = [str(a) if a != np.inf else "inf" for a in alpha_values]
    save_data = {
        "d": np.array([d]),
        "p": np.array([p]),
        "num_simulations": np.array([num_simulations]),
        "alpha_values_str": np.array(alpha_keys_for_saving, dtype=str),
        "results_y": results_y,
    }
    for alpha in alpha_values:
        alpha_key = str(alpha) if alpha != np.inf else "inf"
        save_data[f"results_x_alpha_{alpha_key}"] = results_x_alpha[alpha_key]
        save_data[f"results_x_gap_alpha_{alpha_key}"] = results_x_gap_alpha[
            alpha_key
        ]  # Save X_gap

    np.savez(save_path, **save_data)
    print(f"Results saved to: {save_path}")


def load_simulation_data(d: int, p: float, num_runs: int) -> Optional[Dict[str, Any]]:
    """
    Loads simulation results from a .npz file found using d, p, num_runs.

    Constructs the filepath based on convention, loads the data, and reconstructs
    the necessary data structures (like alpha_values as floats/inf).

    Parameters
    ----------
    d : int
        The dimension of the cubic lattice (d x d x d).
    p : float
        The probability that a site is occupied.
    num_runs : int
        The number of simulation runs.

    Returns
    -------
    loaded_data : dict or None
        A dictionary containing the loaded data ('d', 'p', 'num_runs',
        'alpha_values', 'results_x_alpha', 'results_x_gap_alpha', 'results_y')
        if successful, otherwise None.
    """
    # Construct filepath
    script_dir = os.path.dirname(__file__)
    base_data_dir = os.path.join(
        script_dir, "data", "simple_lattice_percolation_simulation"
    )
    filename = f"percolation_d{d}_p{p:.4f}_runs{num_runs}.npz"
    filepath = os.path.join(base_data_dir, filename)
    print(f"Attempting to load data from: {filepath}")

    try:
        data = np.load(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        print(
            "Please ensure the simulation was run with these exact parameters and the file exists."
        )
        return None
    except Exception as e:
        print(f"Error loading data file {filepath}: {e}")
        return None

    # Extract metadata and results
    try:
        loaded_d = int(data["d"][0])
        loaded_p = float(data["p"][0])
        loaded_num_runs = int(data["num_simulations"][0])
        alpha_values_str = data["alpha_values_str"]
        results_y = data["results_y"]

        # Optional: Verify loaded parameters match requested parameters
        if loaded_d != d or abs(loaded_p - p) > 1e-6 or loaded_num_runs != num_runs:
            print(
                f"Warning: Metadata in file ({loaded_d}, {loaded_p:.4f}, {loaded_num_runs}) "
                f"does not exactly match requested parameters ({d}, {p:.4f}, {num_runs})."
            )

        # Reconstruct alpha values (float/inf) and results_x_alpha dict
        alpha_values = []
        results_x_alpha = {}
        results_x_gap_alpha = {}  # For X_gap_alpha
        for alpha_str in alpha_values_str:
            alpha = np.inf if alpha_str == "inf" else float(alpha_str)
            alpha_values.append(alpha)
            results_x_alpha[alpha] = data[f"results_x_alpha_{alpha_str}"]
            results_x_gap_alpha[alpha] = data[
                f"results_x_gap_alpha_{alpha_str}"
            ]  # Load X_gap

        return {
            "d": loaded_d,
            "p": loaded_p,
            "num_runs": loaded_num_runs,
            "alpha_values": alpha_values,
            "results_x_alpha": results_x_alpha,
            "results_x_gap_alpha": results_x_gap_alpha,  # Add X_gap to return
            "results_y": results_y,
        }

    except KeyError as e:
        print(
            f"Error: Missing expected key '{e}' in data file {filepath}. File might be corrupted or incomplete."
        )
        return None
    except Exception as e:
        print(f"Error processing data from file {filepath}: {e}")
        return None


def plot_correlation_vs_alpha(
    d: int, p: float, num_runs: int, ci_alpha: float = 0.05, use_gap: bool = True
):
    """
    Loads results, calculates correlations (X_alpha vs Y OR X_gap_alpha vs Y)
    with CIs, and plots correlation vs. alpha with error bars.

    Parameters
    ----------
    d : int
        Lattice dimension.
    p : float
        Occupation probability.
    num_runs : int
        Number of simulation runs.
    ci_alpha : float, optional
        Significance level for the confidence interval (default: 0.05).
    use_gap : bool, optional
        If True, plot correlation of X_gap_alpha vs Y.
        If False, plot correlation of X_alpha vs Y (default: True).
    """
    loaded_data = load_simulation_data(d, p, num_runs)
    if loaded_data is None:
        return

    alpha_values = loaded_data["alpha_values"]
    results_y = loaded_data["results_y"]
    d_loaded = loaded_data["d"]
    p_loaded = loaded_data["p"]
    num_runs_loaded = loaded_data["num_runs"]

    # Select the target X data based on use_gap
    if use_gap:
        target_x_results = loaded_data.get("results_x_gap_alpha")
        x_metric_name = "X_gap,α"
        if target_x_results is None:
            print(
                "Error: X_gap_alpha results not found in the loaded data. Cannot plot gap correlation."
            )
            return
    else:
        target_x_results = loaded_data.get("results_x_alpha")
        x_metric_name = "X_α"
        if target_x_results is None:
            print("Error: X_alpha results not found in the loaded data.")
            return

    # Helper for CI calculation (Fisher z-transformation)
    def correlation_ci_fisher(x_data, y_data, conf_level=0.05):
        if len(x_data) < 4:
            r = (
                np.corrcoef(x_data, y_data)[0, 1]
                if len(x_data) > 1 and np.std(x_data) > 1e-9 and np.std(y_data) > 1e-9
                else np.nan
            )
            return r, (np.nan, np.nan)
        r = np.corrcoef(x_data, y_data)[0, 1]
        if np.isnan(r):
            return r, (np.nan, np.nan)
        z = np.arctanh(r)
        se = 1 / np.sqrt(len(x_data) - 3)
        z_crit = stats.norm.ppf(1 - conf_level / 2)
        lo_z, hi_z = z - z_crit * se, z + z_crit * se
        lo_r, hi_r = np.tanh([lo_z, hi_z])
        return r, (lo_r, hi_r)

    correlations: Dict[float, float] = {}
    ci_bounds: Dict[float, Tuple[float, float]] = {}
    valid_indices = np.isfinite(results_y)

    if not np.any(valid_indices):
        print(
            f"Warning: No finite Y values. Cannot calculate {x_metric_name} vs Y correlations."
        )
    else:
        valid_y_for_corr = results_y[valid_indices]
        num_valid_runs_from_data = len(valid_y_for_corr)
        if num_valid_runs_from_data < num_runs_loaded:
            print(
                f"Warning: Correlation calculated using {num_valid_runs_from_data}/{num_runs_loaded} runs."
            )

        for alpha_val in alpha_values:
            # Use the selected target_x_results
            valid_x_data_for_corr = target_x_results[alpha_val][valid_indices]

            if (
                np.std(valid_x_data_for_corr) > 1e-9
                and np.std(valid_y_for_corr) > 1e-9
                and len(valid_x_data_for_corr) > 3
            ):
                r_hat, (lo, hi) = correlation_ci_fisher(
                    valid_x_data_for_corr, valid_y_for_corr, conf_level=ci_alpha
                )
                correlations[alpha_val] = r_hat
                ci_bounds[alpha_val] = (lo, hi)
            else:
                r_hat = np.nan
                if (
                    len(valid_x_data_for_corr) > 1
                    and np.std(valid_x_data_for_corr) > 1e-9
                    and np.std(valid_y_for_corr) > 1e-9
                ):
                    r_hat = np.corrcoef(valid_x_data_for_corr, valid_y_for_corr)[0, 1]
                correlations[alpha_val] = r_hat
                ci_bounds[alpha_val] = (np.nan, np.nan)

    if not correlations or all(np.isnan(v) for v in correlations.values()):
        print(
            f"Skipping correlation plot: No valid {x_metric_name} vs Y correlation data calculated."
        )
        return

    # --- Plotting --- (Use x_metric_name in labels/titles)
    plot_alphas = sorted(correlations.keys())
    finite_alphas = [a for a in plot_alphas if a != np.inf]
    inf_alpha_present = np.inf in plot_alphas

    plt.figure(figsize=(10, 6))

    if finite_alphas:
        y_values_finite = [correlations[a] for a in finite_alphas]
        ci_low_finite = [ci_bounds[a][0] for a in finite_alphas]
        ci_high_finite = [ci_bounds[a][1] for a in finite_alphas]
        y_err_finite = [
            np.array(y_values_finite) - np.array(ci_low_finite),
            np.array(ci_high_finite) - np.array(y_values_finite),
        ]
        valid_err_indices = ~np.isnan(y_err_finite[0]) & ~np.isnan(y_err_finite[1])
        plt.errorbar(
            np.array(finite_alphas)[valid_err_indices],
            np.array(y_values_finite)[valid_err_indices],
            yerr=np.array(y_err_finite)[:, valid_err_indices],
            fmt="o-",
            capsize=5,
            label=f"Finite α ({(1-ci_alpha)*100:.0f}% CI)",
            elinewidth=1.5,
            markeredgewidth=1.5,
        )
        nan_err_indices = ~valid_err_indices
        if np.any(nan_err_indices):
            plt.plot(
                np.array(finite_alphas)[nan_err_indices],
                np.array(y_values_finite)[nan_err_indices],
                "o",
                color="gray",
                label="Finite α (no CI)",
            )

    if inf_alpha_present and np.inf in correlations:
        r_inf = correlations[np.inf]
        lo_inf, hi_inf = ci_bounds.get(np.inf, (np.nan, np.nan))
        inf_plot_x = max(finite_alphas) + 0.5 if finite_alphas else 1.0
        if not np.isnan(r_inf):
            if not np.isnan(lo_inf) and not np.isnan(hi_inf):
                y_err_inf = [[r_inf - lo_inf], [hi_inf - r_inf]]
                plt.errorbar(
                    inf_plot_x,
                    r_inf,
                    yerr=y_err_inf,
                    fmt="s",
                    capsize=5,
                    label=f"α = ∞ ({(1-ci_alpha)*100:.0f}% CI)",
                    color="red",
                    elinewidth=1.5,
                    markeredgewidth=1.5,
                )
            else:
                plt.plot(inf_plot_x, r_inf, "s", label="α = ∞ (no CI)", color="red")
            plt.text(
                inf_plot_x + 0.05,
                r_inf,
                f" ∞ (corr={r_inf:+.3f})",
                verticalalignment="center",
            )

    plt.xlabel("Alpha (α)")
    plt.ylabel(f"Pearson Correlation({x_metric_name}, Y)")  # Updated label
    plt.title(
        f"Correlation between {x_metric_name} and Y vs. α\n(d={d_loaded}, p={p_loaded:.4f}, runs={num_runs_loaded})"
    )  # Updated title
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()


def plot_distributions(
    d: int, p: float, num_runs: int, show_scatter: bool = False, use_gap: bool = True
):
    """
    Loads results and creates joint plots (KDE + histograms).
    Uses X_alpha vs Y or X_gap_alpha vs Y based on use_gap.
    Optionally overlays individual data points.

    Parameters
    ----------
    d : int
        Lattice dimension.
    p : float
        Occupation probability.
    num_runs : int
        Number of simulation runs.
    show_scatter : bool, optional
        If True, overlay individual data points (default: False).
    use_gap : bool, optional
        If True, plot X_gap_alpha vs Y.
        If False, plot X_alpha vs Y (default: True).
    """
    loaded_data = load_simulation_data(d, p, num_runs)
    if loaded_data is None:
        return

    alpha_values = loaded_data["alpha_values"]
    results_y = loaded_data["results_y"]
    d_loaded = loaded_data["d"]
    p_loaded = loaded_data["p"]
    num_runs_loaded = loaded_data["num_runs"]

    # Select the target X data based on use_gap
    if use_gap:
        target_x_results = loaded_data.get("results_x_gap_alpha")
        x_metric_name = "X_gap,α"
        if target_x_results is None:
            print(
                "Error: X_gap_alpha results not found. Cannot plot gap distributions."
            )
            return
    else:
        target_x_results = loaded_data.get("results_x_alpha")
        x_metric_name = "X_α"
        if target_x_results is None:
            print("Error: X_alpha results not found. Cannot plot distributions.")
            return

    valid_indices = np.isfinite(results_y)
    if not np.any(valid_indices):
        print("Skipping distribution plots: No finite Y values found.")
        return

    valid_y = results_y[valid_indices]
    num_valid_runs_from_data = len(valid_y)

    print(
        f"\nGenerating {x_metric_name} vs Y distribution plots (d={d_loaded}, p={p_loaded:.4f}, runs={num_runs_loaded}) - {num_valid_runs_from_data} valid runs..."
    )

    for alpha in sorted(alpha_values):
        if alpha not in target_x_results:
            alpha_display = "∞" if alpha == np.inf else f"{alpha:.2f}"
            print(
                f"  Skipping plot for alpha={alpha_display}: Data not found in target_x_results."
            )
            continue

        # Use the selected target_x_results for this alpha
        valid_x_data = target_x_results[alpha][valid_indices]

        if np.std(valid_x_data) < 1e-9 or np.std(valid_y) < 1e-9:
            alpha_display = "∞" if alpha == np.inf else f"{alpha:.2f}"
            print(f"  Skipping plot for alpha={alpha_display}: Insufficient variance.")
            continue

        try:
            with plt.style.context("seaborn-v0_8-whitegrid"):
                alpha_display = "∞" if alpha == np.inf else f"{alpha:.2f}"
                g = sns.jointplot(x=valid_x_data, y=valid_y, kind="kde", height=5)
                # Update axis labels using x_metric_name
                g.set_axis_labels(
                    f"{x_metric_name} (α={alpha_display})", "Y (Min Vacant Sites)"
                )
                # Update title using x_metric_name
                g.fig.suptitle(
                    f"Joint Distribution of {x_metric_name} and Y\n(d={d_loaded}, p={p_loaded:.4f}, α={alpha_display}, runs={num_runs_loaded})"
                )
                g.fig.subplots_adjust(top=0.92)

                if show_scatter:
                    g.ax_joint.scatter(
                        valid_x_data,
                        valid_y,
                        s=10,
                        alpha=0.3,
                        color="blue",
                        label="Samples",
                    )

        except Exception as e:
            alpha_display = "∞" if alpha == np.inf else f"{alpha:.2f}"
            print(f"  Could not generate joint plot for alpha={alpha_display}: {e}")


if __name__ == "__main__":
    # Define simulation parameters
    D_SIZE = 13
    PROBABILITY = 0.05
    ALPHA_VALS = [0.5, 1.0, 1.5, 2.0, 2.5, np.inf]
    NUM_RUNS = 100000

    print(
        f"Starting simulation with: d={D_SIZE}, p={PROBABILITY:.4f}, num_runs={NUM_RUNS}"
    )
    print(f"Alpha values to be tested: {ALPHA_VALS}")

    # Run simulation (which now saves data and doesn't return)
    run_simulation(D_SIZE, PROBABILITY, ALPHA_VALS, NUM_RUNS)

    print("\nSimulation script finished. Data saved.")
    print("To visualize the results, load the .npz file in a Jupyter Notebook")
    print(
        "and use the provided plotting functions (plot_correlation_vs_alpha, plot_distributions)."
    )
