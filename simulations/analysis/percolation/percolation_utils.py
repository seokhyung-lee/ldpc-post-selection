from typing import Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statsmodels.api as sm


def calculate_com_periodic(
    points: np.ndarray, dims: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculates the center of mass for points in a 2D periodic environment.

    This function implements the algorithm described by Bai and Breen, which
    projects the 2D points onto two orthogonal 3D tubes to correctly find
    the center of mass in a space with periodic boundary conditions.

    Parameters
    ----------
    points : np.ndarray
        An N x 2 array of (i, j) coordinates for N point masses.
    dims : tuple[float, float]
        A tuple (i_max, j_max) representing the dimensions of the 2D rectangle.

    Returns
    -------
    tuple[float, float]
        The (i, j) coordinates of the center of mass.
    """
    # Check if there are any points to avoid division by zero
    if points.shape[0] == 0:
        return (0.0, 0.0)

    i_max, j_max = dims
    i, j = points[:, 0], points[:, 1]

    # Calculate the i-component of the COM
    # Transform the 2D points to 3D on the surface of tube T_i
    r_i = i_max / (2 * np.pi)
    theta_i = (i / i_max) * 2 * np.pi

    x_coords_i = r_i * np.cos(theta_i)
    y_coords_i = j  # y = j for tube T_i
    z_coords_i = r_i * np.sin(theta_i)

    # Calculate the 3D COM of the points on tube T_i
    x_bar_i = np.mean(x_coords_i)
    z_bar_i = np.mean(z_coords_i)

    # Project the 3D COM back to get the i-coordinate
    # The paper notes a degenerate case for atan2(0,0), but np.arctan2 handles this by returning 0.
    # We add pi to map the output to [0, 2*pi] as per the paper.
    theta_i_bar = np.arctan2(-z_bar_i, -x_bar_i) + np.pi
    i_bar = (i_max / (2 * np.pi)) * theta_i_bar

    # Calculate the j-component of the COM
    # Transform the 2D points to 3D on the surface of tube T_j
    r_j = j_max / (2 * np.pi)
    theta_j = (j / j_max) * 2 * np.pi

    x_coords_j = i  # x = i for tube T_j
    y_coords_j = r_j * np.cos(theta_j)
    z_coords_j = r_j * np.sin(theta_j)

    # Calculate the 3D COM of the points on tube T_j
    y_bar_j = np.mean(y_coords_j)
    z_bar_j = np.mean(z_coords_j)

    # Project the 3D COM back to get the j-coordinate
    theta_j_bar = np.arctan2(-z_bar_j, -y_bar_j) + np.pi
    j_bar = (j_max / (2 * np.pi)) * theta_j_bar

    return (i_bar, j_bar)


def calculate_distance_periodic(
    p1: np.ndarray,
    p2: np.ndarray,
    dims: tuple[float, float],
    use_manhattan: bool = False,
) -> np.ndarray:
    """
    Calculates distance between broadcastable arrays of points in a 2D periodic environment.

    The distance is calculated for each corresponding pair of points after
    broadcasting the input arrays p1 and p2. The calculation considers the
    shortest path along each axis, which may wrap around the boundaries.

    Parameters
    ----------
    p1 : np.ndarray
        An array of (i, j) coordinates. The shape must be (..., 2).
    p2 : np.ndarray
        An array of (i, j) coordinates, broadcastable to p1. The shape must
        also be (..., 2).
    dims : tuple[float, float]
        The (i_max, j_max) dimensions of the periodic space.
    use_manhattan : bool, optional
        If True, calculates the Manhattan distance. Otherwise, calculates
        the Euclidean distance (default is False).

    Returns
    -------
    np.ndarray
        An array containing the calculated distances. Its shape is determined
        by broadcasting the shapes of p1 and p2.
    """
    i_max, j_max = dims

    # Separate the i and j coordinates for each array of points
    i1, j1 = p1[..., 0], p1[..., 1]
    i2, j2 = p2[..., 0], p2[..., 1]

    # Calculate the shortest distance along the i-axis for all pairs
    delta_i = np.abs(i1 - i2)
    dist_i = np.minimum(delta_i, i_max - delta_i)

    # Calculate the shortest distance along the j-axis for all pairs
    delta_j = np.abs(j1 - j2)
    dist_j = np.minimum(delta_j, j_max - delta_j)

    if use_manhattan:
        # Manhattan distance is the sum of the axial distances
        return dist_i + dist_j
    else:
        # Euclidean distance is the root of the sum of squares
        return np.sqrt(dist_i**2 + dist_j**2)


def _calculate_ssr(
    params: list[float], eta: np.ndarray, L: np.ndarray, Q: np.ndarray, frac: float
) -> float:
    """
    Calculate the Sum of Squared Residuals (SSR) from a smoothed curve.

    This serves as the objective function for the optimizer. It calculates
    the SSR of all data points from a LOWESS-smoothed master curve.

    Parameters
    ----------
    params : list[float]
        A list or tuple containing the parameters to optimize: [eta_c, nu].
    eta : np.ndarray
        Array of the eta parameter.
    L : np.ndarray
        Array of the L parameter.
    Q : np.ndarray
        Array of the physical quantity Q.
    frac : float
        The fraction of data used for LOWESS smoothing at each point.

    Returns
    -------
    float
        The calculated SSR value to be minimized.
    """
    eta_c, nu = params

    if nu == 0:
        return np.inf

    # Calculate the scaled x-axis
    x = (eta - eta_c) * (L ** (1 / nu))

    # LOWESS requires the data to be sorted by the x-values.
    # We must sort Q according to the sorting of x.
    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]
    Q_sorted = Q[sort_indices]

    # Perform LOWESS smoothing to get the estimated master curve.
    # `return_sorted=False` ensures the output corresponds to our `x_sorted`.
    Q_smooth = sm.nonparametric.lowess(
        Q_sorted, x_sorted, frac=frac, return_sorted=False
    )

    # Calculate the Sum of Squared Residuals (SSR)
    ssr = np.sum((Q_sorted - Q_smooth) ** 2)

    return ssr


def find_optimal_collapse_ssr(
    eta: np.ndarray,
    L: np.ndarray,
    Q: np.ndarray,
    initial_guess: list[float],
    bounds: tuple[tuple[float, float], tuple[float, float]] = None,
    method: str = "Nelder-Mead",
    lowess_frac: float = 0.25,
) -> tuple[float, float, float, object]:
    """
    Find optimal (eta_c, nu) by minimizing the SSR from a smoothed curve.

    Parameters
    ----------
    eta : np.ndarray
        Array of the eta parameter.
    L : np.ndarray
        Array of the L parameter.
    Q : np.ndarray
        Array of the physical quantity Q.
    initial_guess : list[float]
        Initial guess for [eta_c, nu].
    bounds : tuple[tuple[float, float], tuple[float, float]], optional
        Bounds for [eta_c, nu].
    method : str, optional
        Method for the optimizer.
    lowess_frac : float, optional
        Fraction of data used for LOWESS smoothing (default is 0.25).
        Controls the smoothness of the master curve.

    Returns
    -------
    eta_c_opt : float
        Optimized critical point eta_c.
    nu_opt : float
        Optimized critical exponent nu.
    min_ssr : float
        The minimum SSR value found by the optimizer.
    result : object
        The full optimization result object from scipy.optimize.minimize.
    """
    # Set up arguments for the optimizer
    args = (eta, L, Q, lowess_frac)

    # Run the minimization using Nelder-Mead
    result = minimize(
        _calculate_ssr,
        x0=initial_guess,
        args=args,
        bounds=bounds,
        method=method,
    )

    eta_c_opt, nu_opt = result.x
    min_ssr = result.fun

    return eta_c_opt, nu_opt, min_ssr, result
