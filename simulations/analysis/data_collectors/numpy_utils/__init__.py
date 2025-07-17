# Re-export all functions from the refactored modules to maintain backward compatibility

# Histogram utilities
from .histogram_utils import (
    _is_uniform_binning,
    _calculate_histograms_bplsd_numba,
    _calculate_histograms_matching_numba,
)

# Cluster metrics (CSR and flattened data)
from .cluster_metrics import (
    calculate_cluster_metrics_from_csr,
    _calculate_cluster_norms_from_flat_data_numba,
    _calculate_cluster_norms_numba_kernel,
    _calculate_cluster_norms_filtered_numba_kernel,
    _numba_cluster_norm_kernel,
    _calculate_cluster_norms_from_csr_numba,
    _numba_inv_entropy_kernel,
    _calculate_cluster_inv_entropies_from_csr,
    _numba_inv_priors_kernel,
    _calculate_cluster_inv_priors_from_csr,
)

# Cluster size distribution
from .cluster_distribution import (
    get_cluster_size_distribution_from_csr,
    _get_cluster_size_distribution_numba_kernel,
)

# Sliding window operations
from .sliding_window import (
    _calculate_sliding_window_norm_fractions,
    _calculate_sliding_window_norm_fractions_numba,
    _numba_window_norm_fracs_kernel,
    calculate_window_cluster_norm_fracs_from_csr,
    _split_csr_by_windows,
    _compute_logical_or_across_windows,
    calculate_committed_cluster_norm_fractions_from_csr,
)

# Export all public names for * imports
__all__ = [
    # Histogram utilities
    "_is_uniform_binning",
    "_calculate_histograms_bplsd_numba",
    "_calculate_histograms_matching_numba",
    # Cluster metrics
    "calculate_cluster_metrics_from_csr",
    "_calculate_cluster_norms_from_flat_data_numba",
    "_calculate_cluster_norms_numba_kernel",
    "_calculate_cluster_norms_filtered_numba_kernel",
    "_numba_cluster_norm_kernel",
    "_calculate_cluster_norms_from_csr_numba",
    "_numba_inv_entropy_kernel",
    "_calculate_cluster_inv_entropies_from_csr",
    "_numba_inv_priors_kernel",
    "_calculate_cluster_inv_priors_from_csr",
    # Cluster distribution
    "get_cluster_size_distribution_from_csr",
    "_get_cluster_size_distribution_numba_kernel",
    # Sliding window operations
    "_calculate_sliding_window_norm_fractions",
    "_calculate_sliding_window_norm_fractions_numba",
    "_numba_window_norm_fracs_kernel",
    "calculate_window_cluster_norm_fracs_from_csr",
    "_split_csr_by_windows",
    "_compute_logical_or_across_windows",
    "calculate_committed_cluster_norm_fractions_from_csr",
]