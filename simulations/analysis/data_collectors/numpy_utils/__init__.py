# Re-export all functions from the refactored modules to maintain backward compatibility

# Cluster metrics (CSR and flattened data)
from .cluster_metrics import (
    calculate_cluster_metrics_from_csr,
    calculate_cluster_norms_from_flat_data,
    calculate_cluster_inv_entropies_from_csr,
    calculate_cluster_inv_priors_from_csr,
)

# Cluster size distribution
from .cluster_distribution import (
    get_cluster_size_distribution_from_csr,
)

# Sliding window operations
from .sliding_window import (
    calculate_window_cluster_norm_fracs_from_csr,
    calculate_committed_cluster_norm_fractions_from_csr,
)

# Export all public names for * imports
__all__ = [
    # Cluster metrics
    "calculate_cluster_metrics_from_csr",
    "calculate_cluster_norms_from_flat_data",
    "calculate_cluster_inv_entropies_from_csr",
    "calculate_cluster_inv_priors_from_csr",
    # Cluster distribution
    "get_cluster_size_distribution_from_csr",
    # Sliding window operations
    "calculate_window_cluster_norm_fracs_from_csr",
    "calculate_committed_cluster_norm_fractions_from_csr",
]
