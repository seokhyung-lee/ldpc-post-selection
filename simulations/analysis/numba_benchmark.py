import unittest
import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random
import numba

# ===================================================================
# 이전에 작성한 세 개의 함수를 여기에 붙여넣습니다.
# (_calculate_cluster_norms_from_csr_dense_inf,
#  _calculate_cluster_norms_from_csr_sparse_inf,
#  _calculate_cluster_norms_from_csr_numba_optimized 및 그 커널 함수)
# 편의를 위해 여기에 다시 포함합니다.
# ===================================================================


# --- Dense Version ---
def _calculate_cluster_norms_from_csr_dense_inf(
    clusters: csr_matrix, bit_llrs: np.ndarray, norm_order: float, calculate_llrs: bool
) -> tuple[np.ndarray, np.ndarray]:
    num_samples, num_bits = clusters.shape
    dense_clusters = clusters.toarray()
    weights = bit_llrs if calculate_llrs else np.ones(num_bits, dtype=np.float64)
    outside_mask = dense_clusters == 0
    outside_values = np.sum(outside_mask * weights[np.newaxis, :], axis=1)
    max_cluster_id = dense_clusters.max()
    if max_cluster_id == 0:
        return np.zeros(num_samples), outside_values
    instance_ids = (
        np.arange(num_samples)[:, np.newaxis] * (max_cluster_id + 1) + dense_clusters
    )
    tiled_weights = np.tile(weights, (num_samples, 1))
    inside_mask = dense_clusters > 0
    cluster_sums = np.bincount(
        instance_ids[inside_mask], weights=tiled_weights[inside_mask]
    )
    reshaped_sums = np.resize(cluster_sums, num_samples * (max_cluster_id + 1))
    reshaped_sums = reshaped_sums.reshape(num_samples, max_cluster_id + 1)
    if norm_order == np.inf:
        inside_norms = np.max(reshaped_sums, axis=1)
    else:
        powered_sums = np.power(reshaped_sums, norm_order)
        sum_of_powered_sums = np.sum(powered_sums, axis=1)
        inside_norms = np.power(sum_of_powered_sums, 1.0 / norm_order)
    return inside_norms, outside_values


# --- Sparse Version ---
def _calculate_cluster_norms_from_csr_sparse_inf(
    clusters: csr_matrix, bit_llrs: np.ndarray, norm_order: float, calculate_llrs: bool
) -> tuple[np.ndarray, np.ndarray]:
    num_samples, num_bits = clusters.shape
    if clusters.nnz == 0:
        total_outside = np.sum(bit_llrs) if calculate_llrs else float(num_bits)
        return np.zeros(num_samples), np.full(num_samples, total_outside)
    max_cluster_id = int(clusters.data.max())
    cluster_weights = (
        bit_llrs[clusters.indices]
        if calculate_llrs
        else np.ones_like(clusters.data, dtype=np.float64)
    )
    row_indices = np.arange(num_samples).repeat(np.diff(clusters.indptr))
    col_indices = clusters.data - 1
    cluster_sum_matrix = csr_matrix(
        (cluster_weights, (row_indices, col_indices)),
        shape=(num_samples, max_cluster_id),
    )
    if norm_order == np.inf:
        inside_norms = cluster_sum_matrix.max(axis=1).toarray().flatten()
    else:
        cluster_sum_matrix.data **= norm_order
        sum_of_powers = np.array(cluster_sum_matrix.sum(axis=1)).flatten()
        inside_norms = np.power(sum_of_powers, 1.0 / norm_order)
    total_weights = np.sum(bit_llrs if calculate_llrs else np.ones(num_bits))
    inside_weights_sum_per_sample = csr_matrix(
        (cluster_weights, clusters.indices, clusters.indptr),
        shape=(num_samples, num_bits),
    ).sum(axis=1)
    outside_values = total_weights - np.array(inside_weights_sum_per_sample).flatten()
    return inside_norms, outside_values


# --- Numba Version ---
@numba.njit(fastmath=True, cache=True)
def _numba_kernel_optimized(
    data,
    indices,
    indptr,
    num_samples,
    max_cluster_id,
    bit_llrs,
    norm_order,
    calculate_llrs,
    total_weights,
):
    inside_norms = np.zeros(num_samples, dtype=np.float64)
    outside_values = np.zeros(num_samples, dtype=np.float64)
    temp_cluster_sums = np.zeros(max_cluster_id + 1, dtype=np.float64)
    for i in range(num_samples):
        temp_cluster_sums.fill(0)
        start, end = indptr[i], indptr[i + 1]
        total_inside_weight_for_sample = 0.0
        for j in range(start, end):
            cluster_id = data[j]
            bit_index = indices[j]
            weight = bit_llrs[bit_index] if calculate_llrs else 1.0
            temp_cluster_sums[cluster_id] += weight
            total_inside_weight_for_sample += weight
        active_cluster_sums = temp_cluster_sums[1:]
        norm_val = 0.0
        if norm_order == 1.0:
            norm_val = np.sum(active_cluster_sums)
        elif norm_order == 2.0:
            norm_val = np.sqrt(np.sum(active_cluster_sums**2))
        elif norm_order == np.inf:
            if active_cluster_sums.size > 0:
                norm_val = np.max(active_cluster_sums)
        elif norm_order == 0.5:
            norm_val = np.sum(np.sqrt(active_cluster_sums)) ** 2
        else:
            norm_val = np.sum(active_cluster_sums**norm_order) ** (1.0 / norm_order)
        inside_norms[i] = norm_val
        outside_values[i] = total_weights - total_inside_weight_for_sample
    return inside_norms, outside_values


def _calculate_cluster_norms_from_csr_numba_optimized(
    clusters, bit_llrs, norm_order, calculate_llrs
):
    num_samples, num_bits = clusters.shape
    if clusters.nnz == 0:
        total_outside = np.sum(bit_llrs) if calculate_llrs else float(num_bits)
        return np.zeros(num_samples), np.full(num_samples, total_outside)
    total_weights = np.sum(bit_llrs) if calculate_llrs else float(num_bits)
    max_cluster_id = int(clusters.data.max())
    return _numba_kernel_optimized(
        clusters.data,
        clusters.indices,
        clusters.indptr,
        num_samples,
        max_cluster_id,
        bit_llrs,
        norm_order,
        calculate_llrs,
        total_weights,
    )


# ===================================================================
# Test Code Starts Here
# ===================================================================


class TestClusterNormFunctions(unittest.TestCase):

    def _generate_test_data(self, num_samples, num_bits, max_cluster_id, density):
        """Generates random test data."""
        # Create a sparse matrix with random integers from 0 to max_cluster_id
        S = sparse_random(num_samples, num_bits, density=density, format="csr")
        S.data = np.random.randint(1, max_cluster_id + 1, size=S.nnz)

        # bit_llrs are positive to avoid issues with fractional norms
        bit_llrs = np.random.rand(num_bits) + 0.1
        return S, bit_llrs

    def _run_and_assert_all(
        self, case_name, clusters, bit_llrs, norm_order, calculate_llrs
    ):
        """Helper function to run all three implementations and compare results."""
        with self.subTest(msg=f"Case: {case_name}"):
            # 1. Run Dense implementation
            dense_norms, dense_outside = _calculate_cluster_norms_from_csr_dense_inf(
                clusters, bit_llrs, norm_order, calculate_llrs
            )

            # 2. Run Sparse implementation
            sparse_norms, sparse_outside = _calculate_cluster_norms_from_csr_sparse_inf(
                clusters, bit_llrs, norm_order, calculate_llrs
            )

            # 3. Run Numba implementation
            numba_norms, numba_outside = (
                _calculate_cluster_norms_from_csr_numba_optimized(
                    clusters, bit_llrs, norm_order, calculate_llrs
                )
            )

            # --- Assertions ---
            # Compare inside_norms
            np.testing.assert_allclose(
                dense_norms,
                sparse_norms,
                rtol=1e-6,
                err_msg=f"{case_name}: Dense vs Sparse norms mismatch",
            )
            np.testing.assert_allclose(
                dense_norms,
                numba_norms,
                rtol=1e-6,
                err_msg=f"{case_name}: Dense vs Numba norms mismatch",
            )

            # Compare outside_values
            np.testing.assert_allclose(
                dense_outside,
                sparse_outside,
                rtol=1e-6,
                err_msg=f"{case_name}: Dense vs Sparse outside mismatch",
            )
            np.testing.assert_allclose(
                dense_outside,
                numba_outside,
                rtol=1e-6,
                err_msg=f"{case_name}: Dense vs Numba outside mismatch",
            )

    def test_all_scenarios(self):
        """Run tests for a variety of configurations."""
        num_samples, num_bits, max_cluster_id = 50, 100, 7
        clusters, bit_llrs = self._generate_test_data(
            num_samples, num_bits, max_cluster_id, density=0.3
        )

        norm_orders_to_test = [1.0, 2.0, 0.5, np.inf, 3.0]

        for p in norm_orders_to_test:
            # Test with calculate_llrs = True
            self._run_and_assert_all(
                case_name=f"p={p}, use_llrs=True",
                clusters=clusters,
                bit_llrs=bit_llrs,
                norm_order=p,
                calculate_llrs=True,
            )
            # Test with calculate_llrs = False
            self._run_and_assert_all(
                case_name=f"p={p}, use_llrs=False",
                clusters=clusters,
                bit_llrs=bit_llrs,
                norm_order=p,
                calculate_llrs=False,
            )

    def test_edge_case_no_clusters(self):
        """Test with a matrix containing no clusters (all zeros)."""
        num_samples, num_bits = 30, 50
        clusters = csr_matrix((num_samples, num_bits), dtype=int)
        bit_llrs = np.random.rand(num_bits)

        self._run_and_assert_all(
            case_name="Edge Case: No Clusters, p=2, use_llrs=True",
            clusters=clusters,
            bit_llrs=bit_llrs,
            norm_order=2.0,
            calculate_llrs=True,
        )
        self._run_and_assert_all(
            case_name="Edge Case: No Clusters, p=inf, use_llrs=False",
            clusters=clusters,
            bit_llrs=bit_llrs,
            norm_order=np.inf,
            calculate_llrs=False,
        )

    def test_edge_case_full_clusters(self):
        """Test with a matrix where all bits belong to a cluster."""
        num_samples, num_bits, max_cluster_id = 40, 60, 5
        # Density=1.0 ensures no zeros in the sparse representation
        clusters, bit_llrs = self._generate_test_data(
            num_samples, num_bits, max_cluster_id, density=1.0
        )

        self._run_and_assert_all(
            case_name="Edge Case: Full Clusters, p=1, use_llrs=True",
            clusters=clusters,
            bit_llrs=bit_llrs,
            norm_order=1.0,
            calculate_llrs=True,
        )
        self._run_and_assert_all(
            case_name="Edge Case: Full Clusters, p=3, use_llrs=False",
            clusters=clusters,
            bit_llrs=bit_llrs,
            norm_order=3.0,
            calculate_llrs=False,
        )


if __name__ == "__main__":
    # This allows running the test script directly.
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestClusterNormFunctions))
    runner = unittest.TextTestRunner()
    runner.run(suite)
