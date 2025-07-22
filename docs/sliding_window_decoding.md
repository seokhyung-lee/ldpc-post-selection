# Sliding Window Decoding for Quantum LDPC Codes

## Overview

Sliding window decoding is a memory-efficient approach for decoding quantum LDPC codes in time-evolving quantum error correction protocols. This technique enables real-time decoding of large-scale quantum circuits by processing error syndromes in overlapping temporal windows rather than requiring the complete syndrome history.

## Architecture and Implementation

### Core Decoder: `SoftOutputsBpLsdDecoder`

The sliding window decoding functionality is implemented in the `SoftOutputsBpLsdDecoder` class located in `src/ldpc_post_selection/bplsd_decoder.py`. This decoder extends the base `SoftOutputsDecoder` with specialized sliding window capabilities.

#### Key Features:

- **Window-based Processing**: Processes detector outcomes in overlapping time windows
- **Caching System**: Optimizes performance through intelligent caching of window structures and decoders
- **Cluster Analysis**: Provides detailed cluster statistics across windows
- **Parallel Processing**: Supports parallel decoding across multiple cores

#### Time Coordinate System

The decoder uses `detector_time_coords` to organize detectors by time, enabling proper window partitioning:

```python
def detector_time_coords(self) -> np.ndarray:
    """Returns time coordinates for each detector, extracted from circuit or explicitly provided"""
```

### Sliding Window Algorithm

The core sliding window algorithm is implemented in the `decode_sliding_window` method:

#### Parameters:
- `window_size` (W): Number of rounds in each decoding window
- `commit_size` (F): Number of rounds committed from each window (W > F)
- Constraint: Window size must exceed commit size to ensure overlap

#### Algorithm Steps:

1. **Window Extraction**: For window `w`, extract detectors in time range `[w*F, w*F + W - 1]`
2. **Fault Filtering**: Identify active faults excluding previously committed faults
3. **Submatrix Construction**: Create window-specific parity check matrix and priors
4. **Decoding**: Apply BP+LSD decoding to the window
5. **Commitment Strategy**:
   - Regular windows: Commit faults in range `[w*F, w*F + F - 1]`
   - Final window: Commit all remaining faults
6. **Syndrome Update**: Update detector outcomes by removing effects of committed faults
7. **Iteration**: Continue until all time rounds are processed

### Caching and Performance Optimization

The implementation includes sophisticated caching mechanisms for optimal performance:

#### Window Structure Cache
```python
self._window_structure_cache: Dict[Tuple[int, int, int], Dict[str, Any]]
```
Caches window detector masks and H-matrix rows for each `(window_size, commit_size, window_position)` configuration.

#### Decoder Cache
```python
self._decoder_cache: Dict[str, SoftOutputsBpLsdDecoder]
```
Caches decoder instances based on H-matrix and priors hash keys, avoiding repeated decoder instantiation.

#### Cache Management
- `clear_caches()`: Clears all cached data
- `get_cache_info()`: Returns cache utilization statistics

## Analysis Tools and Data Processing

### Cluster Analysis (`simulations/analysis/data_collectors/numpy_utils/sliding_window.py`)

The analysis module provides optimized tools for processing sliding window simulation results:

#### Window Cluster Norm Fractions

Computes cluster norm fractions across sliding windows using high-performance NumPy and Numba implementations:

```python
def calculate_window_cluster_norm_fractions_from_csr(
    all_clusters_csr: csr_matrix,
    priors: np.ndarray,
    norm_order: float,
    value_type: str,  # "size" or "llr"
    aggregation_type: str,  # "avg" or "max"
    eval_windows: Tuple[int, int] | None = None,
) -> np.ndarray
```

**Key Features:**
- **CSR Matrix Input**: Efficiently processes sparse cluster data
- **Multiple Norm Orders**: Supports L_p norms including L₁, L₂, L_∞
- **Value Types**: Computes norms for cluster sizes or LLR sums
- **Window Aggregation**: Averages or maximizes across windows
- **Numba Optimization**: Uses compiled kernels for maximum performance

#### Committed Cluster Analysis

Analyzes clusters formed by committed faults across multiple windows:

```python
def calculate_committed_cluster_norm_fractions_from_csr(
    committed_clusters_csr: csr_matrix,
    committed_faults: List[np.ndarray],
    priors: np.ndarray,
    adj_matrix: np.ndarray,  # this is (H.T @ H) == 1
    norm_order: float,
    value_type: str,
    eval_windows: Tuple[int, int] | None = None,
    _benchmarking: bool = False,
    num_jobs: int = 1,
    num_batches: int | None = None,
) -> np.ndarray
```

**Algorithm:**
1. **Window Splitting**: Splits CSR matrix by window boundaries
2. **Logical OR**: Combines committed clusters across evaluation windows
3. **Graph Construction**: Creates igraph representation for connected component analysis
4. **Parallel Processing**: Uses joblib for parallel sample processing
5. **Norm Calculation**: Computes L_p norms using optimized Numba kernels

### Performance Benchmarking

Both analysis functions include detailed benchmarking capabilities:

```python
_benchmarking: bool = False  # Enable for performance analysis
```

When enabled, provides detailed timing information for:
- Window extraction and matrix operations
- Graph construction and clustering
- Parallel processing overhead
- Memory allocation patterns

## Data Storage and File Organization

The sliding window decoding simulation system uses a three-tier data storage architecture designed for efficiency and scalability: raw simulation data, aggregated statistics, and post-selection results.

### Raw Simulation Data Storage

Raw sliding window simulation data is stored in a hierarchical directory structure under `simulations/data/`:

#### Directory Structure
```
simulations/data/
├── {dataset_name}_sliding_window_{decoder_config}_raw/
│   └── {parameter_combination}/
│       ├── H.npz                    # Parity check matrix (sparse)
│       ├── priors.npy               # Error probabilities
│       ├── committed_faults.npz     # List of committed fault arrays per window
│       └── batch_{i}_{batch_size}/  # Simulation batches
│           ├── fails.npy            # Boolean array: decoding failures
│           ├── all_clusters.npz     # CSR matrix: cluster assignments across all windows
│           └── committed_clusters.npz # CSR matrix: committed cluster assignments
```

#### File Format Details

**Sliding Window Format Files:**
- `fails.npy`: 1D boolean array indicating decoding success/failure for each sample
- `all_clusters.npz`: Sparse integer CSR matrix
- `committed_clusters.npz`: Sparse boolean CSR matrix
- `committed_faults.npz`: List of boolean arrays, one per window, indicating committed faults

**Example Directory Names:**
- `bb_sliding_window_minsum_iter30_lsd0_raw/`: BB codes with sliding window decoding
- `n144_T12_p0.003_W3_F1/`: n=144 qubits, T=12 rounds, p=0.003 error rate, W=3 window size, F=1 commit size

#### CSR Matrix Data Layout

The sliding window system uses Compressed Sparse Row (CSR) matrices for memory-efficient storage:

```python
# all_clusters.npz structure
all_clusters_csr: csr_matrix  # Shape: (num_samples, num_windows * num_faults)
# Column index mapping: window_idx * num_faults + fault_idx → global fault position
# Matrix values: cluster_id (0 = no cluster, >0 = cluster ID)

# committed_clusters.npz structure  
committed_clusters_csr: csr_matrix  # Shape: (num_samples, num_windows * num_faults)
# Boolean values: 1 if fault is committed AND in a cluster, 0 otherwise
```

### Data Format Detection

The system automatically detects data formats in `data_metric_calculation.py`:

```python
def _detect_data_format(batch_dir_path: str, by: str, verbose: bool):
    """
    Detects sliding window format by checking for:
    - fails.npy (instead of scalars.feather)
    - all_clusters.npz and committed_clusters.npz (CSR format)
    """
```

**Format Detection Logic:**
- **Sliding Window Format**: `fails.npy` + `all_clusters.npz` + `committed_clusters.npz`

### BB Sliding Window Data Collection

The `collect_bb_sliding_window_simulation_data.py` script demonstrates practical usage:

```python
# Define sliding window specific metrics
aggregation_types = ["avg_window", "max_window", "committed"]  
value_types = ["size", "llr"]
orders = [2]  # L2 norm

# Data directory configuration
data_dir_name = "bb_sliding_window_minsum_iter30_lsd0_raw"
subdirs = ["n144_T12_p0.003_W3_F1", "n144_T12_p0.005_W3_F1"]

# Process datasets
process_dataset(
    data_dir=data_dir,
    dataset_name="bb_sliding_window", 
    ascending_confidences=ascending_confidences,
    orders=orders
)
```

**Metric Types:**
- `avg_window`: Average cluster norm fractions across windows
- `max_window`: Maximum cluster norm fractions across windows  
- `committed`: Cluster norm fractions for committed faults only

### Integration with Simulation Framework

The sliding window decoder integrates seamlessly with the broader simulation framework:

```python
def simulate_single(self, sliding_window=False, seed=None, **kwargs):
    """Single simulation supporting both regular and sliding window modes"""
    if sliding_window:
        pred, soft_outputs = self.decode_sliding_window(det_outcomes, **kwargs)
    else:
        pred, _, _, soft_outputs = self.decode(det_outcomes, **kwargs)
```

### Memory and Performance Considerations

The storage system is optimized for large-scale simulations:

**Storage Efficiency:**
- CSR sparse matrices reduce memory usage for sparse cluster data
- Batch organization enables incremental processing
- Intelligent caching prevents redundant computations

**Processing Efficiency:**  
- Parallel batch processing using joblib
- Numba-optimized kernels for cluster calculations
- Streaming data loading to handle large datasets

**Scalability Features:**
- Configurable batch sizes for memory management
- Support for distributed computation across multiple cores
- Incremental data processing with resume capability

## Technical Considerations

### Memory Efficiency

Sliding window decoding provides significant memory advantages:
- **Bounded Memory**: Memory usage independent of total circuit depth
- **Local Processing**: Only requires syndrome data within window boundaries
- **Cache Optimization**: Reuses decoder instances across similar windows

### Accuracy Trade-offs

The sliding window approach involves accuracy trade-offs:
- **Window Size**: Larger windows improve accuracy but increase memory usage
- **Commit Strategy**: Earlier commitment reduces memory but may increase errors
- **Overlap**: Window overlap (W > F) maintains decoding context

### Parallel Scalability

The implementation supports various parallelization strategies:
- **Sample-level**: Parallel processing of multiple error instances
- **Window-level**: Independent processing of non-overlapping windows
- **Batch Processing**: Efficient processing of large simulation datasets

## Usage Examples

### Basic Sliding Window Decoding

```python
decoder = SoftOutputsBpLsdDecoder(H=parity_check_matrix, p=priors)

# Decode with sliding window
prediction, soft_outputs = decoder.decode_sliding_window(
    detector_outcomes=syndromes,
    window_size=5,     # W=5 rounds per window
    commit_size=2,     # F=2 rounds committed per window
    verbose=True
)

# Access sliding window specific outputs
all_clusters = soft_outputs["all_clusters"]
committed_clusters = soft_outputs["committed_clusters"]
committed_faults = soft_outputs["committed_faults"]
```

### Analysis of Sliding Window Results

```python
from simulations.analysis.data_collectors.numpy_utils.sliding_window import (
    calculate_window_cluster_norm_fractions_from_csr
)

# Calculate cluster norm fractions across windows
norm_fractions = calculate_window_cluster_norm_fractions_from_csr(
    all_clusters_csr=cluster_matrix,
    priors=error_priors,
    norm_order=2.0,          # L2 norm
    value_type="llr",        # LLR-based clusters
    aggregation_type="avg",  # Average across windows
    eval_windows=(5, 15)     # Evaluate windows 5-15 only
)
```

### Performance Optimization

```python
# Enable caching for repeated simulations
decoder = SoftOutputsBpLsdDecoder(H=H, p=p)

# Check cache utilization
cache_info = decoder.get_cache_info()
print(f"Cached window structures: {cache_info['window_structures']}")
print(f"Cached decoders: {cache_info['decoders']}")

# Clear caches when changing decoder parameters
decoder.clear_caches()
```

## Conclusion

The sliding window decoding implementation provides a comprehensive solution for memory-efficient quantum error correction. The combination of optimized caching, parallel processing, and detailed cluster analysis makes it suitable for large-scale quantum computing simulations. The modular design allows for easy integration with various quantum error correction codes and simulation frameworks.