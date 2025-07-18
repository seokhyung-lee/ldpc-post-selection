# CLAUDE.md

## NumPy Optimization Utilities (`simulations/analysis/data_collectors/numpy_utils/`)

This package provides high-performance numerical computing utilities optimized with NumPy for efficient data processing in quantum error correction simulations. It focuses on vectorized operations for statistical calculations and data manipulation.

## Package Structure

### Core Utilities

#### `cluster_distribution.py`
Cluster distribution analysis tools. Provides efficient NumPy-based functions for analyzing the statistical distribution of error clusters, including size distributions, spatial distributions, and clustering patterns in quantum error correction.

#### `cluster_metrics.py`
Cluster metric calculation utilities. Implements vectorized computations for various cluster-based metrics including cluster size norms, LLR (Log-Likelihood Ratio) distributions, cluster diameters, and other geometric properties essential for post-selection decoding research.

#### `histogram_utils.py`
Histogram computation utilities for efficient statistical analysis. Provides optimized functions for computing histograms, probability distributions, and cumulative distribution functions from large simulation datasets.

#### `sliding_window.py`
Sliding window analysis tools. Implements efficient numerical algorithms for analyzing decoder performance in sliding window scenarios, including temporal correlation analysis and streaming statistics computation.

### Package Configuration

#### `__init__.py`
Package initialization file that sets up the NumPy utilities and provides common imports for numerical computing modules.

## Key Features

### Vectorized Operations
- **Batch Processing**: Efficient processing of large arrays using NumPy vectorization
- **Memory Efficiency**: Optimized memory usage for large-scale data processing
- **Performance**: High-speed computations using optimized NumPy operations
- **Scalability**: Efficient handling of datasets from small tests to large-scale simulations

### Statistical Analysis
- **Distribution Analysis**: Fast computation of error and cluster distributions
- **Metric Calculation**: Vectorized computation of complex statistical metrics
- **Histogram Operations**: Efficient binning and distribution estimation
- **Correlation Analysis**: Temporal and spatial correlation computations

### Cluster Analysis
- **Size Distributions**: Efficient computation of cluster size statistics
- **Geometric Properties**: Vectorized calculation of cluster shapes and dimensions
- **Norm Calculations**: Fast computation of various cluster norms and fractions
- **Spatial Analysis**: Efficient analysis of cluster spatial relationships

## Usage Patterns

### Cluster Distribution Analysis
```python
from simulations.analysis.data_collectors.numpy_utils.cluster_distribution import compute_cluster_sizes

cluster_sizes = compute_cluster_sizes(error_patterns, adjacency_matrix)
```

### Metric Calculations
```python
from simulations.analysis.data_collectors.numpy_utils.cluster_metrics import compute_cluster_norms

size_norms, llr_norms = compute_cluster_norms(clusters, llr_values, alpha=2.0)
```

### Histogram Operations
```python
from simulations.analysis.data_collectors.numpy_utils.histogram_utils import compute_histogram

hist, bin_edges = compute_histogram(data, bins=50, density=True)
```

### Sliding Window Analysis
```python
from simulations.analysis.data_collectors.numpy_utils.sliding_window import analyze_window_performance

window_stats = analyze_window_performance(decoder_outputs, window_size=100)
```

## Performance Optimizations

### NumPy Vectorization
- **Array Operations**: Leverages NumPy's optimized C implementations
- **Broadcasting**: Efficient element-wise operations on arrays of different shapes
- **Indexing**: Advanced indexing for complex data selection and manipulation
- **Reduction Operations**: Fast aggregation and statistical computations

### Memory Management
- **In-Place Operations**: Minimizes memory allocation for large arrays
- **Data Type Optimization**: Efficient use of appropriate NumPy data types
- **Chunked Processing**: Handles datasets larger than available memory
- **Sparse Matrix Support**: Efficient handling of sparse quantum code matrices

### Algorithmic Efficiency
- **Vectorized Loops**: Eliminates explicit Python loops for better performance
- **Compiled Functions**: Uses NumPy's compiled mathematical operations
- **Parallel Operations**: Leverages multi-core capabilities where possible
- **Cache Optimization**: Memory access patterns optimized for CPU cache

## Integration Points

- **Data Collectors**: Provides numerical backend for data collection modules
- **Analysis Pipeline**: Supplies optimized computations for statistical analysis
- **Visualization**: Supports plotting utilities with efficient data preparation
- **Batch Processing**: Enables efficient large-scale data processing

This NumPy utilities package serves as the high-performance numerical computing foundation for the data analysis pipeline, enabling efficient processing of large-scale quantum error correction simulation results with optimal memory usage and computational speed.