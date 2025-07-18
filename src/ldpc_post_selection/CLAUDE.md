# CLAUDE.md

## Core Decoder Package (`src/ldpc_post_selection/`)

This package contains the core quantum error correction decoder implementations for post-selection decoding research. It provides the fundamental building blocks for decoding quantum LDPC codes with soft output generation capabilities.

## Package Structure

### Core Modules

#### `base.py`
Foundation classes and common functionality for all decoders. Defines the base interfaces and shared utilities that other decoder implementations inherit from.

#### `decoder.py`
Primary decoder interface implementations. Contains the main `SoftOutputsDecoder` base class and concrete decoder implementations that provide both hard decisions and soft statistical outputs for post-selection analysis.

#### `bplsd_decoder.py`
Belief Propagation + List Sequential Decoding (BP+LSD) implementation. This module contains the `SoftOutputsBpLsdDecoder` class that combines belief propagation with list sequential decoding to provide enhanced error correction capabilities with detailed statistical outputs.

#### `matching_decoder.py`
Minimum-Weight Perfect Matching (MWPM) decoder implementation. Contains the `SoftOutputsMatchingDecoder` class that uses graph-based matching algorithms for quantum error correction, particularly effective for topological codes like surface codes.

### Integration and Analysis Tools

#### `stim_tools.py`
Integration utilities for the Stim quantum circuit simulator. Provides functions for converting between Stim's detector error models and parity check matrices, enabling seamless integration with Stim-generated quantum circuits and error models.

#### `cluster_tools.py`
Error cluster analysis and investigation tools. Contains functions for analyzing error patterns, computing cluster metrics, and understanding the geometric structure of errors in quantum codes. Essential for post-selection decoding research.

#### `tools.py`
General utility functions and helper tools. Provides common mathematical operations, data processing utilities, and convenience functions used across multiple decoder implementations.

### Package Configuration

#### `__init__.py`
Package initialization file that defines the public API and imports the main decoder classes for easy access when importing the package.

## Key Features

### Decoder Architecture
- **Unified Interface**: All decoders implement the same base interface with `decode()` method
- **Soft Outputs**: Decoders provide both hard decisions and soft statistical information
- **Extensible Design**: Easy to add new decoder types by inheriting from base classes

### Integration Capabilities
- **Stim Integration**: Seamless work with Stim quantum circuit simulator
- **Cluster Analysis**: Built-in tools for analyzing error patterns and clusters
- **Statistical Outputs**: Rich metadata for post-selection analysis

### Performance Considerations
- **Optimized Algorithms**: Efficient implementations of BP+LSD and matching algorithms
- **Sparse Matrix Support**: Handles large quantum codes efficiently
- **Parallelization Ready**: Designed for use in parallel simulation frameworks

## Usage Patterns

### Basic Decoder Usage
```python
from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder

decoder = SoftOutputsBpLsdDecoder(H, max_iter=100)
prediction, soft_outputs = decoder.decode(syndrome)
```

### Stim Integration
```python
from ldpc_post_selection.stim_tools import dem_to_check_matrices

H_X, H_Z = dem_to_check_matrices(detector_error_model)
```

### Cluster Analysis
```python
from ldpc_post_selection.cluster_tools import analyze_error_clusters

cluster_info = analyze_error_clusters(error_pattern, adjacency_matrix)
```

This package serves as the foundation for all quantum error correction research in the project, providing reliable, well-tested decoder implementations with the statistical outputs necessary for post-selection decoding analysis.