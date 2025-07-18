# CLAUDE.md

## Percolation Analysis Framework (`simulations/analysis/percolation/`)

This package provides specialized analysis tools for applying percolation theory to quantum error correction research. It focuses on cluster analysis, percolation threshold calculations, and geometric investigations of error patterns in quantum codes.

## Package Structure

### Core Analysis Tools

#### `percolation_utils.py`
Fundamental percolation analysis utilities. Provides core functions for percolation theory applications in quantum error correction, including cluster identification, connectivity analysis, and percolation threshold estimation.

#### `cluster_diameters.py`
Cluster diameter calculation tools. Implements algorithms for computing geometric properties of error clusters, including cluster diameters, aspect ratios, and other shape characteristics essential for understanding error propagation in quantum codes.

#### `toric_code_percolation.py`
Specialized percolation analysis for toric codes. Provides tools for analyzing percolation phenomena in toric code lattices, including threshold calculations and cluster behavior analysis specific to toric code geometry.

#### `collect_toric_code_stats.py`
Toric code statistics collection framework. Orchestrates systematic data collection for percolation analysis in toric codes, including cluster statistics, threshold measurements, and geometric property analysis.

### Package Configuration

#### `__init__.py`
Package initialization file that sets up the percolation analysis framework and provides common imports for percolation analysis modules.

## Key Features

### Percolation Theory Applications
- **Cluster Identification**: Efficient algorithms for identifying connected error clusters
- **Threshold Calculations**: Percolation threshold estimation for different code families
- **Connectivity Analysis**: Investigation of error cluster connectivity patterns
- **Scaling Analysis**: Study of percolation behavior across different code sizes

### Geometric Analysis
- **Cluster Shapes**: Detailed analysis of error cluster geometric properties
- **Diameter Calculations**: Efficient computation of cluster diameters and dimensions
- **Aspect Ratios**: Analysis of cluster elongation and directional properties
- **Spatial Correlations**: Investigation of spatial relationships between error clusters

### Code-Specific Analysis
- **Toric Code Focus**: Specialized tools for toric code percolation analysis
- **Lattice Geometry**: Analysis adapted to different quantum code lattice structures
- **Boundary Effects**: Investigation of finite-size effects in code percolation
- **Error Pattern Studies**: Systematic analysis of error pattern formation

## Usage Patterns

### Basic Percolation Analysis
```python
from simulations.analysis.percolation.percolation_utils import find_percolation_threshold

threshold = find_percolation_threshold(code_distance, error_rates, n_samples)
```

### Cluster Diameter Analysis
```python
from simulations.analysis.percolation.cluster_diameters import compute_cluster_diameters

diameters = compute_cluster_diameters(error_clusters, lattice_geometry)
```

### Toric Code Analysis
```python
from simulations.analysis.percolation.toric_code_percolation import analyze_toric_percolation

percolation_data = analyze_toric_percolation(distance_range, error_rates)
```

### Statistics Collection
```python
from simulations.analysis.percolation.collect_toric_code_stats import collect_percolation_stats

stats = collect_percolation_stats(code_params, n_samples=10000)
```

## Analysis Workflow

1. **Error Pattern Generation**: Generate error patterns under different noise models
2. **Cluster Identification**: Identify connected error clusters using percolation algorithms
3. **Geometric Analysis**: Calculate cluster diameters, shapes, and spatial properties
4. **Threshold Estimation**: Determine percolation thresholds for different code parameters
5. **Statistical Analysis**: Compute cluster size distributions and scaling properties
6. **Visualization**: Generate plots and visualizations of percolation phenomena

## Percolation Concepts in Quantum Error Correction

### Cluster Formation
- **Error Connectivity**: How individual errors connect to form clusters
- **Percolation Threshold**: Critical error rate where large clusters form
- **Cluster Size Distribution**: Statistical properties of error cluster sizes
- **Geometric Properties**: Shape and size characteristics of error clusters

### Threshold Analysis
- **Critical Behavior**: Analysis of behavior near percolation threshold
- **Finite-Size Scaling**: How percolation properties scale with code size
- **Universal Properties**: Code-independent aspects of percolation behavior
- **Threshold Estimation**: Numerical methods for threshold determination

### Applications to Quantum Codes
- **Decoder Performance**: Relationship between percolation and decoder success
- **Error Correction Capacity**: How percolation limits affect code performance
- **Fault Tolerance**: Understanding error propagation through percolation
- **Code Design**: Using percolation insights for code construction

## Integration Points

- **Simulation Framework**: Integration with quantum error correction simulations
- **Data Analysis**: Connection with broader data analysis pipeline
- **Visualization**: Support for percolation-specific plotting and visualization
- **Statistical Analysis**: Integration with statistical analysis tools

This percolation analysis framework provides essential tools for understanding the geometric and statistical properties of error patterns in quantum error correction, enabling deeper insights into the fundamental limits and behavior of quantum codes through the lens of percolation theory.