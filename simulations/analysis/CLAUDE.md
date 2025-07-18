# CLAUDE.md

## Data Analysis Framework (`simulations/analysis/`)

This package provides comprehensive data analysis capabilities for quantum error correction simulation results, including statistical analysis, metric computation, and visualization tools for post-selection decoding research.

## Package Structure

### Core Analysis Tools

#### `plotting_helpers.py`
Visualization and plotting utilities for simulation results. Provides standardized plotting functions for error rate curves, cluster distributions, decoder performance metrics, and other key visualizations used in quantum error correction research.

### Package Configuration

#### `__init__.py`
Package initialization file that sets up the analysis framework and provides common imports for analysis modules.

## Subdirectories

### `data_collectors/`
Automated batch processing and data collection tools for different quantum code types. Contains code-specific data collectors that orchestrate large-scale simulation runs and aggregate results across parameter sweeps.

### `legacy/`
Legacy data processing utilities and conversion tools. Maintains compatibility with older data formats and provides migration tools for historical simulation results.

### `percolation/`
Specialized analysis tools for percolation theory applications in quantum error correction. Includes cluster analysis, percolation threshold calculations, and geometric investigations of error patterns.

## Key Features

### Statistical Analysis
- **Error Rate Computation**: Logical error rate calculations across different noise levels
- **Confidence Intervals**: Statistical significance testing and confidence bound estimation
- **Distribution Analysis**: Error pattern and cluster size distribution characterization
- **Comparative Analysis**: Performance comparison between different decoders and codes

### Visualization Capabilities
- **Error Rate Plots**: Standard threshold plots for different code families
- **Cluster Visualization**: Geometric visualization of error clusters and fault patterns
- **Performance Metrics**: Decoder efficiency and resource usage visualizations
- **Statistical Distributions**: Histogram and probability density visualizations

### Data Processing Pipeline
- **Result Aggregation**: Combining simulation results across different parameter sets
- **Data Validation**: Consistency checking and quality assurance for simulation data
- **Format Conversion**: Standardized data format handling and conversion utilities
- **Batch Processing**: Automated processing of large simulation datasets

## Usage Patterns

### Basic Analysis
```python
from simulations.analysis.plotting_helpers import plot_error_rates

plot_error_rates(results_df, code_distances, output_file)
```

### Data Processing
```python
from simulations.analysis.data_collectors.data_aggregation import aggregate_results

aggregated_data = aggregate_results(input_directory, output_directory)
```

### Visualization
```python
from simulations.analysis.plotting_helpers import plot_cluster_distributions

plot_cluster_distributions(cluster_data, save_path)
```

## Analysis Workflow

1. **Data Collection**: Automated collection of simulation results using data collectors
2. **Data Validation**: Quality checks and consistency verification
3. **Statistical Analysis**: Computation of error rates, confidence intervals, and distributions
4. **Visualization**: Generation of plots and figures for research publications
5. **Report Generation**: Automated analysis reports with key findings

## Integration Points

- **Simulation Framework**: Direct integration with simulation scripts for real-time analysis
- **Data Collectors**: Orchestration of large-scale data collection campaigns
- **Visualization Tools**: Standardized plotting for consistent research presentation
- **Legacy Support**: Backward compatibility with historical simulation data

This analysis framework provides the essential tools for extracting meaningful insights from quantum error correction simulations, enabling comprehensive characterization of decoder performance and error patterns across different quantum code families.