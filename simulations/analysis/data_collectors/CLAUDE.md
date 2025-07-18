# CLAUDE.md

## Data Collection Framework (`simulations/analysis/data_collectors/`)

This package provides automated batch processing and data collection tools for large-scale quantum error correction simulations. It orchestrates systematic data collection across different code families, parameter sweeps, and decoder configurations.

## Package Structure

### Code-Specific Data Collectors

#### `collect_surface_code_simulation_data.py`
Automated data collection for surface code simulations. Orchestrates parameter sweeps across different code distances, noise levels, and decoder configurations for comprehensive surface code performance analysis.

#### `collect_color_code_simulation_data.py`
Data collection framework for color code simulations. Handles the systematic collection of color code performance data across various geometric configurations and noise models.

#### `collect_bb_simulation_data.py`
Bivariate Bicycle (BB) code data collection module. Manages automated collection of BB code simulation results, including performance metrics and statistical analysis across different code parameters.

#### `collect_bb_sliding_window_simulation_data.py`
Specialized data collector for BB code sliding window simulations. Handles the collection of windowed decoding performance data and streaming error correction metrics.

#### `collect_hgp_simulation_data.py`
Hypergraph Product (HGP) code data collection framework. Orchestrates systematic data collection for HGP codes constructed from different classical LDPC codes.

### Core Data Processing Tools

#### `data_collection.py`
Generic data collection utilities and common functions. Provides the foundational tools and patterns used by code-specific data collectors for consistent data handling and processing.

#### `data_aggregation.py`
Data aggregation tools for combining simulation results across parameter sweeps. Handles the consolidation of distributed simulation results into unified datasets for analysis.

#### `data_metric_calculation.py`
Statistical metric computation module. Provides functions for calculating key performance metrics including error rates, confidence intervals, decoder efficiency measures, and cluster statistics.

#### `data_post_processing.py`
Post-processing utilities for simulation data. Handles data cleaning, validation, format conversion, and preparation for analysis and visualization.

### Package Configuration

#### `__init__.py`
Package initialization file that sets up the data collection framework and provides common imports for data collector modules.

## Subdirectories

### `numpy_utils/`
Optimized numerical computing utilities for efficient data processing. Contains NumPy-based tools for high-performance statistical calculations and data manipulation.

## Key Features

### Automated Data Collection
- **Parameter Sweeps**: Systematic exploration of code distances, noise levels, and decoder parameters
- **Batch Processing**: Efficient handling of large-scale simulation campaigns
- **Resource Management**: Intelligent scheduling and resource allocation for parallel simulations
- **Progress Tracking**: Real-time monitoring and reporting of collection progress

### Data Processing Pipeline
- **Result Aggregation**: Combining results from distributed simulation runs
- **Statistical Analysis**: Computation of error rates, confidence intervals, and performance metrics
- **Data Validation**: Quality assurance and consistency checking
- **Format Standardization**: Consistent data formats across different code families

### Quality Assurance
- **Error Detection**: Identification and handling of failed simulations
- **Data Integrity**: Verification of simulation result consistency
- **Reproducibility**: Ensuring consistent results across different runs
- **Documentation**: Comprehensive metadata and parameter tracking

## Usage Patterns

### Basic Data Collection
```python
from simulations.analysis.data_collectors.collect_surface_code_simulation_data import collect_data

collect_data(
    distance_range=[3, 5, 7, 9],
    error_rate_range=[0.001, 0.01, 0.1],
    output_directory="results/surface_code"
)
```

### Parameter Sweep Configuration
```python
from simulations.analysis.data_collectors.data_collection import run_parameter_sweep

results = run_parameter_sweep(
    code_params, decoder_params, noise_params, n_jobs=16
)
```

### Data Aggregation
```python
from simulations.analysis.data_collectors.data_aggregation import aggregate_simulation_results

aggregated_df = aggregate_simulation_results(input_dir, output_dir)
```

## Collection Workflow

1. **Parameter Definition**: Define parameter spaces for systematic exploration
2. **Batch Generation**: Create simulation tasks for parallel execution
3. **Distributed Execution**: Run simulations across multiple processes/nodes
4. **Result Collection**: Gather and validate simulation outputs
5. **Data Aggregation**: Combine results into unified datasets
6. **Post-Processing**: Apply statistical analysis and format conversion
7. **Quality Assurance**: Validate data integrity and consistency

## Performance Considerations

- **Parallel Processing**: Efficient multi-core and multi-node execution
- **Memory Management**: Careful handling of large datasets and sparse matrices
- **Storage Optimization**: Efficient data storage formats (Feather, HDF5)
- **Checkpoint/Resume**: Robust handling of interrupted collection runs
- **Resource Scaling**: Dynamic adaptation to available computational resources

This data collection framework enables systematic and reproducible large-scale quantum error correction research by automating the complex process of parameter space exploration and result aggregation across different code families and decoder configurations.