# CLAUDE.md

## Simulation Framework (`simulations/`)

This package contains the comprehensive simulation framework for quantum error correction research, focusing on different types of quantum LDPC codes and their error correction performance under various noise models.

## Package Structure

### Core Simulation Scripts

#### `surface_code_simulation.py`
Main simulation script for surface codes with BP+LSD decoding. Implements comprehensive error correction simulations for surface codes under depolarizing noise, collecting detailed statistics on decoding performance, cluster formation, and post-selection metrics.

#### `surface_code_simulation_matching.py`
Surface code simulation using minimum-weight perfect matching (MWPM) decoder. Provides comparative analysis between BP+LSD and matching-based decoding approaches for surface codes.

#### `color_code_simulation.py`
Simulation framework for color codes, a class of topological quantum error-correcting codes. Implements specialized decoding strategies adapted for the unique structure of color codes with both X and Z stabilizers.

#### `bb_simulation.py`
Bivariate Bicycle (BB) code simulation module. Handles the simulation of BB codes, which are quantum LDPC codes with good distance properties. Includes specialized decoding and analysis for these algebraically constructed codes.

#### `bb_sliding_window_simulation.py`
Sliding window simulation for BB codes. Implements windowed decoding approaches that can handle streaming error correction scenarios and provide insights into local vs global decoding performance.

#### `hgp_simulation.py`
Hypergraph Product (HGP) code simulation framework. Simulates HGP codes, which are constructed from classical codes using the hypergraph product construction, providing a rich family of quantum LDPC codes.

#### `toric_code_bitflip_simulation.py`
Specialized simulation for toric codes under bit-flip noise models. Focuses on the analysis of toric codes, which are fundamental examples of topological quantum error correction.

### Package Configuration

#### `__init__.py`
Package initialization file that sets up the simulation framework and provides common imports for simulation scripts.

## Key Features

### Multi-Code Support
- **Surface Codes**: Planar and toric surface codes with different boundary conditions
- **Color Codes**: Triangular and hexagonal color codes with CSS structure
- **BB Codes**: Bivariate bicycle codes with good LDPC properties
- **HGP Codes**: Hypergraph product codes from classical LDPC codes
- **Toric Codes**: Fundamental topological codes for theoretical analysis

### Simulation Capabilities
- **Parallel Processing**: Distributed simulation across multiple CPU cores using joblib
- **Statistical Collection**: Comprehensive error rate and decoder performance metrics
- **Soft Output Analysis**: Collection of decoder soft outputs for post-selection research
- **Cluster Analysis**: Error pattern clustering and geometric analysis
- **Noise Model Support**: Depolarizing, bit-flip, and custom noise models

### Data Pipeline Integration
- **Automated Execution**: Command-line interface for batch simulation runs
- **Data Storage**: Structured output in Feather format for efficient data handling
- **Parameter Sweeps**: Support for systematic parameter exploration
- **Result Aggregation**: Integration with analysis pipeline for result processing

## Usage Patterns

### Basic Simulation Execution
```bash
python simulations/surface_code_simulation.py
python simulations/color_code_simulation.py
```

### Parallel Simulation
```python
from simulations.utils.simulation_utils import bplsd_simulation_task_parallel

results = bplsd_simulation_task_parallel(
    code_params, noise_params, decoder_params, n_jobs=8
)
```

### Data Collection Integration
```python
from simulations.analysis.data_collectors.collect_surface_code_simulation_data import collect_data

collect_data(distance_range, error_rate_range, output_dir)
```

## Simulation Workflow

1. **Circuit Generation**: Use Stim to generate quantum circuits with noise
2. **Parallel Decoding**: Distribute decoding tasks across multiple processes
3. **Statistics Collection**: Gather error rates, soft outputs, and cluster metrics
4. **Data Storage**: Save results in structured format for analysis
5. **Integration**: Connect with analysis pipeline for visualization and post-processing

## Performance Considerations

- **Joblib Parallelization**: Efficient multi-core processing for large simulations
- **Memory Management**: Careful handling of large quantum code matrices
- **Batch Processing**: Chunked execution for memory-efficient long runs
- **Result Caching**: Structured storage to avoid recomputation

This simulation framework provides the foundation for comprehensive quantum error correction research, enabling systematic study of different code families, decoding algorithms, and noise models with detailed statistical analysis capabilities.