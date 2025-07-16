# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantum error correction research project focused on **post-selection decoding for general quantum LDPC codes**. The codebase implements decoders (BP+LSD, matching), simulation frameworks, and analysis tools for various quantum error-correcting codes including surface codes, color codes, BB codes, HGP codes, and toric codes.

## Installation and Setup

**Requirements**: Python ≥ 3.11, virtual environment strongly recommended

```bash
git clone --recurse-submodules git@github.com:seokhyung-lee/ldpc-post-selection.git
cd ldpc-post-selection
pip install -e .
```

**Important**: This project includes a modified version of the `ldpc` library in `ext/ldpc` that conflicts with the standard installation. Always use a virtual environment.

## Package Structure

The project is configured as two Python packages:
- `ldpc_post_selection`: Core decoder implementations (`src/ldpc_post_selection/`)
- `simulations`: Simulation framework and analysis tools (`simulations/`)

When writing executable scripts, import using package names:
```python
from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import bplsd_simulation_task_parallel
```

For non-executable library code, relative imports are acceptable.

## Importing Guidelines

- `'simulations/'` and `'src/ldpc_post_selection/'` are currently installed as python packages "simulations" and "ldpc_post_selection"
- Use package names when importing codes for executable scripts
- For non-executable codes, using relative importing is fine

## Core Architecture

### Decoder Framework (`src/ldpc_post_selection/`)
- **Base class**: `SoftOutputsDecoder` - Foundation for all decoders
- **BP+LSD decoder**: `SoftOutputsBpLsdDecoder` - Belief Propagation + List Sequential Decoding
- **Matching decoder**: `SoftOutputsMatchingDecoder` - Minimum-weight perfect matching
- **Integration tools**: `stim_tools.py` for Stim quantum circuit simulator integration
- **Cluster analysis**: `cluster_tools.py` for error cluster investigations

### Simulation Framework (`simulations/`)
- **Parallel execution**: Uses `joblib` for distributed simulation across multiple cores
- **Code-specific simulations**: Surface code, color code, BB code, HGP code, toric code
- **Data collection**: Automated batch processing with configurable parameters
- **Analysis pipeline**: Data aggregation → Metric calculation → Visualization

### External Dependencies (`ext/ldpc/`)
- **Modified LDPC library**: Custom BP+LSD implementation with additional statistics
- **Sliding window decoder**: Cython-based high-performance decoder in `simulations/utils/SlidingWindowDecoder/`

## Development Commands

### Testing
```bash
# Run main decoder tests
python -m pytest tests/test_decoder.py

# Run simulation utility tests  
python -m pytest simulations/tests/

# Run analysis tests
python -m pytest simulations/analysis/tests/
```

### Running Simulations
Simulations are typically executed as Python scripts with command-line arguments:
```bash
# Example surface code simulation
python simulations/surface_code_simulation.py
```

### Analysis Workflows
1. **Data Collection**: Run parallel simulations using code-specific collectors in `simulations/analysis/data_collectors/`
2. **Data Aggregation**: Use `data_aggregation.py` to combine results across parameter sweeps
3. **Analysis**: Open Jupyter notebooks in `simulations/analysis/notebooks/` for visualization and statistical analysis

## Key Architectural Patterns

### Decoder Interface
All decoders implement the `decode()` method returning:
- `prediction`: Estimated error pattern
- `soft_outputs`: Dictionary with statistics (LLR values, cluster info, gap proxies)

### Simulation Data Pipeline
1. **Circuit generation**: Using Stim with specific noise models
2. **Parallel decoding**: Distributed across multiple processes
3. **Statistics collection**: Error rates, soft decoder outputs, cluster analysis
4. **Data storage**: Feather format for DataFrames, compressed NumPy for sparse matrices

### Stim Integration
- Circuits built using `build_circuit.py` utilities
- Detector error models converted to parity check matrices via `stim_tools.py`
- Support for both standard and decomposed error models

## Performance Considerations

- **Cython extensions**: Critical paths in sliding window decoder use Cython
- **Sparse matrices**: CSR/CSC format for memory efficiency with large codes
- **Parallel processing**: `joblib` with `n_jobs` parameter for simulation scaling
- **Batch processing**: Large simulation runs split into manageable chunks

## Important Development Notes

- The `ext/ldpc` submodule contains research-specific modifications to BP+LSD algorithms
- Simulation results are stored in structured directories with metadata
- Analysis notebooks expect specific data formats and directory structures
- Test fixtures use Stim-generated circuits for reproducible testing