# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantum error correction research project implementing **LDPC (Low-Density Parity-Check) post-selection decoding** for quantum codes. The project focuses on studying various quantum error correction codes (surface codes, color codes, BB codes, HGP codes, etc.) with advanced decoding techniques, particularly the BP+LSD (Belief Propagation + Local Search) decoder with soft outputs.

## Installation and Setup

**Requirements**: Python ≥ 3.11

**Warning**: Use a virtual environment to avoid conflicts with existing `ldpc` package installations, as this project includes a modified version.

```bash
git clone --recurse-submodules git@github.com:seokhyung-lee/ldpc-post-selection.git
cd ldpc-post-selection
pip install -e .
```

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_decoder.py

# Run tests in analysis module
pytest simulations/analysis/tests/

# Run tests in external ldpc module
pytest ext/ldpc/python_test/
```

### Running Simulations
```bash
# Surface code simulation example
python simulations/surface_code_simulation.py

# BB code simulation
python simulations/bb_simulation.py

# Color code simulation
python simulations/color_code_simulation.py

# Data collection scripts
python simulations/analysis/data_collectors/collect_surface_code_simulation_data.py
```

### Jupyter Notebooks
```bash
# Launch Jupyter for analysis notebooks
jupyter notebook simulations/analysis/notebooks/
```

## Code Architecture

### Core Package (`src/ldpc_post_selection/`)
- **Base classes**: `SoftOutputsDecoder` (abstract base class)
- **BP+LSD decoder**: `SoftOutputsBpLsdDecoder` - Main decoder with soft outputs and clustering analysis
- **Matching decoder**: `SoftOutputsMatchingDecoder` - Minimum-weight perfect matching decoder
- **Stim integration**: `stim_tools.py` - Tools for working with Google's Stim quantum circuit simulator
- **Utilities**: `utils.py`, `tools.py` - Helper functions for clustering, analysis, and circuit manipulation

### Simulation Framework (`simulations/`)
- **Quantum code simulators**: Surface codes, color codes, BB codes, HGP codes, toric codes
- **Parallel execution**: Uses `joblib` for parallel simulation tasks
- **Data storage**: Feather format for efficient storage, compressed NumPy arrays for sparse matrices
- **Circuit building**: `utils/build_circuit.py` for constructing quantum circuits

### Analysis Pipeline (`simulations/analysis/`)
- **Data collectors**: Automated data collection for different code types
- **Data aggregation**: `data_aggregation.py` - Statistical analysis and binning
- **Percolation analysis**: Tools for threshold calculations
- **Plotting utilities**: `plotting_helpers.py` - Visualization functions
- **Statistical tools**: `numpy_utils.py` - NumPy-based statistical analysis

### External Dependencies
- **Modified LDPC library**: `ext/ldpc/` (git submodule) - Enhanced with additional BP+LSD statistics
- **SlidingWindowDecoder**: `simulations/utils/SlidingWindowDecoder/` (git submodule)
- **Stim**: Google's quantum circuit simulator
- **Custom packages**: `quits`, `color-code-stim` (installed from GitHub)

## Key Development Patterns

### Decoder Interface
All decoders inherit from `SoftOutputsDecoder` and implement:
- `decode(syndrome)` - Single syndrome decoding
- `decode_batch(syndromes)` - Batch syndrome decoding
- Return format: `(predictions, soft_outputs)` where `soft_outputs` contains LLR, detector density, clustering stats

### Simulation Workflow
1. **Circuit generation**: Build quantum circuits with noise models
2. **Error simulation**: Generate syndromes using Stim
3. **Decoding**: Apply BP+LSD or matching decoders
4. **Data collection**: Store results in structured format (Feather files)
5. **Analysis**: Aggregate data, compute statistics, generate plots

### Data Management
- **Batch processing**: Large simulations split into manageable batches
- **Efficient storage**: Feather format for DataFrames, compressed arrays for sparse matrices
- **Parallel execution**: `joblib` for multiprocessing across CPU cores
- **Structured organization**: Separate directories for different code types and parameter sets

## Testing Approach

- **pytest** framework with fixtures for circuit data
- **Limited test coverage** - mainly focused on decoder functionality
- **Test structure**: 
  - `tests/test_decoder.py` - Core decoder tests
  - `simulations/analysis/tests/` - Data processing tests
  - `ext/ldpc/python_test/` - Modified LDPC library tests

## Data Analysis Workflow

### Typical Research Pipeline
1. **Parameter sweeps**: Vary distance, noise rate, rounds
2. **Data collection**: Run simulations with `collect_*_simulation_data.py` scripts
3. **Aggregation**: Use `data_aggregation.py` for statistical analysis
4. **Visualization**: Generate plots with `plotting_helpers.py`
5. **Threshold estimation**: Use percolation analysis tools

### Important File Formats
- **Feather files**: Pandas DataFrames with simulation results
- **NPZ files**: Compressed NumPy arrays for sparse matrices
- **JSON files**: Configuration and metadata

## Git Submodules

This project uses git submodules for external dependencies:
- `ext/ldpc/` - Modified LDPC library
- `simulations/utils/SlidingWindowDecoder/` - Sliding window decoder

Always clone with `--recurse-submodules` or update submodules with:
```bash
git submodule update --init --recursive
```

## Research Context

This codebase supports academic research on:
- **Post-selection decoding techniques** for quantum error correction
- **Performance comparison** of BP+LSD vs matching decoders
- **Threshold estimation** using percolation theory
- **Soft information analysis** from decoding algorithms
- **LDPC code performance** under various noise models