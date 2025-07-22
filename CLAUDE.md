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

The project is configured as two Python packages, which are already installed:
- `ldpc_post_selection`: Core decoder implementations (`src/ldpc_post_selection/`)
- `simulations`: Simulation framework and analysis tools (`simulations/`)

When writing executable scripts, import using package names:
```python
from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from simulations.utils.simulation_utils import bplsd_simulation_task_parallel
```

For non-executable library code, relative imports are acceptable.

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

### Analysis Workflows
1. **Data Collection**: Run parallel simulations using simulation scripts in `simulations/`.
2. **Data Aggregation**: Use code-specific scripts in `simulations/analysis/data_collectors/` for aggregating saved data.
3. **Analysis**: Use Jupyter notebooks in `simulations/analysis/notebooks/` for visualization and statistical analysis.

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

## Cluster Definitions

Important definitions related to clusters:
- Cluster: a set of faults.
- Cluster size: number of faults in a cluster
- Cluster LLR: summation of LLRs of faults in a cluster
- Cluster size norm of order `alpha`: `alpha`-norm of the cluster size vector, i.e., sum(cluster_sizes**alpha)**(1/alpha)
- Cluster LLR norm of order `alpha`: `alpha`-norm of the cluster LLR vector.
- Cluster size norm fraction: Cluster size norm divided by total number of faults (including faults that don't belong to any clusters).
- Cluster LLR norm fraction: Cluster LLR norm divided by the summation of all fault LLRs

## Documents

`docs/` contains documents in markdown format.

- `sliding_window_decoding.md`: Comprehensive descriptions on sliding window decoding
- `sliding_window_real_time_post_selection.md`: Real-time post-selection strategy for sliding window decoding

## Documentation Guidelines

- Whenever a document is added to `docs/`, add a concise description in this `CLAUDE.md` file under the "Documents" section