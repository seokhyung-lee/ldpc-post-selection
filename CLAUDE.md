# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for quantum LDPC (Low Density Parity Check) code decoding with post-selection. The project includes simulation frameworks for analyzing decoder performance on surface codes, Bivariate Bicycle (BB) codes, and Hypergraph Product (HGP) codes using various decoding strategies.

## Installation and Setup

```bash
# Clone with submodules (includes modified LDPC library)
git clone --recurse-submodules git@github.com:seokhyung-lee/ldpc-post-selection.git
cd ldpc-post-selection

# Install dependencies (requires Python >= 3.11.3)
pip install -r requirements.txt

# Build the modified LDPC library (if needed)
cd src/ldpc_post_selection/ext/ldpc/
pip install -e .
```

**Important**: This project includes a modified version of the `ldpc` package. Use a virtual environment to avoid conflicts with existing installations.

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_decoder.py

# Run tests with unittest
python -m unittest discover tests/

# Run single test file with unittest
python -m unittest tests.test_decoder
```

## Core Architecture

### Decoder Framework (`src/ldpc_post_selection/`)
- **`decoder.py`**: Base `SoftOutputsDecoder` class with concrete implementations:
  - `SoftOutputsBpLsdDecoder`: BP+LSD decoder with soft decision outputs
  - `SoftOutputsMatchingDecoder`: Minimum weight perfect matching decoder
- **`stim_tools.py`**: Utilities for converting Stim circuits to parity check matrices
- **`tools.py`**: General utility functions

### Simulation Framework (`simulations/`)
Three main simulation types, each with parallel batch processing:

1. **Surface Code Simulations** (`surface_code_simulation.py`, `surface_code_simulation_matching.py`)
2. **Bivariate Bicycle Code Simulations** (`bb_simulation.py`) 
3. **Hypergraph Product Code Simulations** (`hgp_simulation.py`)

Each simulation:
- Uses `simulation_utils.py` for batch processing and data management
- Saves results as Feather files (scalar data) + NumPy files (ragged arrays)
- Supports resumable execution by detecting existing batch directories

### Analysis Pipeline (`simulations/analysis/`)
- **`data_collection.py`**: Load and collect simulation results
- **`data_aggregation.py`**: Aggregate data across parameter sweeps  
- **`data_post_processing.py`**: Post-selection analysis and statistics
- **`plotting_helpers.py`**: Visualization utilities
- **`numba_functions.py`**: Optimized numerical computations

### Circuit Building (`simulations/`)
- **`build_circuit.py`**: Surface code and BB code circuit generation
- **`build_random_hgp_circuits.py`**: Random HGP code generation

## Running Simulations

### Surface Code Example
```python
from simulations.surface_code_simulation import simulate

simulate(
    shots=1000000,
    p=0.001,          # Physical error rate
    d=5,              # Code distance  
    T=5,              # Number of rounds
    data_dir="data/surface_d5_T5_p0.001",
    n_jobs=8,         # Parallel workers
    repeat=1,
    decoder_prms={"max_iter": 100}
)
```

### Data Analysis Workflow
```python
from simulations.analysis.data_collection import load_existing_df
from simulations.analysis.data_post_processing import get_df_ps

# Load aggregated simulation data
df = load_existing_df("surface", "pred_llr", "d5_T5_p0.001", "aggregated")

# Apply post-selection analysis
df_ps = get_df_ps(df, method="pred_llr", threshold=0.5)
```

## Data Storage Structure

```
data/
├── surface/           # Surface code results
├── bb/               # Bivariate Bicycle code results  
└── hgp/              # Hypergraph Product code results
    └── d5_T5_p0.001/     # Parameter-specific directory
        ├── batch_0_1000000/   # Batch directory (resumable)
        │   ├── scalar_data.feather
        │   ├── bit_llrs.npy
        │   └── syndrome_llrs.npy
        └── batch_1_1000000/
```

## Key Dependencies

- **Stim**: Quantum circuit simulation and error model generation
- **Modified LDPC library**: BP+LSD decoding with enhanced statistics
- **PyMatching**: Minimum weight perfect matching
- **NumPy/SciPy**: Numerical computation
- **Pandas/PyArrow**: Data handling and serialization
- **Joblib**: Parallel processing

## Development Notes

- The project uses a modified `ldpc` library (submodule) with enhanced BP+LSD that returns additional statistics
- Simulations are designed for HPC environments with batch processing and checkpointing
- All decoders implement the `SoftOutputsDecoder` interface for consistency
- Data is stored in efficient formats (Feather + NumPy) for large-scale analysis