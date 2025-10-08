# Repository Guidelines

## Project Structure & Module Organization
- `src/ldpc_post_selection/`: Core library with decoder hierarchy (`SoftOutputsDecoder`, `SoftOutputsBpLsdDecoder`, matching decoder), Stim integration utilities, and cluster analysis helpers.
- `simulations/`: Batch runners (`run_decode`), analysis scripts in `analysis/`, sliding-window tooling under `analysis/sliding_window`, and prebuilt circuits in `data/hgp_prebuilt/circuits`.
- `tests/`: Mirrors package layout.
- `ext/ldpc/`: Submodule exposing the customized BP+LSD backend; never edit in place without upstream coordination.
- `docs/` and `notebooks/`: Reference material and exploratory studies; commit notebooks only when cleared of secrets and oversized outputs.

## Architecture & Key Components
Decoders expose `decode()` returning predictions plus soft-output statistics. Simulation flow: Stim circuit synthesis → joblib-parallel decoding → Feather/NumPy aggregation → notebook visualization. Sliding-window analysis contrasts ordinary and post-selection modes, reporting `p_fail`, `delta_p_fail`, and cluster metrics.

## Build, Test, and Development Commands
- `pip install -e .`: Installs the packages alongside the bundled `ext/ldpc`.
- `git submodule update --init --recursive`: Ensures decoder sources match the expected revision.

## Coding Style & Naming Conventions
Use Python 3.11, explicit type annotations, and English docstrings following the repository template. Keep one responsibility per function, reuse helpers before adding new ones, and rely on vectorized NumPy/SciPy/pandas operations. Modules and files use snake_case; classes use PascalCase; constants are ALL_CAPS. Limit Python lines to 100 characters and trim notebook outputs before committing.

## Testing Guidelines
Place new tests beside the code under test (e.g., `tests/ldpc_post_selection/test_decoder.py`). Name tests `test_<behavior>` and seed random generators for Monte Carlo runs. Cover new branches, especially interactions with the modified `ldpc` backend and Stim integration helpers.

## Commit & Pull Request Guidelines
Write present-tense, scoped commits (`Add HGP cluster metrics`, `Refactor stim tools`). Document submodule updates explicitly. Pull requests must summarize the motivation, implementation, validation evidence (tests or simulation tables), and note any required environment tweaks. Attach plots or notebook exports when analysis informs the change.

## Environment & Submodule Notes
Work inside a virtual environment to avoid conflicting `ldpc` installs. Never commit large simulation dumps; stash under `data/` in gitignored form or external storage. When modifying `ext/ldpc`, update the submodule pointer, link the upstream PR, and summarize compatibility impacts in the PR body. Store credentials in untracked `.env` files and document new variables in `docs/`.
