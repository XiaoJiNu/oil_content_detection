# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository reproduces the best method from a research paper on oil content detection in camellia seeds using hyperspectral imaging. The core pipeline is: **Spectral Set II + Genetic Algorithm (GA) + Partial Least Squares Regression (PLSR)**.

Currently uses simulated hyperspectral data; designed to be swapped with real measurements once hardware is available.

## Development Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn
```

### Generate Simulated Data
```bash
python scripts/generate_simulated_set_II.py
```
Creates new random hyperspectral cube and ROI mean spectra in `data/processed/set_II/`.

### Run the Full Pipeline
```bash
PYTHONPATH=src MKL_THREADING_LAYER=SEQUENTIAL \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/run_best_method.py
```
This runs GA wavelength selection + PLSR training/testing, printing R² and RMSE metrics.

**Note:** Threading environment variables prevent numerical library conflicts during GA cross-validation.

### Testing & Linting
- Run tests: `pytest -q`
- Linting: `ruff check src tests`
- Formatting: `black src tests`
- Coverage: `pytest --cov=src --cov-report=term-missing`

## Architecture

### Core Pipeline Flow

1. **Data Loading** (`models/plsr_best.py:load_dataset`)
   - Reads `data/processed/set_II/mean_spectra.csv`
   - Expects columns: `sample_id`, `wl_<wavelength>`, `oil_content`
   - Extracts feature matrix X (all `wl_*` columns) and target y (`oil_content`)

2. **Train/Test Split** (`models/plsr_best.py:train_plsr_best`)
   - Test size: 34/102 samples (matching paper protocol)
   - Random state: 2024

3. **GA Wavelength Selection** (`feature_selection/ga_selector.py:GeneticAlgorithmSelector`)
   - Population: 12 individuals, 10 generations
   - Target: ~18 wavelengths (range: 10-22)
   - Fitness: 3-fold CV R² with soft penalty for feature count deviation
   - Selection: tournament (3 candidates), crossover rate: 0.85, mutation rate: 0.04
   - Early stopping: patience=4 generations without improvement
   - **Key:** Each individual is a boolean mask over wavelengths; fitness evaluates PLSR with `n_components = min(10, feature_count // 2)`

4. **PLSR Training** (`models/plsr_best.py:train_plsr_best`)
   - Uses only GA-selected wavelengths
   - Components: `min(10, selected_count // 2)`
   - No scaling (`scale=False`)
   - Outputs train/test R² and RMSE

### Module Responsibilities

- `src/oil_content_detection/feature_selection/ga_selector.py`: Genetic algorithm for feature selection; core logic in `GeneticAlgorithmSelector.fit()`
- `src/oil_content_detection/models/plsr_best.py`: End-to-end pipeline orchestration; entry point `train_plsr_best()`
- `scripts/run_best_method.py`: CLI wrapper setting threading env vars and calling `run_and_print()`
- `scripts/generate_simulated_set_II.py`: Data generator for testing without real hyperspectral hardware

### Configuration

Both GA and pipeline parameters live in dataclasses:
- `GAConfig` (ga_selector.py): GA hyperparameters
- `RunConfig` (plsr_best.py): Data path, test size, GA settings

To adjust GA generations or population size, modify `RunConfig` fields when calling `train_plsr_best()`.

### Data Expectations

**Production data** must match `data/processed/set_II/mean_spectra.csv` format:
- CSV with columns: `sample_id`, `wl_400`, `wl_401`, ..., `wl_1000`, `oil_content`
- Each row is one sample's ROI-averaged spectrum + measured oil percentage
- Replace simulated data by overwriting this file or changing `RunConfig.data_path`

## Project-Specific Notes

- **Threading constraints**: Numerical library conflicts require single-threaded execution during GA cross-validation. Always set threading env vars when running `run_best_method.py`.
- **No tests yet**: `tests/` is empty; add pytest fixtures for GA convergence and PLSR reproducibility when solidifying the pipeline.
- **中文沟通**: Use Chinese for all user communication per AGENTS.md.