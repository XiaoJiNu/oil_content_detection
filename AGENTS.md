# Repository Guidelines
This project targets oil content detection from hyperspectral imagery; use this guide to keep contributions consistent and auditable.

## Project Structure & Module Organization
Place production code in `src/oil_content_detection/` with submodules for `data`, `models`, and `pipelines`. Tests belong in `tests/`, mirroring the package layout (e.g., `tests/models/test_spectra.py`). Store experiments and analysis notebooks under `notebooks/`, and place raw/processed assets in `data/raw/` and `data/processed/` respectively (never commit files larger than 50MB; use `.gitignore`). Keep reference specs, including `docs/reference_docs/功能需求文档/含油量检测需求.md`, under `docs/`.

## Build, Test, and Development Commands
Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`. Install dependencies once `requirements.txt` is updated: `pip install -r requirements.txt`. Run linting with `ruff check src tests` and formatting with `black src tests`. Execute the suite via `pytest -q`. For quick smoke tests of notebooks, run `papermill notebooks/example.ipynb /tmp/out.ipynb`.

## Coding Style & Naming Conventions
Target Python 3.10+, follow PEP 8, and annotate public APIs with type hints. Use snake_case for functions and variables, PascalCase for classes, and UPPER_SNAKE_CASE for constants. Module names should describe their responsibility (`spectral_preprocessing.py`, `model_trainer.py`). Import order should follow `ruff`'s default grouping.

## Testing Guidelines
Unit tests must accompany every new module, using `pytest` fixtures for shared data. Name test files `test_<module>.py` and functions `test_<behavior>()`. Include regression tests for spectral feature extraction and any learned thresholds. Maintain ≥80% coverage (`pytest --cov=src --cov-report=term-missing`); document skipped tests with a TODO back to an issue.

## Commit & Pull Request Guidelines
Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) and keep the subject ≤72 characters. Reference related issues in the body (`Refs #12`). PRs should describe motivation, implementation summary, validation commands, and attach sample outputs or plots when touching models. Request at least one reviewer and ensure CI passes before requesting merge.

## 用中文和我沟通