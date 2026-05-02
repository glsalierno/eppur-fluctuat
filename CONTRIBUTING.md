# Contributing

Thank you for improving **eppur-fluctuat**. This document describes how to set up a dev environment, run tests, and submit changes.

## Setup

```bash
git clone https://github.com/glsalierno/eppur-fluctuat.git
cd eppur-fluctuat
conda env create -f environment.yml
conda activate eppur-fluctuat
pip install -e ".[dev]"
# Optional: hook pre-commit
pre-commit install
```

Open Babel (`obabel`) must be on `PATH` for full docking runs; **unit tests mock** `obabel` and Vina so CI stays lightweight.

## Code style

- Formatting: **Black** (`black scripts eppur_fluctuat tests`).
- Linting: **Ruff** (`ruff check scripts eppur_fluctuat tests`).
- Dead code scan: **Vulture** (see CI command; extend `vulture_whitelist.py` if needed).

Enable all hooks locally:

```bash
pre-commit run --all-files
```

## Tests

```bash
PYTHONPATH=scripts pytest tests/ --cov=scripts --cov=eppur_fluctuat --cov-report=term-missing
```

New behaviour should include **pytest** coverage (mocks for Vina / Open Babel / optional ALCHEMI).

## Adding features or fixing bugs

1. Open an issue or short design note when the change affects CLI contracts, `config.yaml`, or scientific defaults.
2. Keep patches focused; preserve backward-compatible paths in `config.yaml` when possible.
3. Document user-visible behaviour in `README.md` and, if relevant, under `docs/`.

## Dependencies

This project builds on **AutoDock Vina**, **Open Babel**, **RDKit**, **BioPython**, **scikit-learn**, and optional **NVIDIA ALCHEMI Toolkit** components for FIRE2 refinement. Cite those tools in publications.

The **QSAR fallback** is implemented in-repo (OLS + LOO + ranking); no external pretrained QSAR model is downloaded—acknowledge the uncertainty-first design if you compare against black-box ML baselines.
