# eppur-fluctuat

**eppur-fluctuat** is a Python pipeline for rigid-receptor docking with **AutoDock Vina** (CPU) and a **small-data QSAR** layer: univariate ordinary least squares with leave-one-out validation, optional train/test splits, and a **binding-energy ranking fallback** when the model is not statistically convincing.

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate eppur-fluctuat
pip install -e ".[dev]"   # optional developer extras
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install rdkit-pypi   # or use conda-forge RDKit
# Open Babel CLI must be on PATH as ``obabel``.
```

## Quick start

1. Place `protein.pdb`, `ligands.csv`, and optional `curated_pIC50.csv` under paths given in `config.yaml` (default layout uses `workspace/inputs/`).
2. Run:

```bash
python scripts/prepare_protein.py
python scripts/dock_ligands.py
python scripts/build_qsar.py
python scripts/generate_report.py
```

Artifacts land under `workspace/outputs/` (CSVs, HTML report, figures).

## Important note on GPU

The configuration flag `docking.gpu` does **not** enable GPU docking today: **AutoDock Vina runs on CPU** in this workflow. See the repository README for details.

## Further reading

- [Usage](usage.md) — inputs, outputs, configuration.
- [Fallback mechanism](fallback_mechanism.md) — when ranking replaces regression.
- [API reference](api.md) — main Python entry points.
