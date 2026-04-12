# eppur-fluctuat

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black-compatible](https://img.shields.io/badge/code%20style-readable-lightgrey.svg)]()

**Eppur fluctuat** — “*And yet it moves*” (Galileo). The name evokes thermal motion, binding ensembles, and honest uncertainty: this pipeline ranks poses by docking score and, when you have almost no measurements, falls back to **interpretable binding-energy ranking** instead of overfit ML.

## What it does

- **Receptor prep** — strip waters/hetero, rigid receptor PDBQT for AutoDock Vina.
- **Docking** — Vina box from `config.yaml` (`pocket_center: auto` uses Cα centroid of the chosen chain; optional `reference_ligand_resname` for co-ligand–centered boxes).
- **QSAR (optional)** — small-data OLS with **leave-one-out** validation, optional held-out test names, optional covalent/non-covalent split, and **ranking fallback** when LOO R² is below a threshold.
- **Report** — matplotlib figures + self-contained **HTML** with optional **py3Dmol** pose snippet.

> **Note:** The official **AutoDock Vina** Python API is **CPU-only**. The `docking.gpu` flag is reserved for future GPU backends; Vina still uses multiple CPU threads.

## Quick start (repository root)

```bash
# 1) Environment (recommended)
conda env create -f environment.yml
conda activate eppur-fluctuat

# 2) Put your receptor and ligand table (see data/ for column hints)
#    data/protein.pdb   data/ligands.csv   data/curated_pIC50.csv

# 3) Run pipeline
python scripts/prepare_protein.py
python scripts/dock_ligands.py
python scripts/build_qsar.py
python scripts/generate_report.py
```

Outputs appear under `outputs/` (CSV, JSON, HTML, PNG). AChE-specific history and files are under [`examples/ache_demo/`](examples/ache_demo/).

## Installation

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate eppur-fluctuat
```

Conda brings **RDKit**, **Open Babel** (`obabel`), and **OpenMM**; `pip` installs **Vina** and **py3Dmol**.

### Option B — pip + system Open Babel

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Install Open Babel so `obabel` is on PATH, e.g.:
#   conda install -c conda-forge openbabel
```

## Configuration

Edit **`config.yaml`** at the repo root:

| Section | Purpose |
|--------|---------|
| `protein` | `pdb_file`, `chain`, `pocket_center` (`auto` or `[x,y,z]`), `pocket_size`, optional `reference_ligand_resname` |
| `docking` | `exhaustiveness`, `num_modes`, `energy_range` (reserved / documentation), `gpu`, `cpu`, `seed` |
| `qsar` | `use_descriptors` (e.g. `binding_energy` → column `binding_energy_kcal_mol`), fallback, test names, covalent split |
| `paths` | Input/output paths relative to the project root |

## Command-line tools

| Script | Role |
|--------|------|
| `scripts/prepare_protein.py` | Clean PDB → `outputs/receptor_clean.pdb`, `outputs/receptor.pdbqt` |
| `scripts/dock_ligands.py` | Dock `data/ligands.csv` → `outputs/docking_results.csv` |
| `scripts/build_qsar.py` | Merge features + experimental pIC50 → `outputs/predictions.csv`, `outputs/model.joblib`, LOO table |
| `scripts/predict.py` | Reapply saved model to current features |
| `scripts/generate_report.py` | Figures + `outputs/qsar_report.html` |

All scripts accept `--config path/to/config.yaml` (default: `./config.yaml`).

## Outputs

| Artifact | Description |
|----------|-------------|
| `outputs/docking_results.csv` | Per-ligand Vina affinity, pose paths |
| `outputs/computed_features.csv` | Merged ligand + docking descriptors (auto-built if missing) |
| `outputs/predictions.csv` | Predicted pIC50 / IC50 or fallback ranking |
| `outputs/metrics.json` | LOO / test metrics |
| `outputs/loo_predictions.csv` | Leave-one-out table |
| `outputs/qsar_report.html` | Interactive report (embedded PNG; optional py3Dmol) |

## Citation

If you use this software in research, please cite the repository and the underlying tools (**AutoDock Vina**, **RDKit**, etc.) as appropriate:

```bibtex
@software{eppur_fluctuat,
  title = {eppur-fluctuat: protein--ligand docking and small-data QSAR},
  url = {https://github.com/glsalierno/eppur-fluctuat},
  year = {2026},
}
```

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).
