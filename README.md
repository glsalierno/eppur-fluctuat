# eppur-fluctuat

![CI](https://github.com/glsalierno/eppur-fluctuat/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-pytest--cov-lightgrey)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](docs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
<!-- Step 3: replace XXXXXXX after Zenodo assigns a DOI (create GitHub release v1.0.0 first, then archive on Zenodo). -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

> ⚠️ **Important**: The “GPU acceleration” flag is reserved for future implementation. Docking is currently performed with **CPU-only AutoDock Vina**. The flag has no effect on `scripts/dock_ligands.py`. Optional **NVIDIA ALCHEMI** FIRE2 refinement (`scripts/dock_ligands_alchemi.py`) can use **CUDA** when PyTorch + ALCHEMI are installed—that path is separate from core Vina docking.

**Eppur fluctuat** — “*And yet it moves*” (Galileo). The name evokes thermal motion, binding ensembles, and honest uncertainty: this pipeline ranks poses by docking score and, when you have almost no measurements, falls back to **interpretable binding-energy ranking** instead of overfit ML.

## Repository layout

| Location | Contents |
|----------|----------|
| `scripts/` | Python entry points (`prepare_protein.py`, `dock_ligands.py`, …) |
| `workspace/inputs/` | Receptor PDB, `ligands.csv`, `curated_pIC50.csv` (paths in `config.yaml`) |
| `workspace/outputs/` | Docking CSVs, poses, QSAR artifacts, HTML report |
| `var/logs/` | Short run logs from docking scripts |
| `var/cache/` | Reusable intermediates (e.g. protonated receptor cache) |
| `var/tmp/` | Ephemeral merged complexes / per-ligand scratch |
| `examples/ache_demo/` | AChE demo data and comparison utilities |
| `examples/tutorial.ipynb` | Step-by-step tutorial (install, config, fallback) |
| `docs/` | MkDocs sources |
| `validation/` | Placeholder benchmark script + README |
| `tests/` | Pytest suite (mocked Vina / Open Babel) |

## What it does

- **Receptor prep** — strip waters/hetero, rigid receptor PDBQT for AutoDock Vina.
- **Docking (strategy 1 — baseline)** — `scripts/dock_ligands.py`: Vina only; single seed from `docking.seed`; ranks by best-mode affinity (**CPU**).
- **Docking (strategy 2 — Vina + NVIDIA ALCHEMI)** — `scripts/dock_ligands_alchemi.py`: same Vina search, then merged receptor–ligand complex → **FIRE2** relaxation with the **ALCHEMI Toolkit** demo MLIP (protein atoms frozen, ligand mobile), extract ligand, **Vina rescoring**. Optional **multi-seed** runs (`uncertainty` in `config.yaml`). *The bundled Demo potential is for pipeline/integration testing only.*
- **QSAR (optional)** — small-data OLS with **leave-one-out** validation, optional held-out test names, optional covalent/non-covalent split, and **ranking fallback** when LOO R² is below a threshold.
- **Report** — matplotlib figures + self-contained **HTML** with optional **py3Dmol** pose snippet.

### Pipeline flow (baseline)

```
workspace/inputs/protein.pdb
        │
        ▼
 prepare_protein.py  ──► receptor_clean.pdb , receptor.pdbqt
        │
        ▼
 dock_ligands.py     ──► docking_results.csv , poses/
        │
        ▼
 build_qsar.py       ──► predictions.csv , metrics.json , model.joblib
        │
        ▼
 generate_report.py  ──► qsar_report.html , PNG figures
```

### Fallback mechanism (short)

If `qsar.fallback_to_ranking` is enabled and pooled LOO **R²** \< `qsar.fallback_loo_r2_threshold`, the pipeline **stops emitting OLS pIC50 predictions** and uses **binding-energy ranking** instead (`prediction_source = fallback_ranking`). See [`docs/fallback_mechanism.md`](docs/fallback_mechanism.md).

## Validation

A **placeholder** benchmark (illustrative AChE-related compounds + synthetic predicted values) lives under [`validation/`](validation/). Run:

```bash
pip install -r validation/requirements_validation.txt
python validation/benchmark_ache.py
```

This writes `parity.png`, `errors.png`, and `benchmark_metrics.json` (gitignored). Replace synthetic predictions with real pipeline outputs before citing metrics in papers.

## Quick start (repository root)

```bash
# 1) Environment (recommended)
conda env create -f environment.yml
conda activate eppur-fluctuat

# 2) Put data under workspace/inputs/
#    workspace/inputs/protein.pdb   workspace/inputs/ligands.csv   workspace/inputs/curated_pIC50.csv

# 3) Baseline pipeline (Vina only)
python scripts/prepare_protein.py
python scripts/dock_ligands.py
python scripts/build_qsar.py
python scripts/generate_report.py
```

### Optional: ALCHEMI docking v2

```bash
pip install -r requirements-alchemi.txt
python scripts/dock_ligands_alchemi.py
```

### AChE demo: baseline vs ALCHEMI + uncertainty

```bash
python examples/ache_demo/AChE_screening_v2/compare_ache_vina_alchemi.py \
  --max-ligands 0 --seeds 3 --exhaustiveness 8 --n-poses 5
```

## Installation

### End users

```bash
pip install -e .
```

Conda remains recommended for RDKit + Open Babel (see `environment.yml`).

### Developers

```bash
pip install -e ".[dev]"
pip install -r requirements-dev.txt
pre-commit install
```

## Testing and code quality

```bash
PYTHONPATH=scripts pytest tests/ --cov=scripts --cov=eppur_fluctuat --cov-report=term-missing
pre-commit run --all-files
```

## Documentation

Documentation is built with **MkDocs** (Material theme). Sources live in [`docs/`](docs/). Local build:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Hosted docs can be deployed via GitHub Pages (`mkdocs gh-deploy`) when you choose.

## Tutorial

See **[`examples/tutorial.ipynb`](examples/tutorial.ipynb)** for an overview of editable install, CLI steps, and the QSAR fallback idea.

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** (style: Black + Ruff, tests required for new behaviour).

## Citation

- Repository: [https://github.com/glsalierno/eppur-fluctuat](https://github.com/glsalierno/eppur-fluctuat)
- Software citation metadata: **[`CITATION.cff`](CITATION.cff)** (Zenodo DOI placeholder until step 3 below).
- Zenodo metadata template: **[`.zenodo.json`](.zenodo.json)**

### Release and DOI (ResearchGate-ready)

Use this **order**: Zenodo archives the **GitHub release**, then you paste the assigned DOI into the repo.

| Step | Action |
|------|--------|
| **1** | Create GitHub release `v1.0.0` |
| **2** | Archive on Zenodo ([connect the GitHub repo](https://help.zenodo.org/docs/profile/repositories/) or manual upload) so Zenodo assigns a **DOI** |
| **3** | Replace `XXXXXXX` in `CITATION.cff` and uncomment/update the DOI badge in this README with the real DOI |
| **4** | *(Optional)* GitHub Pages: `mkdocs gh-deploy` |
| **5** | *(Optional)* Replace placeholder validation data with real pipeline outputs |

After steps **1–3**, the repository is fully **ResearchGate-ready** for software citation.

```bibtex
@software{eppur_fluctuat,
  title = {eppur-fluctuat: protein--ligand docking and small-data QSAR},
  url = {https://github.com/glsalierno/eppur-fluctuat},
  year = {2026},
}
```

## Known limitations

- **No GPU Vina** — `docking.gpu` does not accelerate AutoDock Vina; docking is CPU-bound unless you adopt a future backend.
- **Small-data QSAR** — univariate OLS + LOO; intended for transparency, not maximum accuracy on large sets.
- **Open Babel / RDKit required** for production docking and cheminformatics (tests mock these).
- **ALCHEMI demo potential** is not a validated binding model—swap MLIPs for publication-quality refinement.

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).
