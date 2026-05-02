# API reference

Python CLIs live under [`scripts/`](../scripts/). Add `scripts` to `PYTHONPATH` (as in `pyproject.toml` test configuration) to import modules such as `prepare_protein` or `dock_ligands`.

| Module | Primary symbols | Role |
|--------|-----------------|------|
| [`prepare_protein.py`](https://github.com/glsalierno/eppur-fluctuat/blob/main/scripts/prepare_protein.py) | `prepare_receptor_for_vina`, `write_clean_protein_pdb`, `obabel_pdb_to_pdbqt`, `main` | Receptor cleaning + PDBQT |
| [`dock_ligands.py`](https://github.com/glsalierno/eppur-fluctuat/blob/main/scripts/dock_ligands.py) | `run_docking`, `main` | Vina docking loop |
| [`docking_common.py`](https://github.com/glsalierno/eppur-fluctuat/blob/main/scripts/docking_common.py) | `load_config`, `resolve_center`, `smiles_to_pdbqt`, `vina_cpu_count`, … | Shared docking utilities |
| [`build_qsar.py`](https://github.com/glsalierno/eppur-fluctuat/blob/main/scripts/build_qsar.py) | `should_use_ranking_fallback`, `ensure_computed_features`, `main` | QSAR / fallback |
| [`predict.py`](https://github.com/glsalierno/eppur-fluctuat/blob/main/scripts/predict.py) | `main` | Reapply saved `model.joblib` |
| [`generate_report.py`](https://github.com/glsalierno/eppur-fluctuat/blob/main/scripts/generate_report.py) | `main`, `pdbqt_to_pdb_text` | Figures + HTML report |

Install metadata only:

| Package | Symbol | Role |
|---------|--------|------|
| [`eppur_fluctuat/__init__.py`](https://github.com/glsalierno/eppur-fluctuat/blob/main/eppur_fluctuat/__init__.py) | `__version__` | Version string (`1.0.0`) |

Auto-generated API pages can be added later with **mkdocstrings** once modules are packaged for discovery.
