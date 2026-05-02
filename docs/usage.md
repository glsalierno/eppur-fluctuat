# Usage

## Inputs

| File | Role |
|------|------|
| Receptor PDB | Cleaned to protein-only PDB and rigid PDBQT in `prepare_protein.py`. |
| `ligands.csv` | Columns **`name`**, **`smiles`** (required). Optional metadata columns are preserved when merging. |
| `curated_pIC50.csv` | Experimental **`name`**, **`pIC50`**, optional **`ic50_nM`** for QSAR training/evaluation. |

Paths are controlled entirely by **`config.yaml`** (`paths` and `protein` sections).

## Pocket definition

- `protein.pocket_center: auto` — centroid of Cα atoms for `protein.chain`.
- Or specify `[x, y, z]` in Å.
- Optional `reference_ligand_resname` — centroid of that hetero residue in the raw PDB.

Box size comes from `protein.pocket_size` (Å).

## Docking

`scripts/dock_ligands.py` converts SMILES to PDBQT via Open Babel, maps the receptor once per run, docks each ligand, and writes:

- `workspace/outputs/docking_results.csv` — best-mode affinity (kcal/mol), pose paths, errors.
- Per-ligand PDBQT poses under `workspace/outputs/docked/` (default).

`docking.cpu` caps CPU threads (0 = auto). **`docking.gpu` is reserved** and does not change Vina execution.

## QSAR and predictions

`scripts/build_qsar.py` merges docking scores into `computed_features.csv` if needed, fits **univariate OLS** on configured descriptors (`qsar.use_descriptors`), evaluates **leave-one-out** performance, optionally scores a held-out name list, and may switch to **pure ranking** by binding energy when LOO R² is below `qsar.fallback_loo_r2_threshold` (if `qsar.fallback_to_ranking` is true).

Outputs include `predictions.csv`, `metrics.json`, `loo_predictions.csv`, and `model.joblib`.

## Report

`scripts/generate_report.py` reads predictions and metrics, writes matplotlib figures, and emits a self-contained **`qsar_report.html`** (optional py3Dmol snippet when poses are available).
