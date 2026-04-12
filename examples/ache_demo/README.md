# Acetylcholinesterase (AChE) demo

This folder preserves the original **target-specific** screening and QSAR materials used during development:

- `AChE_QSAR_Pipeline/` — earlier QSAR pipeline (curated pIC50, scripts, configs).
- `AChE_screening_v2/` — comprehensive docking run vs human AChE (PDB **4EY7**), HTML reports, and poses.
- `screen_alkyl_phosphates.py`, `interaction_analysis.py`, `mechanism_data.py`, `unified_ache_report.py` — legacy helpers (organophosphate-focused heuristics, interaction analysis).

The **general** workflow lives at the repository root (`config.yaml`, `scripts/`). To reproduce a full AChE-style run, point `protein.pdb_file` at a 4EY7 structure, set `protein.reference_ligand_resname` (e.g. `E20`) or explicit `pocket_center`, and run the root scripts.

See the main [README](../../README.md) for commands.
