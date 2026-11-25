"""End-to-end pipeline smoke tests with mocked externals."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_e2e_creates_expected_csvs(tmp_project: Path, monkeypatch: pytest.MonkeyPatch, tiny_pdb_text: str) -> None:
    import build_qsar
    import dock_ligands
    import prepare_protein

    cfg = tmp_project / "config.yaml"
    cfg.write_text(
        """
protein:
  pdb_file: workspace/inputs/protein.pdb
  chain: "A"
  pocket_center: [0, 0, 0]
  pocket_size: [10, 10, 10]
docking:
  exhaustiveness: 1
  num_modes: 2
  seed: 1
  cpu: 2
  gpu: false
qsar:
  fallback_to_ranking: true
  fallback_loo_r2_threshold: 10.0
  use_descriptors: [binding_energy]
  min_train_samples: 3
paths:
  ligands: workspace/inputs/ligands.csv
  docking_results: workspace/outputs/docking_results.csv
  computed_features: workspace/outputs/computed_features.csv
  curated_pIC50: workspace/inputs/curated_pIC50.csv
  predictions: workspace/outputs/predictions.csv
  model_dir: workspace/outputs
  loo_table: workspace/outputs/loo_predictions.csv
  receptor_clean: workspace/outputs/receptor_clean.pdb
  receptor_pdbqt: workspace/outputs/receptor.pdbqt
  docked_dir: workspace/outputs/docked
  logs_dir: var/logs
""",
        encoding="utf-8",
    )

    (tmp_project / "workspace" / "inputs" / "protein.pdb").write_text(tiny_pdb_text, encoding="utf-8")

    pd.DataFrame(
        [
            {"name": "A", "smiles": "CC"},
            {"name": "B", "smiles": "CCC"},
            {"name": "C", "smiles": "CCCC"},
        ]
    ).to_csv(tmp_project / "workspace" / "inputs" / "ligands.csv", index=False)

    pd.DataFrame(
        [
            {"name": "A", "pIC50": 5.0, "ic50_nM": 10000.0},
            {"name": "B", "pIC50": 6.0, "ic50_nM": 1000.0},
            {"name": "C", "pIC50": 7.0, "ic50_nM": 100.0},
        ]
    ).to_csv(tmp_project / "workspace" / "inputs" / "curated_pIC50.csv", index=False)

    monkeypatch.chdir(tmp_project)
    monkeypatch.setattr("shutil.which", lambda _: "/fake/obabel")

    def fake_ob_run(cmd, check=True, capture_output=True, text=True, **_kw):  # noqa: ARG001
        out_idx = cmd.index("-O") + 1
        Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[out_idx]).write_text("ROOT\nTORSDOF 0\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("prepare_protein.subprocess.run", side_effect=fake_ob_run):
        monkeypatch.setattr("sys.argv", ["prepare_protein.py", "--config", str(cfg)])
        prepare_protein.main()

    def ctor(**kwargs):
        inst = MagicMock()
        inst.energies.return_value = [[[-7.0, 0, 0, 0, 0, 0]]]
        return inst

    def fake_smiles(_smi: str, out_pdbqt: Path) -> None:
        out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        out_pdbqt.write_text("ROOT\nTORSDOF 0\n", encoding="utf-8")

    monkeypatch.setattr(dock_ligands, "smiles_to_pdbqt", fake_smiles)
    with patch.object(dock_ligands, "Vina", side_effect=ctor):
        monkeypatch.setattr("sys.argv", ["dock_ligands.py", "--config", str(cfg)])
        dock_ligands.main()

    dock_csv = tmp_project / "workspace" / "outputs" / "docking_results.csv"
    assert dock_csv.is_file()
    ddf = pd.read_csv(dock_csv)
    assert "affinity_kcal_mol" in ddf.columns
    assert "pose_pdbqt" in ddf.columns

    cf = tmp_project / "workspace" / "outputs" / "computed_features.csv"
    if cf.is_file():
        cf.unlink()

    monkeypatch.setattr("sys.argv", ["build_qsar.py", "--config", str(cfg)])
    build_qsar.main()

    pred = tmp_project / "workspace" / "outputs" / "predictions.csv"
    assert pred.is_file()
    pdf = pd.read_csv(pred)
    for col in ("name", "prediction_source", "pIC50_pred", "rank_by_binding_energy"):
        assert col in pdf.columns


def test_e2e_flat_docking_scores_trigger_ranking_fallback(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flat docking scores force weak structure–activity mapping; ranking fallback still emits predictions."""
    import build_qsar

    cfg = tmp_project / "config.yaml"
    cfg.write_text(
        """
qsar:
  fallback_to_ranking: true
  fallback_loo_r2_threshold: 10.0
  use_descriptors: [binding_energy]
  min_train_samples: 3
paths:
  ligands: workspace/inputs/ligands.csv
  docking_results: workspace/outputs/docking_results.csv
  computed_features: workspace/outputs/computed_features.csv
  curated_pIC50: workspace/inputs/curated_pIC50.csv
  predictions: workspace/outputs/predictions.csv
  model_dir: workspace/outputs
  loo_table: workspace/outputs/loo_predictions.csv
""",
        encoding="utf-8",
    )
    (tmp_project / "workspace" / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_project / "workspace" / "outputs").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"name": "A", "smiles": "CC"},
            {"name": "B", "smiles": "CCC"},
            {"name": "C", "smiles": "CCCC"},
        ]
    ).to_csv(tmp_project / "workspace" / "inputs" / "ligands.csv", index=False)
    pd.DataFrame(
        [
            {"name": "A", "affinity_kcal_mol": -7.0},
            {"name": "B", "affinity_kcal_mol": -7.0},
            {"name": "C", "affinity_kcal_mol": -7.0},
        ]
    ).to_csv(tmp_project / "workspace" / "outputs" / "docking_results.csv", index=False)
    pd.DataFrame(
        [
            {"name": "A", "pIC50": 5.0, "ic50_nM": 10000.0},
            {"name": "B", "pIC50": 6.0, "ic50_nM": 1000.0},
            {"name": "C", "pIC50": 7.0, "ic50_nM": 100.0},
        ]
    ).to_csv(tmp_project / "workspace" / "inputs" / "curated_pIC50.csv", index=False)

    monkeypatch.chdir(tmp_project)
    monkeypatch.setattr("sys.argv", ["build_qsar.py", "--config", str(cfg)])
    build_qsar.main()

    pdf = pd.read_csv(tmp_project / "workspace" / "outputs" / "predictions.csv")
    assert len(pdf) == 3
    assert pdf["prediction_source"].eq("fallback_ranking").all()
