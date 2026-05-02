"""Tests for Vina docking wrapper (mocked ``vina.Vina``)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_run_docking_returns_expected_dict(
    tmp_path: Path, mock_vina_engine: MagicMock
) -> None:
    from dock_ligands import run_docking

    out_pose = tmp_path / "best.pdbqt"
    result = run_docking(
        mock_vina_engine, "ligand.pdbqt", exhaustiveness=8, n_poses=9, out_pose=out_pose
    )

    assert result["affinity"] == -7.5
    assert isinstance(result["poses"], list)
    assert "pose_path" in result
    mock_vina_engine.set_ligand_from_file.assert_called_once_with("ligand.pdbqt")
    mock_vina_engine.dock.assert_called_once_with(exhaustiveness=8, n_poses=9)
    mock_vina_engine.write_pose.assert_called_once()


def test_main_passes_cpu_to_vina_constructor(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dock_ligands

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
  cpu: 4
  gpu: false
paths:
  ligands: workspace/inputs/ligands.csv
  receptor_clean: workspace/outputs/receptor_clean.pdb
  receptor_pdbqt: workspace/outputs/receptor.pdbqt
  docking_results: workspace/outputs/docking_results.csv
  docked_dir: workspace/outputs/docked
  logs_dir: var/logs
""",
        encoding="utf-8",
    )

    tiny = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
    (tmp_project / "workspace" / "inputs" / "protein.pdb").write_text(
        tiny, encoding="utf-8"
    )
    (tmp_project / "workspace" / "outputs" / "receptor_clean.pdb").write_text(
        tiny, encoding="utf-8"
    )
    (tmp_project / "workspace" / "outputs" / "receptor.pdbqt").write_text(
        "ROOT\nTORSDOF 0\n", encoding="utf-8"
    )

    pd.DataFrame([{"name": "L1", "smiles": "CC"}]).to_csv(
        tmp_project / "workspace" / "inputs" / "ligands.csv"
    )

    captured: dict = {}

    def ctor(**kwargs):
        captured.update(kwargs)
        inst = MagicMock()
        inst.energies.return_value = [[[-6.0, 0, 0, 0, 0, 0]]]
        return inst

    def fake_smiles(_smi: str, out_pdbqt: Path) -> None:
        out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        out_pdbqt.write_text("ROOT\nTORSDOF 0\n", encoding="utf-8")

    monkeypatch.setattr("shutil.which", lambda _: "/fake/obabel")
    monkeypatch.setattr(dock_ligands, "smiles_to_pdbqt", fake_smiles)

    with patch.object(dock_ligands, "Vina", side_effect=ctor):
        monkeypatch.setattr(
            "sys.argv",
            ["dock_ligands.py", "--config", str(cfg)],
        )
        dock_ligands.main()

    assert captured.get("cpu") == 4
    assert captured.get("seed") == 1


def test_no_gpu_import_side_effect() -> None:
    """Docking module must not require CUDA-only stacks for import."""
    import importlib

    import dock_ligands

    importlib.reload(dock_ligands)
    assert hasattr(dock_ligands, "run_docking")
