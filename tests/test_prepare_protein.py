"""Tests for receptor preparation (BioPython path; obabel subprocess mocked)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def test_prepare_receptor_writes_outputs(tmp_path: Path, tiny_pdb_text: str) -> None:
    from prepare_protein import prepare_receptor_for_vina

    raw = tmp_path / "in.pdb"
    raw.write_text(tiny_pdb_text, encoding="utf-8")
    out_clean = tmp_path / "clean.pdb"
    out_pdbqt = tmp_path / "rec.pdbqt"

    def fake_run(
        cmd, check=True, capture_output=True, text=True, **_kw
    ):  # noqa: ARG001
        assert "obabel" in cmd[0]
        out_idx = cmd.index("-O") + 1
        Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[out_idx]).write_text("ROOT\nTORSDOF 0\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("prepare_protein.subprocess.run", side_effect=fake_run):
        prepare_receptor_for_vina(raw, out_clean, out_pdbqt)

    assert out_clean.is_file()
    assert out_pdbqt.is_file()


def test_missing_raw_raises(tmp_path: Path) -> None:
    from prepare_protein import write_clean_protein_pdb

    missing = tmp_path / "nope.pdb"
    with pytest.raises(FileNotFoundError):
        write_clean_protein_pdb(missing, tmp_path / "c.pdb")


def test_main_system_exit_missing_receptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prepare_protein

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "protein:\n  pdb_file: workspace/inputs/missing.pdb\n"
        "paths:\n  receptor_clean: workspace/outputs/rc.pdb\n"
        "  receptor_pdbqt: workspace/outputs/r.pdbqt\n",
        encoding="utf-8",
    )
    (tmp_path / "workspace" / "inputs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _: "/fake/obabel")

    monkeypatch.setattr(
        "sys.argv",
        ["prepare_protein.py", "--config", str(cfg)],
    )
    with pytest.raises(SystemExit) as ei:
        prepare_protein.main()
    assert "not found" in str(ei.value).lower()
