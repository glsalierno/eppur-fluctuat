"""Pytest fixtures for eppur-fluctuat (mocked externals; no Vina/RDKit/obabel runs)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def repo_root() -> Path:
    """Repository root path."""
    return ROOT


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Temporary project directory with ``workspace/inputs`` and ``workspace/outputs`` layout.
    """
    (tmp_path / "workspace" / "inputs").mkdir(parents=True)
    (tmp_path / "workspace" / "outputs").mkdir(parents=True)
    (tmp_path / "var" / "logs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def ache_example_dir(repo_root: Path) -> Path:
    """Path to bundled AChE QSAR example data (ligands + curated pIC50)."""
    p = repo_root / "examples" / "ache_demo" / "AChE_QSAR_Pipeline" / "data"
    if not p.is_dir():
        pytest.skip("AChE example data directory missing")
    return p


@pytest.fixture
def mock_vina_engine() -> MagicMock:
    """A fake Vina-like object with deterministic energies and pose write."""
    m = MagicMock()
    m.energies.return_value = [[[-7.5, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    return m


@pytest.fixture
def tiny_pdb_text() -> str:
    """Minimal valid ATOM records for BioPython (single ALA in chain A)."""
    return (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       2.070  -0.786   1.234  1.00  0.00           C\n"
        "END\n"
    )
