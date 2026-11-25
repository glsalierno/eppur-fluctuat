"""Configuration parsing and defaults."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_repo_config_yaml_loads(repo_root: Path) -> None:
    cfg_path = repo_root / "config.yaml"
    assert cfg_path.is_file()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "protein" in cfg
    assert "docking" in cfg
    assert "qsar" in cfg
    assert "paths" in cfg
    assert cfg["protein"].get("chain") == "A"
    assert "workspace/" in str(cfg["paths"].get("ligands", ""))


def test_apply_implicit_defaults_for_minimal_config(tmp_path: Path) -> None:
    """Unknown keys fall back to script-internal defaults when sections omit optional fields."""
    from docking_common import load_config, resolve_center, vina_cpu_count

    p = tmp_path / "mini.yaml"
    p.write_text(
        "protein:\n  pdb_file: x.pdb\n  chain: A\n  pocket_center: [1,2,3]\n" "docking: {}\npaths: {}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    tiny = tmp_path / "clean.pdb"
    tiny.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    assert resolve_center(cfg, tiny) == (1.0, 2.0, 3.0)
    assert vina_cpu_count(cfg) >= 1
