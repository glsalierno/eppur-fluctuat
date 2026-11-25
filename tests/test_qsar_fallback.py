"""QSAR ranking fallback and docking-failure style inputs (fully mocked docking file merge)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_should_use_ranking_fallback_logic() -> None:
    from build_qsar import should_use_ranking_fallback

    assert should_use_ranking_fallback(0.01, fallback_enabled=True, fallback_threshold=0.2) is True
    assert should_use_ranking_fallback(0.5, fallback_enabled=True, fallback_threshold=0.2) is False
    assert should_use_ranking_fallback(None, fallback_enabled=True, fallback_threshold=0.2) is False
    assert should_use_ranking_fallback(0.01, fallback_enabled=False, fallback_threshold=0.2) is False


def test_predictions_same_schema_fallback_vs_ols(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback and OLS branches must emit the same prediction columns."""
    import build_qsar

    cfg_hi = tmp_project / "c_hi.yaml"
    cfg_hi.write_text(
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
  predictions: workspace/outputs/predictions_hi.csv
  model_dir: workspace/outputs
  loo_table: workspace/outputs/loo_hi.csv
""",
        encoding="utf-8",
    )

    cfg_lo = tmp_project / "c_lo.yaml"
    cfg_lo.write_text(
        """
qsar:
  fallback_to_ranking: false
  fallback_loo_r2_threshold: 0.0
  use_descriptors: [binding_energy]
  min_train_samples: 3
paths:
  ligands: workspace/inputs/ligands.csv
  docking_results: workspace/outputs/docking_results.csv
  computed_features: workspace/outputs/computed_features.csv
  curated_pIC50: workspace/inputs/curated_pIC50.csv
  predictions: workspace/outputs/predictions_lo.csv
  model_dir: workspace/outputs
  loo_table: workspace/outputs/loo_lo.csv
""",
        encoding="utf-8",
    )

    lig = tmp_project / "workspace" / "inputs" / "ligands.csv"
    dock = tmp_project / "workspace" / "outputs" / "docking_results.csv"
    pic = tmp_project / "workspace" / "inputs" / "curated_pIC50.csv"
    feat = tmp_project / "workspace" / "outputs" / "computed_features.csv"

    pd.DataFrame(
        [
            {"name": "A", "smiles": "CC"},
            {"name": "B", "smiles": "CCC"},
            {"name": "C", "smiles": "CCCC"},
        ]
    ).to_csv(lig, index=False)

    # Degenerate docking scores (no spread): mimics unreliable poses without NaN median issues.
    pd.DataFrame(
        [
            {"name": "A", "affinity_kcal_mol": -7.0},
            {"name": "B", "affinity_kcal_mol": -7.0},
            {"name": "C", "affinity_kcal_mol": -7.0},
        ]
    ).to_csv(dock, index=False)

    pd.DataFrame(
        [
            {"name": "A", "pIC50": 5.0, "ic50_nM": 10000.0},
            {"name": "B", "pIC50": 6.0, "ic50_nM": 1000.0},
            {"name": "C", "pIC50": 7.0, "ic50_nM": 100.0},
        ]
    ).to_csv(pic, index=False)

    if feat.is_file():
        feat.unlink()

    monkeypatch.chdir(tmp_project)

    monkeypatch.setattr("sys.argv", ["build_qsar.py", "--config", str(cfg_hi)])
    build_qsar.main()
    df_hi = pd.read_csv(tmp_project / "workspace" / "outputs" / "predictions_hi.csv")

    monkeypatch.setattr("sys.argv", ["build_qsar.py", "--config", str(cfg_lo)])
    build_qsar.main()
    df_lo = pd.read_csv(tmp_project / "workspace" / "outputs" / "predictions_lo.csv")

    assert list(df_hi.columns) == list(df_lo.columns)
    assert (df_hi["prediction_source"] == "fallback_ranking").all()
    assert not (df_lo["prediction_source"] == "fallback_ranking").all()
