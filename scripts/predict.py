#!/usr/bin/env python3
"""Re-run predictions from a saved model.joblib (same feature columns as build_qsar.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


def load_config(cfg_path: Path) -> dict:
    """
    Load YAML configuration from disk.

    Parameters
    ----------
    cfg_path : pathlib.Path
        Path to ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_name(s: str) -> str:
    """
    Normalize compound names for stable merges (strip and collapse whitespace).

    Parameters
    ----------
    s : str
        Raw name string.

    Returns
    -------
    str
        Normalized label.
    """
    return " ".join(str(s).strip().split())


def ic50_nm_from_pic50(pic50: float) -> float:
    """
    Convert pIC50 (molar −log10) to IC50 in nM (approximate using fixed convention).

    Parameters
    ----------
    pic50 : float
        Negative log10 molar concentration.

    Returns
    -------
    float
        IC50 in nanomolar units.
    """
    return 10.0 ** (9.0 - pic50)


def main() -> None:
    """
    Recompute ``predictions.csv`` using the saved ``model.joblib`` pack.

    Raises
    ------
    FileNotFoundError
        If ``computed_features.csv`` or ``model.joblib`` is missing.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = args.config.resolve().parent
    paths = cfg.get("paths", {})
    feat_path = base / paths.get("computed_features", "workspace/outputs/computed_features.csv")
    pic_path = base / paths.get("curated_pIC50", "workspace/inputs/curated_pIC50.csv")
    out_dir = base / paths.get("model_dir", "workspace/outputs")
    pack = joblib.load(out_dir / "model.joblib")
    cols: list[str] = pack["feature_columns"]
    mode = pack.get("mode", "single")
    rank_col = str(pack.get("rank_column") or "binding_energy_kcal_mol")
    covalent_names = {norm_name(x) for x in (pack.get("covalent_names") or [])}

    merged = pd.read_csv(feat_path)
    merged["name"] = merged["name"].map(norm_name)
    if pic_path.is_file():
        exp = pd.read_csv(pic_path)
        exp["name"] = exp["name"].astype(str).map(norm_name)
        pic_map = exp.set_index("name")["pIC50"].to_dict()
        ic50_map = exp.set_index("name")["ic50_nM"].to_dict() if "ic50_nM" in exp.columns else {}
        merged["pIC50_exp"] = merged["name"].map(pic_map)
        merged["ic50_nM_exp"] = merged["name"].map(ic50_map) if ic50_map else np.nan
        merged["has_experimental"] = merged["pIC50_exp"].notna()
    else:
        merged["pIC50_exp"] = np.nan
        merged["ic50_nM_exp"] = np.nan
        merged["has_experimental"] = False

    for c in cols:
        merged[c] = merged[c].replace([np.inf, -np.inf], np.nan).fillna(merged[c].median())

    train_mask = merged["has_experimental"]
    y_mean = float(merged.loc[train_mask, "pIC50_exp"].mean()) if train_mask.any() else float("nan")

    merged["prediction_source"] = ""
    merged["pIC50_pred"] = np.nan
    merged["IC50_nM_pred"] = np.nan
    if rank_col in merged.columns:
        merged["rank_by_binding_energy"] = merged[rank_col].rank(ascending=True, method="min")
    else:
        merged["rank_by_binding_energy"] = np.nan

    if mode == "fallback":
        merged["prediction_source"] = "fallback_ranking"
    else:
        for idx, row in merged.iterrows():
            Xr = row[cols].values.astype(float).reshape(1, -1)
            nm = row["name"]
            if mode == "split":
                if nm in covalent_names and pack.get("covalent") and pack["covalent"].get("model"):
                    p = float(pack["covalent"]["model"].predict(Xr)[0])
                    src = "covalent_univariate"
                elif pack.get("non_covalent") and pack["non_covalent"].get("model"):
                    p = float(pack["non_covalent"]["model"].predict(Xr)[0])
                    src = "non_covalent_univariate"
                elif pack.get("global") and pack["global"].get("model"):
                    p = float(pack["global"]["model"].predict(Xr)[0])
                    src = "global_univariate"
                else:
                    p = y_mean
                    src = "mean_fallback"
            else:
                p = float(pack["global"]["model"].predict(Xr)[0])
                src = "global_univariate"
            merged.at[idx, "pIC50_pred"] = p
            merged.at[idx, "IC50_nM_pred"] = ic50_nm_from_pic50(p)
            merged.at[idx, "prediction_source"] = src

    pred_path = base / paths.get("predictions", "workspace/outputs/predictions.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
