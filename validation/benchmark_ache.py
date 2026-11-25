#!/usr/bin/env python3
"""
Placeholder AChE benchmark: compare illustrative experimental pIC50 vs synthetic ``predicted'' scores.

Replace ``predicted_pIC50`` generation with outputs from ``scripts/dock_ligands.py`` /
``scripts/build_qsar.py`` for publication-grade validation.

Examples
--------
Run from the repository root::

    pip install -r validation/requirements_validation.txt
    python validation/benchmark_ache.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

VAL_DIR = Path(__file__).resolve().parent


def load_benchmark_table(csv_path: Path) -> pd.DataFrame:
    """
    Load benchmark compounds with experimental pIC50.

    Parameters
    ----------
    csv_path : pathlib.Path
        CSV with columns ``name``, ``pIC50``, optional ``smiles``.

    Returns
    -------
    pandas.DataFrame
        Parsed table sorted by name.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    """
    df = pd.read_csv(csv_path)
    if "pIC50" not in df.columns:
        raise ValueError("benchmark CSV must contain pIC50")
    return df.sort_values("name").reset_index(drop=True)


def synthetic_predicted_pic50(df: pd.DataFrame, noise: float = 0.35, seed: int = 0) -> np.ndarray:
    """
    Generate placeholder predictions correlated with experimental pIC50 (for CI / demos).

    Parameters
    ----------
    df : pandas.DataFrame
        Must include ``pIC50``.
    noise : float, optional
        Gaussian noise standard deviation added to a affine mapping.
    seed : int, optional
        RNG seed.

    Returns
    -------
    numpy.ndarray
        Placeholder predicted pIC50 values.
    """
    rng = np.random.default_rng(seed)
    y = df["pIC50"].astype(float).values
    return 0.85 * y + 1.1 + rng.normal(0.0, noise, size=len(y))


def metrics_binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, cutoff: float = 7.5) -> float:
    """
    Threshold classification accuracy (active if pIC50 >= cutoff).

    Parameters
    ----------
    y_true : numpy.ndarray
        Experimental pIC50.
    y_pred : numpy.ndarray
        Predicted pIC50.
    cutoff : float, optional
        Activity cutoff on the pIC50 scale.

    Returns
    -------
    float
        Fraction of compounds with matching active/inactive labels.
    """
    t_act = y_true >= cutoff
    p_act = y_pred >= cutoff
    return float((t_act == p_act).mean())


def main() -> None:
    csv_path = VAL_DIR / "benchmark_compounds.csv"
    df = load_benchmark_table(csv_path)
    y_exp = df["pIC50"].astype(float).values
    y_pred = synthetic_predicted_pic50(df)

    rmse = float(np.sqrt(np.mean((y_pred - y_exp) ** 2)))
    r, _ = pearsonr(y_exp, y_pred)
    acc = metrics_binary_accuracy(y_exp, y_pred, cutoff=7.5)

    out_metrics = VAL_DIR / "benchmark_metrics.json"
    payload = {
        "pearson_r": float(r),
        "rmse_pic50": rmse,
        "accuracy_threshold_7_5": acc,
        "n": int(len(df)),
    }
    out_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_exp, y_pred, c="#2980b9", edgecolors="k", linewidths=0.4)
    lo = float(min(y_exp.min(), y_pred.min())) - 0.2
    hi = float(max(y_exp.max(), y_pred.max())) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
    ax.set_xlabel("Experimental pIC50")
    ax.set_ylabel("Predicted pIC50 (placeholder)")
    ax.set_title("Parity (placeholder predictions)")
    fig.tight_layout()
    fig.savefig(VAL_DIR / "parity.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6.5, 3.8))
    err = y_pred - y_exp
    ax2.bar(df["name"].astype(str), err, color="#8e44ad", alpha=0.85)
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_ylabel("Error (pred − exp) / pIC50")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    fig2.tight_layout()
    fig2.savefig(VAL_DIR / "errors.png", dpi=150)
    plt.close(fig2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {VAL_DIR / 'parity.png'}, {VAL_DIR / 'errors.png'}, {out_metrics}")


if __name__ == "__main__":
    main()
