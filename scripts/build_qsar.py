#!/usr/bin/env python3
"""
Small-data QSAR: OLS on docking-derived descriptors, LOO validation, optional test holdout, ranking fallback.

Reads paths and model options from config.yaml (no target-specific hardcoding).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

# Config "logical" names → column names in computed_features.csv
DESCRIPTOR_ALIASES: Dict[str, str] = {
    "binding_energy": "binding_energy_kcal_mol",
    "binding_energy_kcal_mol": "binding_energy_kcal_mol",
    "covalent_factor": "covalent_factor",
}


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
        Parsed configuration tree.
    """
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_name(s: str) -> str:
    """
    Normalize compound names for merges (strip + collapse internal whitespace).

    Parameters
    ----------
    s : str
        Raw label.

    Returns
    -------
    str
        Normalized name string.
    """
    return " ".join(str(s).strip().split())


def ic50_nm_from_pic50(pic50: float) -> float:
    """
    Convert pIC50 to IC50 in nM using the pipeline convention.

    Parameters
    ----------
    pic50 : float
        Negative log10 molar activity.

    Returns
    -------
    float
        IC50 in nanomolar units.
    """
    return 10.0 ** (9.0 - pic50)


def resolve_descriptor_columns(use_descriptors: List[str]) -> List[str]:
    cols: List[str] = []
    for raw in use_descriptors or []:
        key = str(raw).strip()
        cols.append(DESCRIPTOR_ALIASES.get(key, key))
    return cols


def ensure_computed_features(cfg: dict, base: Path, paths: dict) -> Path:
    """
    Build ``computed_features.csv`` from docking + ligand tables if absent.

    Parameters
    ----------
    cfg : dict
        Full configuration (used only indirectly via paths).
    base : pathlib.Path
        Project root.
    paths : dict
        ``paths`` section mapping logical names to relative paths.

    Returns
    -------
    pathlib.Path
        Path to ``computed_features.csv``.

    Raises
    ------
    SystemExit
        If required inputs are missing or docking lacks ``affinity_kcal_mol``.
    """
    out_path = base / paths.get(
        "computed_features", "workspace/outputs/computed_features.csv"
    )
    if out_path.is_file():
        return out_path
    dock_path = base / paths.get(
        "docking_results", "workspace/outputs/docking_results.csv"
    )
    lig_path = base / paths.get("ligands", "workspace/inputs/ligands.csv")
    if not dock_path.is_file() or not lig_path.is_file():
        raise SystemExit(
            f"Missing computed features and could not build: need {dock_path} and {lig_path}. "
            "Run scripts/dock_ligands.py first."
        )
    dock = pd.read_csv(dock_path)
    lig = pd.read_csv(lig_path)
    lig["name"] = lig["name"].astype(str).map(norm_name)
    dock["name"] = dock["name"].astype(str).map(norm_name)
    dcols = ["name", "affinity_kcal_mol"]
    if "covalent_proximity_distance" in dock.columns:
        dcols.append("covalent_proximity_distance")
    dsub = dock[[c for c in dcols if c in dock.columns]].copy()
    m = lig.merge(dsub, on="name", how="left")
    if "affinity_kcal_mol" not in m.columns:
        raise SystemExit("docking_results.csv must contain affinity_kcal_mol")
    m["binding_energy_kcal_mol"] = m["affinity_kcal_mol"]
    if "covalent_proximity_distance" in m.columns:
        m["covalent_factor"] = (
            pd.to_numeric(m["covalent_proximity_distance"], errors="coerce") < 3.5
        ).astype(int)
    elif "covalent_factor" not in m.columns:
        m["covalent_factor"] = 0
    keep = ["name", "smiles", "binding_energy_kcal_mol", "covalent_factor"]
    for c in ("cas", "type"):
        if c in m.columns:
            keep.insert(2, c)
    m = m[[c for c in keep if c in m.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (merged from docking + ligands).", file=sys.stderr)
    return out_path


def loo_univariate(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    pred = np.zeros(n, dtype=float)
    if n < 2:
        return np.full(n, y.mean())
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        lr = LinearRegression()
        lr.fit(X[train_idx], y[train_idx])
        pred[test_idx] = lr.predict(X[test_idx])
    return pred


def fit_lr(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    lr = LinearRegression()
    lr.fit(X, y)
    return lr


def regression_ci_mean(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(x_train).ravel()
    y = np.asarray(y_train).ravel()
    n = len(x)
    if n < 3:
        xm, ym = float(x.mean()), float(y.mean())
        b = 0.0
        a = ym
        yhat = a + b * np.asarray(x_grid).ravel()
        return yhat, yhat, a, b
    xm = x.mean()
    ym = y.mean()
    Sxx = np.sum((x - xm) ** 2)
    if Sxx < 1e-18:
        yhat = np.full(len(x_grid), ym)
        return yhat, yhat, ym, 0.0
    b = float(np.sum((x - xm) * (y - ym)) / Sxx)
    a = float(ym - b * xm)
    yfit = a + b * x
    resid = y - yfit
    mse = float(np.sum(resid**2) / max(1, (n - 2)))
    t = float(stats.t.ppf(1 - alpha / 2, max(1, n - 2)))
    xg = np.asarray(x_grid).ravel()
    se = np.sqrt(mse * (1.0 / n + (xg - xm) ** 2 / Sxx))
    yhat = a + b * xg
    return yhat - t * se, yhat + t * se, a, b


def ranking_column_for_qsar(feat_cols: List[str]) -> str:
    for c in feat_cols:
        if "binding_energy" in c or c == "binding_energy_kcal_mol":
            return c
    return feat_cols[0]


def should_use_ranking_fallback(
    loo_r2: Optional[float],
    *,
    fallback_enabled: bool,
    fallback_threshold: float,
) -> bool:
    """
    Decide whether to use binding-energy ranking instead of OLS predictions.

    Mirrors the logic used after leave-one-out scoring in :func:`main`.

    Parameters
    ----------
    loo_r2 : float or None
        Pooled leave-one-out R² on the training set (may be NaN).
    fallback_enabled : bool
        ``qsar.fallback_to_ranking`` from configuration.
    fallback_threshold : float
        ``qsar.fallback_loo_r2_threshold``.

    Returns
    -------
    bool
        True if ranking fallback should replace OLS predictions.
    """
    if not fallback_enabled:
        return False
    if loo_r2 is None:
        return False
    try:
        if np.isnan(float(loo_r2)):
            return False
    except (TypeError, ValueError):
        return False
    return float(loo_r2) < float(fallback_threshold)


def main() -> None:
    """
    CLI entry: fit / validate small-data QSAR and write predictions plus metrics.

    Raises
    ------
    SystemExit
        On missing inputs, insufficient training compounds, or missing descriptor columns.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    qsar = cfg.get("qsar", {})
    base = args.config.resolve().parent
    paths = cfg.get("paths", {})

    feat_path = ensure_computed_features(cfg, base, paths)
    pic_path = base / paths.get("curated_pIC50", "workspace/inputs/curated_pIC50.csv")
    out_dir = base / paths.get("model_dir", "workspace/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = base / paths.get("predictions", "workspace/outputs/predictions.csv")
    metrics_path = out_dir / "metrics.json"
    loo_path = base / paths.get("loo_table", "workspace/outputs/loo_predictions.csv")

    feats = pd.read_csv(feat_path)
    exp = pd.read_csv(pic_path)
    exp = exp.dropna(subset=["pIC50", "name"])
    exp["name"] = exp["name"].map(norm_name)
    pic_map = exp.set_index("name")["pIC50"].to_dict()
    ic50_map = (
        exp.set_index("name")["ic50_nM"].to_dict() if "ic50_nM" in exp.columns else {}
    )

    merged = feats.copy()
    merged["name"] = merged["name"].map(norm_name)
    merged["pIC50_exp"] = merged["name"].map(pic_map)
    merged["ic50_nM_exp"] = merged["name"].map(ic50_map) if ic50_map else np.nan
    merged["has_experimental"] = merged["pIC50_exp"].notna()

    test_names: Set[str] = {
        norm_name(x) for x in (qsar.get("test_set_compounds") or [])
    }
    covalent_names: Set[str] = {
        norm_name(x) for x in (qsar.get("covalent_group") or [])
    }

    logical = list(qsar.get("use_descriptors") or ["binding_energy"])
    resolved = resolve_descriptor_columns(logical)

    n_exp = int(merged["has_experimental"].sum())
    if n_exp < int(qsar.get("min_train_samples", 3)):
        raise SystemExit(
            f"Need at least {qsar.get('min_train_samples', 3)} compounds with experimental pIC50."
        )

    train_df = merged[
        merged["has_experimental"] & ~merged["name"].isin(test_names)
    ].copy()
    test_df = merged[
        merged["has_experimental"] & merged["name"].isin(test_names)
    ].copy()
    train_df = train_df.reset_index(drop=True)

    if len(train_df) < 2 and len(test_df) > 0:
        print(
            "WARNING: Test holdout leaves fewer than 2 training points — consider fewer test compounds.",
            file=sys.stderr,
        )

    feat_cols: List[str] = []
    for c in resolved:
        if c not in merged.columns:
            raise SystemExit(
                f"Descriptor column '{c}' not in computed features ({feat_path})."
            )
        feat_cols.append(c)
    if not feat_cols:
        feat_cols = ["binding_energy_kcal_mol"]

    if (
        qsar.get("allow_second_descriptor", False)
        and int(qsar.get("second_descriptor_min_n", 10)) <= len(train_df)
        and "covalent_factor" in train_df.columns
        and "covalent_factor" not in feat_cols
    ):
        feat_cols.append("covalent_factor")

    rank_col = ranking_column_for_qsar(feat_cols)

    for c in feat_cols:
        merged[c] = (
            merged[c].replace([np.inf, -np.inf], np.nan).fillna(merged[c].median())
        )
        train_df[c] = (
            train_df[c].replace([np.inf, -np.inf], np.nan).fillna(train_df[c].median())
        )

    X_train = train_df[feat_cols].values.astype(float)
    y_train = train_df["pIC50_exp"].values.astype(float)
    names_train = train_df["name"].values

    cov_mask = train_df["name"].isin(covalent_names)
    cov_df = train_df[cov_mask].reset_index(drop=True)
    non_df = train_df[~cov_mask].reset_index(drop=True)
    split_ok = (
        bool(qsar.get("split_covalent_models", False))
        and len(covalent_names) > 0
        and len(cov_df) >= 2
        and len(non_df) >= 2
    )
    if bool(qsar.get("split_covalent_models", False)) and not split_ok:
        print(
            "NOTE: Covalent split disabled (need ≥2 covalent and ≥2 non-covalent training points). "
            "Using single model.",
            file=sys.stderr,
        )

    split_models = split_ok
    fallback_thr = float(qsar.get("fallback_loo_r2_threshold", 0.2))
    use_fallback_flag = bool(qsar.get("fallback_to_ranking", True))

    metrics: Dict[str, Any] = {
        "feature_columns": feat_cols,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_compounds": list(test_names),
        "split_covalent": split_models,
    }

    pack: Dict[str, Any] = {
        "feature_columns": feat_cols,
        "covalent_names": list(covalent_names),
        "rank_column": rank_col,
    }

    loo_rows: List[Dict[str, Any]] = []
    pooled_r2: float = float("nan")

    if split_models and len(train_df) >= 2:
        loo_by_name: Dict[str, float] = {}
        for sub_df, key in ((cov_df, "covalent"), (non_df, "non_covalent")):
            if len(sub_df) < 2:
                continue
            Xs = sub_df[feat_cols].values.astype(float)
            ys = sub_df["pIC50_exp"].values.astype(float)
            pl = loo_univariate(Xs, ys)
            for k in range(len(sub_df)):
                loo_by_name[str(sub_df.iloc[k]["name"])] = float(pl[k])
            lr_f = fit_lr(Xs, ys)
            pack[key] = {"model": lr_f, "n": len(sub_df)}
            metrics[f"{key}_loo_n"] = len(sub_df)
        ym = float(y_train.mean())
        pred_loo = np.array(
            [
                loo_by_name.get(str(train_df.iloc[j]["name"]), ym)
                for j in range(len(train_df))
            ]
        )
        y_loo_all = pred_loo.tolist()
        y_true_all = y_train.tolist()
        if len(y_true_all) >= 2:
            pooled_r2 = float(r2_score(y_true_all, y_loo_all))
        for i, nm in enumerate(names_train):
            loo_rows.append(
                {
                    "name": nm,
                    "pIC50_exp": float(y_true_all[i]),
                    "pIC50_loo_pred": float(y_loo_all[i]),
                    "group": "covalent" if nm in covalent_names else "non_covalent",
                }
            )
        metrics["loo_r2"] = pooled_r2
        metrics["loo_rmse"] = float(np.sqrt(mean_squared_error(y_true_all, y_loo_all)))

        pack["covalent"] = (
            {
                "model": fit_lr(
                    cov_df[feat_cols].values.astype(float), cov_df["pIC50_exp"].values
                ),
                "n": len(cov_df),
            }
            if len(cov_df) >= 2
            else None
        )
        pack["non_covalent"] = (
            {
                "model": fit_lr(
                    non_df[feat_cols].values.astype(float), non_df["pIC50_exp"].values
                ),
                "n": len(non_df),
            }
            if len(non_df) >= 2
            else None
        )
        pack["mode"] = "split"
        pred_fn = "split"

    else:
        pred_loo = loo_univariate(X_train, y_train)
        y_loo_all = pred_loo.tolist()
        y_true_all = y_train.tolist()
        if len(y_true_all) >= 2:
            pooled_r2 = float(r2_score(y_true_all, y_loo_all))
        metrics["loo_r2"] = pooled_r2
        metrics["loo_rmse"] = float(np.sqrt(mean_squared_error(y_true_all, y_loo_all)))
        for i, nm in enumerate(names_train):
            loo_rows.append(
                {
                    "name": nm,
                    "pIC50_exp": float(y_train[i]),
                    "pIC50_loo_pred": float(pred_loo[i]),
                    "group": "all",
                }
            )
        lr_full = fit_lr(X_train, y_train)
        pack["global"] = {"model": lr_full}
        pack["mode"] = "single"
        pred_fn = "single"

    pd.DataFrame(loo_rows).to_csv(loo_path, index=False)
    print("\n=== Leave-one-out (training set) ===\n")
    print(pd.DataFrame(loo_rows).to_string(index=False))
    lr = metrics.get("loo_r2", float("nan"))
    lm = metrics.get("loo_rmse", float("nan"))
    print(f"\nPooled LOO R² = {lr:.4f}  LOO RMSE = {lm:.4f}\n")

    metrics["test_r2"] = None
    metrics["test_rmse"] = None
    if len(test_df) > 0:
        X_te = test_df[feat_cols].values.astype(float)
        y_te = test_df["pIC50_exp"].values.astype(float)
        if pred_fn == "split" and pack.get("covalent") and pack.get("non_covalent"):
            y_hat = []
            for _, row in test_df.iterrows():
                nm = row["name"]
                Xr = row[feat_cols].values.astype(float).reshape(1, -1)
                if (
                    nm in covalent_names
                    and pack.get("covalent")
                    and pack["covalent"].get("model")
                ):
                    y_hat.append(float(pack["covalent"]["model"].predict(Xr)[0]))
                elif pack.get("non_covalent") and pack["non_covalent"].get("model"):
                    y_hat.append(float(pack["non_covalent"]["model"].predict(Xr)[0]))
                elif pack.get("global"):
                    y_hat.append(float(pack["global"]["model"].predict(Xr)[0]))
                else:
                    y_hat.append(float(y_train.mean()))
            y_hat = np.array(y_hat)
        else:
            y_hat = pack["global"]["model"].predict(X_te)
        metrics["test_r2"] = (
            float(r2_score(y_te, y_hat)) if len(y_te) >= 2 else float("nan")
        )
        metrics["test_rmse"] = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        metrics["test_predictions"] = [
            {
                "name": test_df.iloc[i]["name"],
                "pIC50_exp": float(y_te[i]),
                "pIC50_pred": float(y_hat[i]),
            }
            for i in range(len(test_df))
        ]
        print("=== Held-out test set ===\n")
        print(pd.DataFrame(metrics["test_predictions"]).to_string(index=False))
        print(f"\nTest R² = {metrics['test_r2']}  Test RMSE = {metrics['test_rmse']}\n")

    use_fallback = should_use_ranking_fallback(
        metrics.get("loo_r2"),
        fallback_enabled=use_fallback_flag,
        fallback_threshold=fallback_thr,
    )
    metrics["fallback_ranking"] = bool(use_fallback)
    metrics["fallback_reason"] = (
        f"LOO R² ({metrics.get('loo_r2')}) < threshold {fallback_thr}"
        if use_fallback
        else None
    )

    merged["prediction_source"] = ""
    merged["pIC50_pred"] = np.nan
    merged["IC50_nM_pred"] = np.nan
    merged["rank_by_binding_energy"] = merged[rank_col].rank(
        ascending=True, method="min"
    )

    if use_fallback:
        print(
            f"\n*** FALLBACK: LOO R² below {fallback_thr} — ranking by {rank_col} only.\n",
            file=sys.stderr,
        )
        pack["mode"] = "fallback"
        merged["prediction_source"] = "fallback_ranking"
        metrics["selected_model"] = f"none (ranking by {rank_col})"
    else:
        metrics["selected_model"] = "OLS (LOO validation)"
        if pack.get("mode") == "single" and pack.get("global"):
            lr_g = pack["global"]["model"]
            metrics["ols_coefficients"] = {
                c: float(lr_g.coef_[i]) for i, c in enumerate(feat_cols)
            }
            metrics["ols_intercept"] = float(lr_g.intercept_)
        for idx, row in merged.iterrows():
            Xr = row[feat_cols].values.astype(float).reshape(1, -1)
            nm = row["name"]
            if pred_fn == "split":
                if (
                    nm in covalent_names
                    and pack.get("covalent")
                    and pack["covalent"].get("model")
                ):
                    p = float(pack["covalent"]["model"].predict(Xr)[0])
                    src = "covalent_univariate"
                elif pack.get("non_covalent") and pack["non_covalent"].get("model"):
                    p = float(pack["non_covalent"]["model"].predict(Xr)[0])
                    src = "non_covalent_univariate"
                elif pack.get("global") and pack["global"].get("model"):
                    p = float(pack["global"]["model"].predict(Xr)[0])
                    src = "global_univariate"
                else:
                    p = float(y_train.mean())
                    src = "mean_fallback"
            else:
                p = float(pack["global"]["model"].predict(Xr)[0])
                src = "global_univariate"
            merged.at[idx, "pIC50_pred"] = p
            merged.at[idx, "IC50_nM_pred"] = ic50_nm_from_pic50(p)
            merged.at[idx, "prediction_source"] = src

    joblib.dump(pack, out_dir / "model.joblib")

    x_plot = merged[rank_col].values.astype(float)
    plot_payload = {
        "loo": loo_rows,
        "train_binding_energy": train_df[rank_col].tolist(),
        "train_pic50": train_df["pIC50_exp"].tolist(),
        "feat_cols": feat_cols,
        "rank_column": rank_col,
        "fallback": use_fallback,
    }
    if not use_fallback and len(train_df) >= 3:
        xg = np.linspace(float(np.nanmin(x_plot)), float(np.nanmax(x_plot)), 80)
        y_lo, y_hi, a, b = regression_ci_mean(
            train_df[rank_col].values,
            train_df["pIC50_exp"].values,
            xg,
        )
        plot_payload["line"] = {
            "a": a,
            "b": b,
            "x_grid": xg.tolist(),
            "y_low": y_lo.tolist(),
            "y_high": y_hi.tolist(),
        }
    with open(out_dir / "plot_data.json", "w", encoding="utf-8") as f:
        json.dump(plot_payload, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    merged.to_csv(pred_path, index=False)
    print(f"Saved → {pred_path}")
    print(f"Saved → {metrics_path}")
    print(f"Saved → {loo_path}")
    print(f"Saved → {out_dir / 'plot_data.json'}")


if __name__ == "__main__":
    main()
