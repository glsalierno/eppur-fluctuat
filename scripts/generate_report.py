#!/usr/bin/env python3
"""HTML report + matplotlib figures for eppur-fluctuat QSAR outputs."""

from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
import tempfile
from glob import glob
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(cfg_path: Path) -> dict:
    """
    Load ``config.yaml`` for reporting paths and titles.

    Parameters
    ----------
    cfg_path : pathlib.Path
        Configuration file path.

    Returns
    -------
    dict
        Parsed YAML tree.
    """
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _glob_first(pattern: str, search_dir: Path) -> Optional[Path]:
    """
    Return the first path matching ``pattern`` under ``search_dir``.

    Parameters
    ----------
    pattern : str
        Glob pattern (may include ``*``).
    search_dir : pathlib.Path
        Directory to search.

    Returns
    -------
    pathlib.Path or None
        First sorted match, if any.
    """
    paths = sorted(glob(str(search_dir / pattern)))
    if not paths:
        return None
    return Path(paths[0])


def pdbqt_to_pdb_text(pdbqt: Path) -> str:
    """
    Convert PDBQT to PDB text using Open Babel (temporary file).

    Parameters
    ----------
    pdbqt : pathlib.Path
        Ligand PDBQT path.

    Returns
    -------
    str
        PDB file contents.

    Raises
    ------
    subprocess.CalledProcessError
        If ``obabel`` fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as t:
        tmp = Path(t.name)
    try:
        subprocess.run(
            ["obabel", str(pdbqt), "-O", str(tmp)],
            check=True,
            capture_output=True,
            text=True,
        )
        return tmp.read_text(encoding="utf-8", errors="replace")
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    """
    Build matplotlib figures and a standalone HTML QSAR report.

    Raises
    ------
    FileNotFoundError
        If required CSV/JSON inputs are missing (depends on workflow state).
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
    proj = cfg.get("project", {})
    paths = cfg.get("paths", {})
    report_title = str(proj.get("report_title", "eppur-fluctuat report"))
    project_name = str(proj.get("name", "eppur-fluctuat"))

    pred_path = base / paths.get("predictions", "workspace/outputs/predictions.csv")
    metrics_path = base / paths.get("model_dir", "workspace/outputs") / "metrics.json"
    plot_data_path = (
        base / paths.get("model_dir", "workspace/outputs") / "plot_data.json"
    )
    out_html = base / paths.get("report_html", "workspace/outputs/qsar_report.html")
    plot_scatter = base / paths.get(
        "plot_scatter", "workspace/outputs/model_performance.png"
    )
    plot_imp = base / paths.get(
        "plot_importance", "workspace/outputs/feature_importance.png"
    )
    plot_be_path = base / paths.get(
        "plot_be_vs_pic50", "workspace/outputs/be_vs_pic50.png"
    )
    pose_glob = paths.get("pose_glob", "workspace/outputs/docked/*_best.pdbqt")

    df = pd.read_csv(pred_path)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    plot_payload: dict = {}
    if plot_data_path.is_file():
        with open(plot_data_path, encoding="utf-8") as f:
            plot_payload = json.load(f)

    rank_label = str(plot_payload.get("rank_column") or "binding_energy_kcal_mol")

    loo_rows = plot_payload.get("loo") if plot_payload else None
    if (
        not loo_rows
        and (
            base / paths.get("loo_table", "workspace/outputs/loo_predictions.csv")
        ).is_file()
    ):
        loo_df = pd.read_csv(
            base / paths.get("loo_table", "workspace/outputs/loo_predictions.csv")
        )
        loo_rows = loo_df.to_dict("records")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if loo_rows:
        ye = [float(r["pIC50_exp"]) for r in loo_rows]
        yp = [float(r["pIC50_loo_pred"]) for r in loo_rows]
        ax.scatter(
            ye,
            yp,
            s=52,
            c="#2980b9",
            edgecolors="k",
            linewidths=0.4,
            label="LOO train",
            zorder=3,
        )
        lo = float(min(min(ye), min(yp)))
        hi = float(max(max(ye), max(yp)))
        pad = max(0.15, (hi - lo) * 0.05)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            "k--",
            lw=1,
            alpha=0.6,
            label="y = x",
        )
        for r in loo_rows:
            ax.annotate(
                str(r["name"])[:14],
                (float(r["pIC50_exp"]), float(r["pIC50_loo_pred"])),
                fontsize=6,
                alpha=0.85,
            )
        tp = metrics.get("test_predictions") or []
        if tp:
            te = [float(x["pIC50_exp"]) for x in tp]
            tpr = [float(x["pIC50_pred"]) for x in tp]
            ax.scatter(
                te,
                tpr,
                s=72,
                marker="s",
                facecolors="none",
                edgecolors="#c0392b",
                linewidths=1.4,
                label="Held-out test",
                zorder=4,
            )
        ax.set_xlabel("Experimental pIC50")
        ax.set_ylabel("Predicted pIC50 (leave-one-out)")
        ax.legend(loc="lower right", fontsize=7)
    else:
        sub = df.dropna(subset=["pIC50_exp", "pIC50_pred"])
        if len(sub):
            ax.scatter(
                sub["pIC50_exp"],
                sub["pIC50_pred"],
                s=45,
                c="#2980b9",
                edgecolors="k",
                linewidths=0.4,
            )
            lo = float(min(sub["pIC50_exp"].min(), sub["pIC50_pred"].min()))
            hi = float(max(sub["pIC50_exp"].max(), sub["pIC50_pred"].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
            ax.set_xlabel("Experimental pIC50")
            ax.set_ylabel("Predicted pIC50")
        else:
            ax.text(
                0.5,
                0.5,
                "No LOO table / experimental overlap",
                ha="center",
                transform=ax.transAxes,
            )
    ax.set_title("Experimental vs LOO-predicted pIC50")
    plot_scatter.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_scatter, dpi=150)
    plt.close(fig)

    plot_be: Optional[Path] = None
    if plot_payload.get("line") and not plot_payload.get("fallback"):
        xbe = np.asarray(plot_payload.get("train_binding_energy") or [], dtype=float)
        ypi = np.asarray(plot_payload.get("train_pic50") or [], dtype=float)
        line = plot_payload["line"]
        xg = np.asarray(line["x_grid"], dtype=float)
        y_lo = np.asarray(line["y_low"], dtype=float)
        y_hi = np.asarray(line["y_high"], dtype=float)
        y_mid = 0.5 * (y_lo + y_hi)
        figb, axb = plt.subplots(figsize=(5.5, 4.2))
        axb.scatter(
            xbe, ypi, s=52, c="#2980b9", edgecolors="k", linewidths=0.4, zorder=3
        )
        axb.fill_between(
            xg, y_lo, y_hi, color="#3498db", alpha=0.25, label="OLS mean 95% CI"
        )
        axb.plot(xg, y_mid, "-", color="#2980b9", lw=1.4, label="OLS fit")
        axb.set_xlabel(f"Descriptor ({rank_label})")
        axb.set_ylabel("Experimental pIC50 (training)")
        axb.legend(loc="best", fontsize=8)
        axb.set_title("Descriptor vs experimental pIC50")
        figb.tight_layout()
        plot_be_path.parent.mkdir(parents=True, exist_ok=True)
        figb.savefig(plot_be_path, dpi=150)
        plt.close(figb)
        plot_be = plot_be_path

    imp = metrics.get("feature_importance") or {}
    coefs = metrics.get("ols_coefficients") or {}
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if coefs:
        items = list(coefs.items())
        names = [x[0] for x in items]
        vals = [x[1] for x in items]
        ax2.barh(names[::-1], vals[::-1], color="#27ae60")
        ax2.set_title("OLS coefficients (fitted model)")
        ax2.axvline(0, color="#888", lw=0.8)
    elif imp:
        items = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:16]
        names = [x[0] for x in items]
        vals = [x[1] for x in items]
        ax2.barh(names[::-1], vals[::-1], color="#27ae60")
        ax2.set_title(f"Feature importance ({metrics.get('selected_model', '')})")
    else:
        ax2.text(
            0.5,
            0.5,
            "No coefficients / importance (fallback ranking or split model)",
            ha="center",
        )
    fig2.tight_layout()
    fig2.savefig(plot_imp, dpi=150)
    plt.close(fig2)

    pred_ok = df["pIC50_pred"].notna()
    best = (
        df.loc[df.loc[pred_ok, "pIC50_pred"].idxmax()] if pred_ok.any() else df.iloc[0]
    )

    pose_path = _glob_first(pose_glob, base)
    viewer_html = (
        '<p class="note">No docked pose found for py3Dmol. Set paths.pose_glob in config.yaml '
        f"(default: {html.escape(pose_glob)}).</p>"
    )
    if pose_path and pose_path.is_file():
        try:
            import py3Dmol

            lig_txt = pdbqt_to_pdb_text(pose_path)
            v = py3Dmol.view(width=800, height=420)
            v.addModel(lig_txt, "pdb")
            v.setStyle({"model": 0}, {"stick": {"radius": 0.18}})
            v.zoomTo()
            viewer_html = (
                f'<p class="note">Example pose (glob match): {pose_path.name}. '
                f"Highest predicted pIC50 in table: <b>{best['name']}</b>.</p>"
                + v.write_html()
            )
        except Exception as exc:
            viewer_html = f'<p class="note">py3Dmol viewer skipped: {exc}</p>'

    scatter_b64 = base64.b64encode(plot_scatter.read_bytes()).decode("ascii")
    imp_b64 = base64.b64encode(plot_imp.read_bytes()).decode("ascii")
    be_b64 = ""
    if plot_be is not None and plot_be.is_file():
        be_b64 = base64.b64encode(plot_be.read_bytes()).decode("ascii")

    def _fmt(v, nd=None):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        try:
            x = float(v)
        except (TypeError, ValueError):
            return html.escape(str(v))
        if nd is not None:
            return f"{x:.{nd}f}"
        return f"{x:.4g}"

    rows = []
    for _, r in df.iterrows():
        be = r.get("binding_energy_kcal_mol")
        ice = r.get("ic50_nM_exp")
        ipp = r.get("IC50_nM_pred")
        ppp = r.get("pIC50_pred")
        ppe = r.get("pIC50_exp")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('name', '')))}</td>"
            f"<td>{html.escape(str(r.get('type', '')))}</td>"
            f"<td>{_fmt(be, 2) if pd.notna(be) else '—'}</td>"
            f"<td>{int(r.get('covalent_factor', 0)) if pd.notna(r.get('covalent_factor')) else '—'}</td>"
            f"<td>{_fmt(ice) if pd.notna(ice) else '—'}</td>"
            f"<td>{_fmt(ipp)}</td>"
            f"<td>{_fmt(ppp, 3)}</td>"
            f"<td>{_fmt(ppe, 3) if pd.notna(ppe) else '—'}</td>"
            "</tr>"
        )

    sel = metrics.get("selected_model", "")
    cv_block = (
        f"<tr><td>LOO (pooled)</td><td>{_fmt(metrics.get('loo_r2'), 4)}</td>"
        f"<td>{_fmt(metrics.get('loo_rmse'), 4)}</td></tr>"
    )
    if metrics.get("test_rmse") is not None:
        cv_block += (
            f"<tr><td>Held-out test</td><td>{_fmt(metrics.get('test_r2'), 4)}</td>"
            f"<td>{_fmt(metrics.get('test_rmse'), 4)}</td></tr>"
        )
    for k in ("mlr", "ridge", "pls", "random_forest"):
        if k in metrics and isinstance(metrics[k], dict):
            cv_block += f"<tr><td>{k} (legacy)</td><td>{metrics[k].get('cv_r2', '—')}</td><td>{metrics[k].get('cv_rmse', '—')}</td></tr>"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(report_title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; background: #f4f6fb; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; background: #fff; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 0.88rem; }}
    th {{ background: #2c3e50; color: #fff; }}
    .note {{ color: #555; font-size: 0.9rem; }}
    img {{ max-width: 100%; height: auto; }}
    section {{ margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>{html.escape(project_name)} — QSAR report</h1>
  <p class="note">Model: <b>{html.escape(str(sel))}</b>. Metrics use <b>leave-one-out (LOO)</b> on training compounds where applicable.
  With very few experimental points, R² can be <b>negative or unstable</b>; use ranking fallback when triggered.</p>

  <section>
    <h2>Validation metrics</h2>
    <table><tr><th>Split</th><th>R²</th><th>RMSE (pIC50)</th></tr>{cv_block}</table>
  </section>

  <section>
    <h2>Experimental vs LOO-predicted (pIC50 space)</h2>
    <img src="data:image/png;base64,{scatter_b64}" alt="scatter"/>
  </section>

  <section>
    <h2>Descriptor vs experimental pIC50</h2>
    {f'<img src="data:image/png;base64,{be_b64}" alt="descriptor vs pic50"/>' if be_b64 else '<p class="note">No OLS CI plot (fallback mode or &lt;3 training points).</p>'}
  </section>

  <section>
    <h2>Coefficients / feature importance</h2>
    <img src="data:image/png;base64,{imp_b64}" alt="importance"/>
  </section>

  <section>
    <h2>All compounds</h2>
    <table>
      <tr><th>Name</th><th>Type</th><th>BE (kcal/mol)</th><th>Covalent (1/0)</th><th>IC50 exp (nM)</th><th>IC50 pred (nM)</th><th>pIC50 pred</th><th>pIC50 exp</th></tr>
      {"".join(rows)}
    </table>
  </section>

  <section>
    <h2>Docked pose viewer (example)</h2>
    {viewer_html}
  </section>
</body>
</html>
"""
    out_html.write_text(page, encoding="utf-8")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
