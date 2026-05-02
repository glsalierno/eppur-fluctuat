# Validation benchmark (placeholder)

This folder holds a **small, illustrative** acetylcholinesterase (AChE) style table (`benchmark_compounds.csv`) with **example** literature-scale potencies (rounded / aggregated values for demonstration — **not** a curated primary-data reproducibility study).

## What `benchmark_ache.py` does

1. Loads `benchmark_compounds.csv`.
2. Builds **placeholder** predicted pIC50 values (`synthetic_predicted_pic50`) correlated with experiment plus noise — stand-in until you wire in real `docking_results.csv` / `predictions.csv` from the pipeline.
3. Computes **Pearson r**, **RMSE** (pIC50 units), and **classification accuracy** at pIC50 ≥ 7.5.
4. Writes `parity.png`, `errors.png`, and `benchmark_metrics.json`.

Run:

```bash
pip install -r validation/requirements_validation.txt
python validation/benchmark_ache.py
```

## Latest placeholder metrics

After running the script, open `benchmark_metrics.json`. Example shape (your RNG may differ slightly):

| Metric | Example value | Meaning |
|--------|---------------|---------|
| `pearson_r` | ~0.96 | Linear correlation experiment vs placeholder prediction |
| `rmse_pic50` | ~0.22 | RMSE in pIC50 units |
| `accuracy_threshold_7_5` | ~0.9 | Fraction correct active/inactive at cutoff 7.5 |
| `n` | 10 | Compound count |

## Limitations

- **Not** an independent benchmark suite against ChEMBL/BindingDB extracts (yet).
- **Placeholder predictions** — replace with docking affinity → calibrated model outputs for science.
- **Small n = unstable correlation** — interpret uncertainty honestly.

When you publish, replace rows with **primary citations**, uncertainty, and **frozen pipeline versions** (Git tag + Zenodo DOI).
