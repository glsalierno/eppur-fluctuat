# QSAR fallback mechanism

When experimental measurements are scarce, flexible ML models can **overfit**. This pipeline defaults to a **transparent baseline**:

1. **Descriptor** — typically docking **binding energy** (`binding_energy_kcal_mol`), optionally augmented with a simple **covalent proximity indicator** when columns exist.
2. **Model** — **univariate ordinary least squares (OLS)** relating the descriptor to experimental **pIC50**.
3. **Validation** — **leave-one-out (LOO)** cross-validation on the training set compounds that have experimental values.

## When ranking replaces regression

If `qsar.fallback_to_ranking` is **true** and the pooled LOO **R²** is **below** `qsar.fallback_loo_r2_threshold`, the pipeline **does not trust** the linear extrapolation. It sets `prediction_source` to `fallback_ranking` and ranks compounds by more favourable (more negative) docking scores via `rank_by_binding_energy`.

This is intentionally conservative: it prefers **honest ordering by the physics-based score** over **illusory numeric pIC50 predictions** from a weak linear fit.

## Limitations

- **Small n** — thresholds and split logic are heuristics, not guarantees of statistical power.
- **Single descriptor default** — multivariate extensions are optional and still linear.
- **Docking noise** — if docking scores are flat or unreliable, ranking fallback collapses to ties / arbitrary order among equals.

See `scripts/build_qsar.py` (`should_use_ranking_fallback` and `main`) for the exact decision logic.
