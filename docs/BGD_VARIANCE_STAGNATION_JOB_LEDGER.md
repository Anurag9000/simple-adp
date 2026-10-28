# BGD-shadow variance stagnation: matched ADP scientific jobs

Scope: canonical and generic adaptive architecture routes in the native
`SuitePlan.build` runner, across every actually materialized compatible
dataset provider, task, split, fold, model seed, model adapter and repeat.
`MLP-Fixed` and screening controls are deliberately not duplicated as
architecture-expansion algorithms.

For each original ADP `RunSpec`, the suite planner creates exactly one
alternative with:

- Same original route and architecture algorithm; same dataset task,
  preprocessing split, seed, loss, optimizer, training epochs, architecture
  proposal style, validation-based candidate selection and rollback.
- Separate immutable `RunIdentity.model_route`, `run_id`, result directory,
  worker spec, checkpoint and logs.
- `worker_settings.expansion_criterion="variance_stagnation"` and
  `worker_settings.counterpart_of` containing the original `run_id`.

The new `ShadowBGDVariance` estimates the diagonal BGD posterior standard
deviation using Monte Carlo gradient-noise correlation and the positive
quadratic root from Zeno et al. The trainable mean weights remain under their
original Adam/AdamW optimizer. The shadow posterior does not update weights,
gradients or model buffers, and is checkpointed with its own RNG and sigma
state. It is **not** exact online Bayesian inference.

After each fitted architecture stage, a new growth proposal is permitted
only if BGD shadow sigma variance contraction has stalled persistently.
The original validation loss still determines which candidate is retained.
The variance observer is initialized freshly for each candidate stage,
matching the original per-stage optimizer reset. Other training and
capacity stopping conditions remain unchanged.

Inspect a real materialized plan and verify matching `counterpart_of`
fields before running experiments:

```bash
python -m simple_adp.runner.launcher --dataset-suites tabzilla,ctr23 --dry-run --output benchmark-results/coordinated-suite
pytest tests/test_shadow_variance_growth_counterparts.py -q
```

Caveats: source-backed automatic pairing applies only to native
`SuitePlan.build` jobs. External hand-authored workers that bypass that
planner must supply the new SearchSettings flag explicitly.
No claims are made for fabricated data combinations, missing datasets,
completed training runs, or validation of every custom adapter.
The variance-stagnation threshold is a research heuristic, not a proved
ADP expansion rule from the BGD/FOO-VB papers.
