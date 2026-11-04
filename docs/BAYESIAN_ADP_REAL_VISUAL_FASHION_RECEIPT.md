# Verified official Fashion-MNIST visual loss versus shadow-BGD worker pairs

**Evidence grade:** `BOUNDED_REAL_VISUAL_WORKER_PAIRS_PASSED`.

**Source revision:** `9c240c6ded31e8a12a4093647789159098dddbd5`.

**Successful workflow:** https://github.com/Anurag9000/simple-adp/actions/runs/41850723919

**Downloaded artifact:** GitHub Actions artifact `13334392587`;
`official-fashion-four-bayesian-worker-pairs-9c240c6ded31e8a12a4093647789159098dddbd5`.

**Manifest SHA-256:** `64201b5fce558aad590d64ff7820c188bd80b0543193bee8c6a5c176123244b4`.

## Data, method and execution scope

- Genuine, explicitly downloaded torchvision **Fashion-MNIST**, 10 classes,
  original official training/testing distribution.
- Exactly **40 train / 20 architecture-validation / 20 held-out test** images,
  one fixed partition, input model seed 73; matched *derived worker seed*
  `401840107` for every original/Bayesian pair.
- Existing signed `SuitePlan` and actual separate
  `python -m simple_adp.runner.worker --spec` executions.
- Four frozen-random source architecture families, each with native
  `ADP-W-RR`: V02 residual adapter, V03 tiny ViT, V04 tiny ConvNeXt,
  V07 token Mixer.
- Exactly eight successful worker runs, with separate source and shadow-BGD
  checkpoints, inferred model bundles, result JSON files and one paired
  completion receipt.
- Only the architecture expansion trigger changes within each pair. The
  trained source tensors were proven identical via per-pair frozen SHA-256.
- **No published pretrained source weights were used** in this experiment.

## Actual held-out measurements

| Family | Loss-growth selected width | Bayesian selected width | Loss accuracy | Bayesian accuracy | Loss macro F1 | Bayesian macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| V02 residual adapter | 3 | 2 | 0.1000 | 0.1000 | 0.01818 | 0.01818 |
| V03 tiny ViT | 3 | 2 | 0.1000 | 0.1000 | 0.01818 | 0.01818 |
| V04 tiny ConvNeXt | 3 | 2 | 0.1000 | 0.1000 | 0.01905 | 0.01905 |
| V07 token Mixer | 3 | 2 | 0.1000 | 0.1000 | 0.01818 | 0.01818 |

All loss-growth methods executed **2 candidate-history stages**.
All Bayesian alternatives executed **1 candidate-history stage**.
At these bounded settings the original accepted width 3 while the
Bayesian arm remained at initial width 2. In all eight cases, the
final held-out classification accuracy is **chance level**, 10%.
The results validate routing and data/seed consistency, **not**
effective learning, superiority, calibration or compute efficiency.

The official test rows were not used for model selection by the source
visual adapter; it evaluates those rows after fitting and candidate
selection. The script checks the real worker `result.json` metadata,
actual held-out metrics, matched source data/seed/split/frozen weights
and model and checkpoint SHA-256 hashes for each of the eight jobs.

## Open scientific and implementation gates

- Extend beyond four representative families and one width-growth method:
  registered V05–V11/V03/V04-source routes, other growth controllers,
  larger architecture capacity and realistic training budgets.
- Run actual published pretrained-source ViT and ConvNeXt original/BGD
  **matched pairs** (the pretrained one-arm smoke tests do not close this).
- Conduct multiple independent model seeds and fixed train/validation/test
  splits, additional official datasets and uncertainty evaluation.
- Verify actual posterior contraction trajectories and event-level growth
  decisions for visual stages, not just selected widths.
- Account for posterior Monte Carlo passes, full trained FLOPs,
  memory/energy and capacity-matched compute.
- Rerun and verify full CPU/GPU suites at the exact final commit.

This is a bounded *real-data source-to-worker reproducibility proof*,
not a completed visual Bayesian ADP scientific benchmark.


## Complete best-stage posterior gate trace, verified source `186f0054`

The follow-up real Fashion-MNIST four-architecture workflow
[41853011680](https://github.com/Anurag9000/simple-adp/actions/runs/41853011680)
completed with a **fully checkpointed posterior-gate trace** for all
four Bayesian stages, after the search-stage temporary posterior
objects were cleared. Each Bayesian variant had **one successful
posterior observation**, sampled with two Monte Carlo passes; every
required stagnation streak of three was unmet. The verified
best-epoch contraction/streak receipts are:

| Architecture | Contraction | Tolerance | Streak | Required | Gate ready |
|---|---:|---:|---:|---:|---|
| V02 residual | +0.00024274 | 0.0001 | 0 | 3 | No |
| V03 tiny ViT | +0.00179098 | 0.0001 | 0 | 3 | No |
| V04 tiny ConvNeXt | +0.00004313 | 0.0001 | 1 | 3 | No |
| V07 token Mixer | −0.00003505 | 0.0001 | 1 | 3 | No |

Thus **the bounded original runs tried one added-width candidate
(stage count two), while all four Bayesian runs remained at the initial
width**. This is grounded in actual checkpoint decision state,
not an inference from final test accuracy. The one-batch training
budget is insufficient to determine whether the shadow gate would
work at realistic convergence or a different patience threshold.
The trace is limited to stage-selected best-epoch evidence, not
the full stochastic posterior time series.
