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


## Corrected loss-patience interference (2026-10-29)

Both source-native controller families previously checked the
shadow-BGD growth permission but *also* terminated their width,
depth and nested search loops when source **validation-loss failure**
patience was exhausted. Under a variance-stagnation counterpart that
was a second, unintended loss-based growth criterion. The generic
and dedicated controller loops now use loss-failure termination
**only for the original `expansion_criterion="loss"` jobs**.
Variance counterparts continue only while the most recent stage's
actual auxiliary posterior measurements report persistent non-
contraction and an architecture candidate fits the original capacity
or max-candidate budget. Candidate retention, global best and
validation scoring remain unchanged. See
`tests/test_shadow_variance_growth_counterparts.py::
test_variance_criterion_not_cut_short_by_original_loss_patience`.
This is source-level testing, not evidence of full trained-job parity.


## Native unsliced plan completeness check

`SuitePlan.build` now checks every concrete eligible ADP growth
job *before* applying `job_start_index` or `job_limit`. Each must
have exactly one alternative with the same task/dataset, split
fingerprint, model adapter, architecture strategy, seed and original
training configuration, differing only by the explicit shadow
variance criterion and its provenance fields. Duplicate run identities
or unmatched original growth jobs fail plan construction.
Fixed vanilla and screening routes are classified by their
source-defined `suite_group`, not only one historic model name.
The native manifest includes `variance_pair_audit` as source-backed
coverage evidence, not as proof that any job executed.

The OpenML plan tests now assert original route counts and separately
paired ADP growth counts while excluding screening/fixed controls.


## Executed CPU regression evidence — 29 October 2026

At source commit `27b9297011945d01a687a7b1d1498dd53aaf5021`,
GitHub Actions workflow
`Standalone ADP original-versus-variance contracts` completed
successfully:
https://github.com/Anurag9000/simple-adp/actions/runs/40880236237

Executed on an Ubuntu Python 3.11 CPU runner with Torch 2.6.0.
The actual test command covered
`tests/test_shadow_variance_growth_counterparts.py` and
`tests/test_runner_openml_provider_plan.py`:
**17 passed in 3.55 seconds** (job succeeded).
These tests validate targeted source mechanisms, sampled shadow-BGD
behavior and matched suite-plan construction, not complete OpenML
training, all original/alternate jobs, fault-injected checkpoint
equivalence or GPU execution. The job-wide completed count remains
zero until individually recorded experiment artifacts are verified.


## Paired-criterion CPU model-adapter execution — 29 October 2026

At source commit `569b3480a5c8f396c9055888a2c418a55f6e733b`,
the actual GitHub Actions Ubuntu CPU run
https://github.com/Anurag9000/simple-adp/actions/runs/40881288873
executed the paired original/variance test suite:
**208 passed in 7.19 seconds**.

The `test_runner_models.py` canonical ADP parametrization
now runs each registered adaptive model/profile on both
classification and regression fixtures under
`expansion_criterion="loss"` **and**
`expansion_criterion="variance_stagnation"`.
The same run also includes posterior and plan-integrity tests.
Training budgets and capacity are deliberately tiny for CPU
contract verification; no claim is made that every full OpenML
task, full-size architecture expansion, GPU runner or fault-injected
checkpoint has been scientifically completed.


## Stage best-epoch posterior/checkpoint consistency

The common `StageTrainer` used by dedicated and generic
ADP model families now stores the auxiliary sigma/RNG state
alongside the best-validation model snapshot and restores both
at stage completion. Previously it returned best weights but
a final-epoch posterior, allowing rejected training epochs to
affect variance-triggered growth. Intermediate and final
variance checkpoints include `best_shadow_bgd_variance`;
resume rejects old variance-stage checkpoints lacking that
metadata rather than silently using mismatched evidence.
Original loss-mode models and their optimizer trajectory
are unaffected. Regression:
`tests/test_stage_variance_best_epoch_alignment.py`.


## Safe NumPy RNG state under Torch 2.6 (2 November 2026)

A real GitHub Actions fault-injection test initially failed while loading
the durable mid-epoch checkpoint under `weights_only=True`, because the
legacy `numpy.random.get_state()` record contained a NumPy ndarray.
New checkpoints serialize that MT19937 key vector as ordinary Python
integers, which NumPy's restore routine accepts without conversion.
The scientific RNG stream is unchanged and existing in-memory legacy
tuple/array state still restores. The new regression exercises safe
disk load, interrupted minibatch recovery and posterior/RNG identity.
Older checkpoints containing raw ndarray RNG records may require
trusted legacy-load migration; no silent unsafe fallback has been
introduced to this test.


## Verified fault-injected variance stage resume (2 November 2026)

On source commit `55e712938b5b9ad7a1585f1183153ee5d76f792f`,
GitHub Actions run
https://github.com/Anurag9000/simple-adp/actions/runs/41394892660
passed **212 CPU tests**, including safe Torch 2.6 loading of a
persisted mid-epoch variance-stage checkpoint, restart after the
second minibatch, and equality with an uninterrupted stage for model
weights, optimizer-step counts, best-epoch BGD sigma, observer RNG
and deterministic minibatch scheduling.

At commit `edc75eba9d0b3eb484b3dc3746bea92eb29098e9`,
https://github.com/Anurag9000/simple-adp/actions/runs/41395163878
passed **219 CPU tests**, adding eight criterion-specific fault
injection cases spanning Adam/AdamW, classification/regression,
and interruptions both before and after a best-validation epoch.
Neither run establishes full coordinated runner restart, CUDA
bitwise parity or whole-science-job completion.

## Newly added visual pipelines

The separately developed visual V01 frozen encoder and V02
residual/single-site ADP pathways have their own documented
source-to-variance pairing, profile and signed split contracts in
`docs/visual_adp/IMPLEMENTATION_LEDGER.md`. They were not
part of the original tabular inventory and must be audited
and executed independently as new scientific job families.
V03 onwards are not declared supported jobs; unfinished
architectures are not counted as executable counterparts.


## 2026-11-02: fresh evidence after parameter-geometry changes

The auxiliary Adam/AdamW shadow-BGD observer now detects a change in
the **named trainable parameter geometry** (new/removed name or resized
tensor). It retains learned diagonal sigma estimates on overlapping
coordinates and initializes newly added coordinates at the configured
prior standard deviation, but clears the old architecture's contraction
measurement, observation count, and non-contraction streak. This
prevents old-model stagnation votes from authorizing growth before
the new model has accumulated a fresh persistence window.

Source: `src/simple_adp/shadow_bgd_variance.py`. Regression: `tests/test_shadow_variance_growth_counterparts.py`.
Source-and-test integration commit: `d156a809a6e6809dd2cf825e8e1f4da1318ea425` on `main`.
The source and test were committed, but this ledger **does not claim a
successful new full-suite run**, GPU parity, real-dataset execution, or
fault-injected exact-resume equivalence on that commit. These remain
verification requirements; existing original loss-trigger jobs are unchanged.


## 2026-11-02: bounded CI evidence and distinct visual-path blocker

The source/test variance-geometry change above passed the
**Standalone ADP original-versus-variance contracts** GitHub Actions workflow
on commit `d156a809a6e6809dd2cf825e8e1f4da1318ea425`:
https://github.com/Anurag9000/simple-adp/actions/runs/41415123296 .
This verifies that workflow's focused CPU tests, **not** the complete
cross-dataset scientific suite, CUDA parity, or every interruption boundary.

A separate visual workflow
https://github.com/Anurag9000/simple-adp/actions/runs/41415150793
failed during test collection because a dictionary comprehension in
`src/simple_adp/visual_multistage_schedule.py` had a trailing comma
after `for site in SITES`. This syntax defect was corrected on `main`
at commit `73db80bfd43229762f0a0158d356f6f00e639716`.
The corrected visual workflow has not been verified green by this ledger.
The variance and visual paths must be tracked separately.
