# Supervised ADP: loss-growth versus auxiliary Bayesian-variance growth

## Status and scope

This work is a **software-hardening increment**, not a completed scientific
comparison. It targets the existing supervised `simple-adp` experiments; it
does not add continual learning, task heads, reinforcement learning or a new
Bayesian optimizer. Original Adam/AdamW weight updates remain unchanged.

The inspected repository source baseline is
`c81a426fd760df2f40d006527bf166523508390d` (2026-11-03). The source
file tests for the changes described here have **not** been run against a
full local checkout or final GitHub HEAD. Isolated module/fixture tests passed
(`36 passed` locally, CPU) after applying the accompanying patch.

## Changes in the paired implementation patch

1. Exact posterior resume: `ShadowBGDVariance.load_state_dict(..., model=...)`
   rejects nonfinite/nonpositive sigma, missing or mismatched model geometry,
   corrupt generator state and invalid observation evidence. It restores only
   after full validation. Actual growth-time `_synchronize` overlap migration
   is left intact. `StageTrainer` passes its model during active and best-epoch
   observer restoration. Posterior-state serialization schema remains 1.
2. Immutable run specification publication: the original and alternative
   branches of `SuitePlan.build` call the same checked publisher. Identical
   preexisting specs are reused; a changed criterion, seed, optimizer,
   architecture or arbitrary worker field is an error. Publish is through an
   fsynced temporary payload and atomic hard link (no replace), so competing
   writers cannot silently overwrite one another. A filesystem without
   hard-link support fails closed instead of silently reverting to replace.
3. An explicit static auditor reads the complete `SuitePlan` manifest,
   checks its digest, duplicate IDs/artifacts, original/counterpart mapping,
   effective scientific settings, output isolation, split identifiers,
   training command parity, and finite full-plan counts. It emits
   `pair_inventory.json` with per-pair identities, fingerprints, provenance,
   fixed/control exclusions and **`NOT_EXECUTED`** experiment statuses.
   A sliced manifest is rejected by default rather than misreported as a
   complete population.

## Usage on a real checkout (after patch is applied)

```bash
python -m compileall -q src/simple_adp
python -m pytest -q tests/test_shadow_variance_resume_integrity.py \
  tests/test_run_spec_immutable_integrity.py \
  tests/test_bayesian_pair_audit.py
python -m pytest -q tests/test_shadow_variance_growth_counterparts.py \
  tests/test_stage_variance_best_epoch_alignment.py \
  tests/test_stage_variance_interrupt_resume.py \
  tests/test_runner_openml_provider_plan.py \
  tests/test_runner_end_to_end.py
python -m pytest -q
# Produce an UNSLICED actual provider-backed dry-run manifest using
# the installed runner and available/licensed source datasets.
python -m simple_adp.runner.launcher --dataset-suites tabzilla,ctr23 \
  --dry-run --output benchmark-results/coordinated-suite
# Replace the path with the actual run-root manifest from the dry run:
python scripts/audit_bayesian_pairs.py RUN_ROOT/manifest.json \
  --source-commit CURRENT_GIT_SHA \
  --output artifacts/bayesian_adp/pair_inventory.json
```

For an intentionally scheduled slice, use `--permit-slice`; its report status
must remain `PARTIAL_SELECTED_SLICE`. A static plan, empty ledger, passing
unit test or successful simulator run cannot be elevated to
`PAIR_EXECUTED_VALID` without separately validated training receipts.

## Science definitions and boundaries

The alternative changes the *outer growth authorization* to the existing
auxiliary observer's posterior-variance contraction; it does not replace
Adam/AdamW, the candidate proposal objective, validation-based candidate
acceptance or best-epoch rollback. The shadow observer is a train-only,
diagonal BGD-inspired estimator, **not** a full calibrated Bayesian posterior.
The default gate is the code's persistent low contraction heuristic. Loss
is still valid for choosing among architectures after a proposed expansion.

Each valid pair must share training source content, train/validation/test
partition, model initialization, seed, optimizer and schedule, proposal and
capacity budget, task and metrics. The auditor checks these via the reported
plan fields; it **cannot** independently rehash missing raw datasets, load
weights, measure actual compute or verify training metrics. Those are separate
execution and scientific evidence gates.

## Outstanding work (not certified by this patch)

- Source-route inventory: verify every native, visual V01–V11 (including
  V06a/V06b), standalone, and external worker route by tracing executed
  proposal and structural mutation, not names. The auditor's built-in
  adapter set mirrors the then-current planner allowlist; move both to a
  shared capability registry before claiming future adapter coverage.
- End-to-end correctness: run actual runner/launcher processes and verify
  original/alternative execution, restart, interrupted checkpoint recovery,
  split checksums, source asset provenance and no evaluator/test leakage.
- Posterior fidelity: trace actual classification, regression, ranking and
  forecast objectives and verify the shadow probe matches each native
  training loss or is marked incompatible. Test partial plasticity and all
  model-specific geometry transformations.
- Accurate hardware cost: measure observer Monte Carlo work, model FLOPs,
  optimizer steps, peak RAM/VRAM and wall time on the actual execution paths.
- Full finite datasets: materialize eligible suites and export the complete
  original-to-Bayesian ledger with exact unknown/blocked reasons, using only
  permitted data with verified split files.
- Scientific validation: run bounded real-source pairs, then full multi-seed
  studies and matched-parameter/compute supplementary controls. Report
  task-appropriate metrics, calibration limitations and paired uncertainty.
- CI and publication: run current-HEAD CPU full suite and GPU smoke where
  hardware exists. Retain CI/logs/receipts with SHA, then commit/push coherent
  source, tests and documentation to `main` after verified completion.

A current source commit, executed job receipts, validated dataset count,
completed multi-seed cohort size and final main-branch CI certification
**remain unverified** and must never be inferred from this document.


## 2026-11-04 — Legacy split-hash partition ambiguity and new archive verification

The historical `split_sha256` intentionally remains byte-compatible for
existing job IDs, manifests and cached source experiments. Its original
construction concatenates the train/validation/test int64 arrays without
recording their lengths; partition-boundary shifts can produce the same
legacy digest even with distinct learner-visible training rows.

Newly published split archives now include a separate
`split_partition_sha256_v2` using explicit partition lengths. The loader
verifies that digest when present; older archived splits remain readable.
Existing split files are compared against all planned train/validation/test
indices, not just the legacy hash. Publication uses unique temporary files,
fsync and non-overwriting atomic hard links so concurrent planners cannot
silently replace another split.

**Limit:** A historical archive without the v2 field remains legacy-verifiable
only, and should not be presented as cryptographically bound to its exact
partition boundaries. This is a backward-compatible integrity improvement,
not evidence of real dataset campaign execution.


## Shared native/visual adapter capability registry (2026-11-04)

`src/simple_adp/runner/adapter_capabilities.py` is the single source of
builtin modality and shadow-BGD pairing capability for the suite planner
and static pair auditor. The registry covers one native adapter and 21
explicit opt-in vision adapters; V03/V04 published-source variants retain
their external weight/checksum gates. External third-party plugins are
not automatically classified as eligible. A capability entry asserts a
registered code route, **not** verified source data, training execution,
real architecture mutation in every concrete job, or scientific completion.
Tests assert shared object identity and preserve the distinct V06a/V06b,
V07 token/channel/both, V08 insertion, and V09 bank routes.


## 2026-11-04 — Canonical run identity verification

The static manifest auditor now reconstructs every planned `RunIdentity.run_id`
using the same source codec as the suite planner, requires exactly the declared
identity fields, and checks the identity's model route against its actual
registered `route_label` (appending the variance suffix only for a paired arm).
A manually changed or copied job ID is therefore rejected even when someone
recomputes the ordinary JSON manifest digest. The digest alone is not a
cryptographic signature or authorization mechanism. This safeguard does not
substitute for checking real task data, source checksums or training receipts.


## 2026-11-04 — Gradient-connected posterior coordinate guards

Shadow-BGD variance contraction now excludes nominally trainable but
gradient-disconnected parameters from the contraction mean. Their sigma
remains unchanged, but they cannot manufacture posterior stagnation votes.
A loss with no parameter gradient graph or a model with zero observed
gradient-connected coordinates fails closed (no architecture proposal).
These guards apply only to the Bayesian observer; original Adam/AdamW
training and loss-based growth decisions are untouched. Focused tests
exercise partially connected, fully disconnected and constant-output
models. Full scientific calibration remains unverified.


## 2026-11-04 — Job output namespace binding

The static pair auditor now enforces the planner's actual output namespace
contract: each `result_dir` must terminate in the identity-derived nested
`run_id`, and the worker spec command must terminate in
`<run_id>/run_spec.json`. Unique-but-misbound paths are rejected, so both
source and variance jobs retain distinguishable, self-consistent artifacts.
This is still a structural certificate; it is not proof that the paths
contain real completed training results.


## Bounded genuine dataset pair: sklearn Iris (2026-11-04)

`scripts/run_bounded_iris_bayesian_pair.py` defines a reproducible single-seed
classification proof on the real Fisher Iris dataset distributed with
scikit-learn, not synthetic images or FakeData. It creates one train/validation/
test split, checks legacy/v2 split digests, resets the model RNG before each
arm, and uses `SimpleADPModelAdapter.fit_predict` on ADP-W-All with loss
versus shadow-BGD growth only. The two arms have independent checkpoints and
inference bundles, with an inspectable JSON metric/artifact receipt.

`.github/workflows/bounded-iris-bayesian-pair.yml` executes this proof on a
CPU GitHub Actions runner and uploads its receipt and model artifacts. Merely
committing the workflow is **not** evidence that its execution succeeded.
Until a successful run and verified log/artifact is observed, its status is
`NOT_EXECUTED`. Even once passed, it remains one dataset/one seed, not the
full native/visual/provider multi-seed study.


## Executed bounded real-data result (GitHub Actions, 2026-11-04)

The Iris CPU pair `41838348200` **completed successfully** at source
`2452f9e951d9d40a3bd1f245b1a142dc683e4f39`.
Original held-out accuracy was 0.6667 with final width 6; shadow-BGD
accuracy was 0.3333 with final width 3, for the same 90/30/30 split and
seed 73. The exact receipt, model/checkpoint SHA-256 hashes and limits
are recorded in [BAYESIAN_ADP_IRIS_REAL_DATA_RECEIPT.md](BAYESIAN_ADP_IRIS_REAL_DATA_RECEIPT.md).
This advances **one bounded real supervised pair** to
`BOUNDED_REAL_DATA_PASSED`; it does not certify the full planned matrix
or statistical superiority.


## Bounded Iris per-seed paired bootstrap analyzer (2026-11-04)

`scripts/analyze_bounded_iris_pair_results.py` accepts multiple downloaded
`pair_receipt.json` files with their per-arm model bundles and split archives.
Before comparison, it verifies each arm's inference-bundle SHA-256, split
SHA-256, growth trigger, metric finiteness, distinct seed and common dataset,
source revision, split and route. It reports per-seed paired differences
and seeded percentile bootstrap intervals over *model seeds only*.
This is exploratory uncertainty conditional on the **one fixed Iris split**,
not cross-dataset confidence or a substitute for compute-/capacity-matched
controls. The five-seed CI run must finish before it can be counted as
empirical evidence.


## Bounded real diabetes regression pair

`scripts/run_bounded_diabetes_bayesian_pair.py` defines a single-seed paired
supervised regression study on the 442-row real scikit-learn Diabetes dataset.
It uses a shared fixed split, identical native ADP-W-All capacity/optimizer/
epoch limits, the actual Adam/AdamW-compatible regression loss, and separate
model/checkpoint hashes. `.github/workflows/bounded-diabetes-bayesian-pair.yml`
runs it without downloading an external dataset.

Status remains **NOT_EXECUTED** until the workflow completes and its actual
result logs and artifacts are verified. This bounded regression run does not
claim forecasting/recommendation, multi-seed or full benchmark closure.


## CI-verified five-seed Iris results (2026-11-04)

Five real-data Iris pairs at seeds 73–77 passed GitHub Actions
`41839475995`, including the independent downstream receipt-hash
verification and paired-bootstrap analysis. The observed mean held-out
accuracy difference (Bayesian minus original) was −0.113333; exploratory
seed-bootstrap 95% interval [−0.240000, −0.006667], **conditional on one
fixed split**. Full original/variance, all-dataset scientific closure is
still outstanding. Source, metrics, artifacts and limits:
[BAYESIAN_ADP_IRIS_MULTI_SEED_REPORT.md](BAYESIAN_ADP_IRIS_MULTI_SEED_REPORT.md).


## Verified bounded real Diabetes regression (2026-11-04)

The source-matched Diabetes regression CPU pair succeeded in GitHub Actions
`41839748050` at commit `d385153130d0a0d51c441f4cbfa0ed20653f2b30`.
Independent ZIP verification confirmed the model/best-checkpoint hashes,
split archive, finite held-out RMSE/MAE/R² and safe checkpoint loading.
Original RMSE 164.9989 (width 6), Bayesian RMSE 165.2817 (width 3),
one seed, four-epoch bounded training. Exact reproducibility evidence:
[BAYESIAN_ADP_DIABETES_REAL_DATA_RECEIPT.md](BAYESIAN_ADP_DIABETES_REAL_DATA_RECEIPT.md).
This proves one bounded real supervised **regression** pair, not full
statistical or task-family closure.


## Real Diabetes five-seed comparison protocol

A five-model-seed CPU workflow now pairs original loss-triggered and
shadow-BGD variance-triggered ADP using the same real 264/89/89 Diabetes
partition. Each seed exports both trained model bundles and stage
checkpoints with SHA-256 receipts. The downstream
`scripts/analyze_bounded_diabetes_pair_results.py` independently
checks dataset, source commit, split identity, every model and
checkpoint digest and finite regression metrics before seed-paired
RMSE/MAE/R² bootstrap analysis.

The analysis conditions on one fixed Diabetes split; it is not
cross-dataset science. Source availability, incomplete optimization at
four epochs, and unmeasured shadow MC compute remain explicit limits.
The five-seed cohort is NOT_EXECUTED until new GitHub Actions evidence
is inspected and the artifacts verified.


## CI-verified five-seed real Diabetes regression (2026-11-04)

Successful workflow `41840790207` at source
`50937db2f2874efc67b0eae0d96d7a03786cec2b`: five native
original/variance regression pairs (seeds 73–77), independently
downloaded model/checkpoint receipts, and a successful hash-validated
seed-bootstrap analyzer. Mean paired RMSE difference (Bayesian minus
original) +0.424389, exploratory 95% model-seed bootstrap interval
[+0.352232, +0.480131]. This is **conditional on one fixed Diabetes
dataset/split and four-epoch training**; resulting R² is severely
negative. See [BAYESIAN_ADP_DIABETES_MULTI_SEED_REPORT.md](BAYESIAN_ADP_DIABETES_MULTI_SEED_REPORT.md).
Do not label it multi-dataset scientific closure.


## Bounded real Iris four-architecture mechanism proof

`scripts/run_bounded_iris_bayesian_pair.py` now also accepts a
`--model-id` from four representative native routes, using the
same genuine Iris dataset, one split and original/variance settings.
The `bounded-iris-growth-families.yml` workflow runs
width-only, depth-only, width-then-depth and depth-then-width pairs
independently and exports original and variance checkpoints.

A downstream validator `scripts/audit_bounded_iris_growth_families.py`
requires exactly four distinct supported native methods and verifies
the same source/split/seed, criterion-only difference and per-arm model
bundle SHA-256 before certifying four bounded family examples. Its
status remains NOT_EXECUTED until full job and receipt evidence exists.
No result here represents all 13 growing model configurations, real
visual architecture pairs, multi-seed significance or full science.


## Four representative native growth-family real-source receipts (2026-11-04)

The actual paired Iris matrix in GitHub Actions run `41841932711`
successfully trained loss/variance arms under `ADP-W-All`,
`ADP-D-Dcur`, `ADP-WD-All-Dcur`, and `ADP-DW-Dcur-All`,
with all eight checkpoints and inference bundles independently validated.
Source `01323511dea54075ed06f46f36d3c076c21c7961`.

The depth-only and depth→width **original** controllers proposed new
depth and rejected it; no accepted depth expansion, and no variance-arm
expansion, was proved by these short jobs. Thus this is **four executed
policy families**, not four proved successful growth mutations.
Precise model histories and hashes:
[BAYESIAN_ADP_IRIS_FOUR_GROWTH_FAMILIES_REPORT.md](BAYESIAN_ADP_IRIS_FOUR_GROWTH_FAMILIES_REPORT.md).

There are now 13 distinct (dataset, seed, native method) bounded
loss/variance configurations across the verified Iris and Diabetes
cohorts, but only **two datasets**, and only direct native adapter
training, not a completed coordinated-worker/all-provider campaign.


## 2026-11-04 — Complete canonical 13 adaptive-method bounded real-data experiment

The canonical `MODEL_REGISTRY` includes 14 source configurations; 13 are
adaptive and `MLP-Fixed` remains explicitly unpaired. The real Iris
runner now permits every adaptive ID while retaining the original four
representative-method group unchanged. The new workflow
`.github/workflows/bounded-iris-all-native.yml` schedules 13 independent
source/variance pairs at model seed 73, one real dataset/fixed split, all
with the same four-epoch/optimizer/capacity budgets and pair-specific
inference/checkpoint artifact namespaces. Its downstream auditor checks
exactly 13 unique methods (26 run records), complete membership,
growth-family/route identity, original-vs-variance trigger parity,
shared data/split/settings/source and SHA-256 integrity of **both**
inference bundles and stage checkpoints. This is a bounded CI proof,
not a claim of multi-seed or cross-dataset scientific closure. Until the
workflow and its downstream audit succeed, its status is `NOT_EXECUTED`.


## Opt-in real sklearn provider and actual persisted child-worker execution

`src/simple_adp/runner/sklearn_real.py` provides downloaded-free genuine
Fisher Iris classification and Diabetes regression sources behind
`SIMPLE_ADP_ENABLE_SKLEARN_REAL=1`; the default provider inventory is
unchanged. The provider regenerates the same fixed 731/732 source split,
content hash and metadata in planner and separate worker processes.

`scripts/run_bounded_sklearn_real_worker_pairs.py` builds an **unsliced**
`SuitePlan` with one selected canonical ADP-W-All route, then uses the
actual production `simple_adp.runner.worker --spec` subprocess for each
of two original and two shadow-variance jobs. It audits plan identity,
original/alternative source and seed parity, result.json status and run
signature, and SHA-256 of the persisted worker result and inference model.

`.github/workflows/bounded-real-sklearn-worker-pairs.yml` tests the
provider and executes the real workers, retaining run specs, manifest,
worker logs, checkpoint and inference bundles. This is **not** synthetic
plugin data and does not claim full GPU, visual or full-suite closure.
It is `NOT_EXECUTED` until the source-pinned workflow actually succeeds.


## Independently observed real Iris 13-method complete native taxonomy proof

The all-native CPU workflow at source `10a69c2615427d4fcd6ffe9566deee955944cee3`
completed **13/13 original and 13/13 variance child executions**, with a
successful downstream model/checkpoint-hash audit and source/split/seed parity.
The actual method-level final architectures and accuracy metrics are recorded
in [BAYESIAN_ADP_13_NATIVE_REAL_DATA_RECEIPT.md](BAYESIAN_ADP_13_NATIVE_REAL_DATA_RECEIPT.md).
`MLP-Fixed` is excluded as a fixed-network control. This closes bounded
**method execution coverage only**, not independent data/seed breadth,
accepted structural mutation for every family, runtime compute, visual
adaptation, official provider datasets or entire scientific campaign.


## Verified real-source planner → actual subprocess worker proof

The GitHub run `41844517144` completed at
`ea2c1af3ac3913afd770a0dd592f34b108d5e77a`. It exercised the
actual unsliced source planner, emitted two original/two Bayesian worker
specifications for real packaged Iris classification and Diabetes
regression, executed the four production child workers and independently
SHA-verified the persisted result and inference files. Details,
held-out metrics, derived pair seeds and exact stage-checkpoint hashes
are preserved in [BAYESIAN_ADP_REAL_WORKER_RECEIPT.md](BAYESIAN_ADP_REAL_WORKER_RECEIPT.md).
This closes a bounded **worker-path execution proof** only, not the entire
data/visual/multi-seed/science campaign.


## Whole-repository CPU suite evidence: complete native-method source baseline

GitHub Actions run
[41842958104](https://github.com/Anurag9000/simple-adp/actions/runs/41842958104)
successfully executed **all 19,528 CPU-enabled pytest tests** at source
`10a69c2615427d4fcd6ffe9566deee955944cee3`, with exactly two
physical-CUDA-only skips, in 678.37 seconds. This source revision includes
all 13 bounded real Iris adaptive method paired jobs. No failure filtering
was applied. The run demonstrates repository-wide CPU regression for that
**exact revision** and does not certify subsequent commits; the real-data
worker provider and later integrity/CI changes require their own
source-pinned full suite status.

The standalone Bayesian regression suite at
[41843801780](https://github.com/Anurag9000/simple-adp/actions/runs/41843801780)
also passed **305 focused tests** after scikit-learn was added to the
test workflow. No physical GPU or external pretrained source checkpoint
execution is inferred from either run.


## 2026-11-04 — Worker and manifest canonical identity binding

Persisted `RunSpec` serialization now rejects internally inconsistent
canonical `run_id`, reconstructed `route_label`, model route/suffix,
original-vs-shadow criterion, Bayesian posterior marker, source counterpart
and model seed before execution. `SuitePlan.from_manifest` applies
the same validation when resuming from a serialized plan. After the dataset
loads, `runner.worker.run_spec` checks the separately loaded split archive
SHA-256 against the `RunIdentity.split_sha256` and refuses negative or
out-of-source row indices. A validly checksummed but *wrong* split archive
cannot silently execute against a different planned split.

These are local internal-consistency/provenance checks, not cryptographic
proof of authorized source publication. They do not change Adam/AdamW,
growth gates, sampling, model initialization or held-out evaluation.
Focused corruption tests and CI coverage accompany the implementation.


## 2026-11-04 — Official real-image four-family original/variance worker proof

`scripts/run_official_fashion_visual_variance_pairs.py` uses genuine
official `torchvision.datasets.FashionMNIST` train/test images, one
fixed held-out partition (40 train/20 validation/20 official test)
and the production unsliced suite planner. It executes eight separately
signed `runner.worker` child jobs: original-loss and shadow-BGD variance
growth for `vision_inner` (V02), `vision_v03` (tiny ViT),
`vision_v04` (tiny ConvNeXt) and `vision_v07_token` (Mixer token site).

The strict receipt requires data/split/derived seed parity, identical frozen
source weight SHA within each pair, exact job IDs/criterion, finite held-out
metrics and both model/checkpoint hashes. The test workflow is
`.github/workflows/official-fashion-visual-bayesian-pairs.yml`.
**Status remains NOT_EXECUTED until the actual workflow and artifact pass.**
The compared encoders are **tiny frozen random architectures**, not actual
published pretrained checkpoints, and neither all 21 vision adapters nor
all 13 controllers are covered by this small proof. No matched-compute
or general superiority inference is warranted.


## Best-stage posterior gate evidence, actual real worker checkpoints

The real worker checkpoint artifact at `444856cf535009b66d75d0e132e0614b2f0e19ed`
contains inspectable, checkpointed posterior evidence. Direct read of its
`torch.load(..., weights_only=True)` stage payloads found:

| Real task | BGD observations (best stage) | MC samples per observation | Final contraction | Stagnation streak | Required streak | Gate met? |
|---|---:|---:|---:|---:|---:|---|
| Iris classification | 24 | 2 | +0.0152425 | 0 | 3 | **No** |
| Diabetes regression | 68 | 2 | −0.0264586 | 1 | 3 | **No** |

Both used recorded tolerance `0.0001` and initial sigma standard deviation
`0.5`. The best-stage signed search histories have only the initial
architecture, so no additional Bayesian structural candidate was evaluated
in either bounded worker job. This explains the absence of growth **at those
recorded best-stage decisions**; it is not a complete per-minibatch trace.

`scripts/audit_bounded_worker_shadow_evidence.py` verifies original and
Bayesian checkpoint SHA-256, requires original arms to be free of shadow
posterior state, validates positive finite sigma coordinates and observer
configuration, then produces `best_stage_posterior_audit.json` during
future real-worker CI. Its MC-forward counts are declared **lower bounds**,
not full training FLOPs, runtime or energy. Posterior calibration and
a complete decision time series are still outstanding.
