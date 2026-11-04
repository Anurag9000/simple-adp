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
