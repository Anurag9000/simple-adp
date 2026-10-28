# Training-only scientific-job catalogue — simple-adp

> **Source-grounded static inventory, not a claim of completed or exhaustive executable runs.** Snapshot: `main` at `e15dbdfdc7bd569ab4d90df704f205141b120209`. This file was derived from the complete non-truncated tracked Git tree and the native repository sources; no external datasets, generated local plans, checkpoint stores or execution environments were available.

## Scope and identity

A distinct scientific **training job** is one reachable, compatible trainable method/model + dataset/task stream + learner-visible protocol + architecture/objective configuration that materially changes training. Repeat seeds and evaluation-only routing/analytics are not additional scientific jobs. Do **not** infer all combinations from a registry, file names, or paper descriptions. Suites labelled ablation/sweep/smoke require per-cell classification before inclusion; ablations are excluded from the primary count, not erased from this source ledger. Controls that train valid baselines may be legitimate jobs and are not automatically discarded.

**Verified exact distinct training-job count: NOT ESTABLISHED.** The following counts are of **tracked candidate source files, not materialized jobs**: 0 configuration/registry files; 6 candidate launch/training scripts; 10 selected internal registry/dataset/experiment files; 0 configuration paths explicitly named ablation/sweep/smoke. Even when a planner exists, its realized count requires using its native compatibility checks, actual selected datasets and resource/seed/manifest inputs.

## Native scientific authority and central training path

- Primary evidence: [`docs/COORDINATED_PRESSURE_AWARE_SUITE_GUIDE.md`](../docs/COORDINATED_PRESSURE_AWARE_SUITE_GUIDE.md).
- Existing launcher/entry point: [`src/simple_adp/runner/launcher.py`](../src/simple_adp/runner/launcher.py).
- **Read-only inventory or planning command:** `python -m simple_adp.runner.launcher --dataset-suites tabzilla,ctr23 --dry-run --output benchmark-results/coordinated-suite`.
- **Native execution example (NOT run for this audit):** `python -m simple_adp.runner.launcher --dataset-suites tabzilla,ctr23 --output benchmark-results/coordinated-suite`.
- Protocol-specific caveat: The repository declares 14 ADP algorithm configurations and MLP-Fixed architecture candidates, but dataset pools, screening-gated folds, regression/classification adapters, user-provided providers and validation-gated exclusions determine materialized training jobs. A generic algorithm×all-OpenML Cartesian multiplication is invalid. Native SuitePlan emits stable split/seed/route identities and the scheduler owns resource isolation, resume and artifacts.

The native launcher is authoritative for validation, deterministic job identity, dependencies, resume, resource checks and artifact storage. This index does not replace that code with a generic runner, does not create empty fake job records and does not declare a job completed merely because a configuration exists.

## Registered inputs in tracked tree (0)

These paths are an exhaustive **path inventory of files matching** `configs|suites|manifests|registry|specs` JSON/YAML/TOML conventions; their contents are **not** all individually certified to be runnable training jobs. Preserve per-file role and filter ablation/evaluation/diagnostic material when building native plans.

- None identified in tracked source

## Possible execution and orchestration entry points (6)

These paths were conservatively identified by filename, not executed. Some may be development, infrastructure or evaluation-only; verify callers/CLI before treating any as training.

- `scripts/run_representative_pressure_aware_suite.sh` — blob `ce24cdf45de4`
- `scripts/simple-adp-suite.ps1` — blob `eabad7c8f2e0`
- `scripts/simple-adp-suite.sh` — blob `71fbb5716f18`
- `src/simple_adp/runner/plan.py` — blob `c6332d906412`
- `src/simple_adp/runner/scheduler.py` — blob `65ff0aeba3b3`
- `src/simple_adp/training.py` — blob `c052feae12af`

## Additional registry, dataset and experiment source anchors (10)

This selected source index helps trace registered methods, dataset builders, manifest validators and experiment kernels. It is not a proof of reachability.

- `docs/SUPERVISED_DATASET_CATALOG.json` — blob `e6acabf5c1e8`
- `src/simple_adp/registry.py` — blob `f502a6c29e33`
- `src/simple_adp/runner/datasets.py` — blob `5f6a61f7007f`
- `src/simple_adp/runner/monash_datasets.py` — blob `de6880d062f0`
- `src/simple_adp/runner/nab_datasets.py` — blob `71702f4c9c4a`
- `src/simple_adp/runner/odds_datasets.py` — blob `ff3b40c5d06a`
- `src/simple_adp/runner/physionet_datasets.py` — blob `14d327a42835`
- `src/simple_adp/runner/public_datasets.py` — blob `72b519c41083`
- `src/simple_adp/runner/sequence_datasets.py` — blob `54a453d75729`
- `src/simple_adp/runner/temporal_datasets.py` — blob `ccfb15814de4`

## Explicitly named ablation/sweep/smoke config sources (0)

File naming alone cannot distinguish a legitimate comparison from a hyperparameter ablation. The primary non-ablation job planner should inspect the actual expanded cells, classify them and deduplicate on canonical scientific training identity, not on suite name.

- None identified in tracked source

## Coverage, unresolved items and execution contract

| Check | Status | Evidence needed for closure |
|---|---|---|
| Tracked tree path census | **Complete at captured commit** | Non-truncated Git tree paths and blob identities above |
| Native runner located | **Source-identified** | Exercise help/plan in a checked-out environment |
| All methods and datasets actually runnable | **Unverified** | Native registry compatibility + prepared real data + dependency checks |
| Unique concrete training-job records | **Unmaterialized here** | Generate native plan, export canonical IDs and deduplicate scientific aliases |
| Seed/repeat-expanded execution count | **Unmaterialized here** | Explicit seed list, repeats, budget and a validated immutable plan |
| Trained results and resume integrity | **Not run** | Native checkpoints, data fingerprints, output manifests and successful completion evidence |

### Reproducible next verification

From a clean checkout at the pinned SHA, inspect the native CLI and run the **planning-only** command above. Archive the plan JSON/CSV, resolved configurations, rejected/skipped cells with reasons, input dataset identities, seed list, and source revision. Compare method/dataset/protocol membership with this inventory; only then calculate **exact distinct jobs** versus **execution records**. Execution is a separate operator action. Do not automatically download datasets or treat optional provider availability as success.
