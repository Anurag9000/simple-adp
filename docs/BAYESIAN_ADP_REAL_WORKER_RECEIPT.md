# Verified actual planner → worker execution on two real packaged sklearn datasets

**Evidence:** `BOUNDED_REAL_DATA_WORKER_EXECUTED` at source
`ea2c1af3ac3913afd770a0dd592f34b108d5e77a`.

- **Successful GitHub Actions run:**
  https://github.com/Anurag9000/simple-adp/actions/runs/41844517144
- **Artifact ID:** `13330257866`, containing manifest, four immutable
  worker specifications, real child logs, model inference bundles,
  estimator checkpoints, source split files and signed job results.
- **Runtime path:** `SuitePlan.build` → `audit_manifest` (unsliced) →
  four independent `python -m simple_adp.runner.worker --spec` child
  executions → checked `result.json` receipts.
- **Source model route:** `ADP-W-All@depth-1-width-3`; separate Bayesian
  route suffix `__bgd_variance_stagnation`.
- **Datasets:** real packaged Iris (150 rows) and Diabetes (442 rows), no
  network download, enabled only by `SIMPLE_ADP_ENABLE_SKLEARN_REAL=1`.
- **Exact independent worker pairs:** two original loss-growth jobs +
  two shadow-BGD variance-growth counterparts. Four distinct run IDs and
  result lineages; no scheduler slice. A plan-time auditor still correctly
  reports zero **proven** runs until a separate runtime receipt is checked.

## Observed held-out metrics

| Real task | Original loss-triggered worker | Bayesian variance-triggered worker |
|---|---:|---:|
| Iris classification accuracy | 33.33% (width 3) | 33.33% (width 3) |
| Diabetes regression RMSE | 164.3095 (width 6) | 165.1798 (width 3) |

These are **not** the same seed as the separate direct adapter proofs:
real worker Iris uses derived model seed `1968597252`, Diabetes uses
`1125583831`. Both arms of each true pair use the *same derived seed*.

Iris source SHA-256:
`d51e090047c0dc3825bc350fe3c9282962c8d44dbf0fafba565b0328c461872b`.

Diabetes source SHA-256:
`5b54beae03062886d76f420a73e8a582f6b98e2d7052331b15dc79deedd5da0b`.

The original/alternative arms share their dataset-specific split SHA-256:

- Iris `6c74fbc568ea47e0234b51b3d695ef819ac63e34e0145d00afa8cffefa3341fa`.
- Diabetes `4d6bc52b36431557579cf002e820217dea63dde9cf0ecb37bc52c5b48ce01575`.

## Independently verified artifacts

The downloaded GitHub Actions ZIP contained four completed worker
`result.json` documents and four `estimator.model.pt` inference
bundles. For every arm, each ZIP member was independently SHA-256 hashed
and matched the GitHub-generated receipt. The ZIP also contained
four `estimator.checkpoint.pt` stage/optimizer snapshots, whose
SHA-256 checksums are:

| Task | Original checkpoint SHA-256 | Bayesian checkpoint SHA-256 |
|---|---|---|
| Iris | `ddf4233fd78641e7cbc913d32f4a8cf27a27c20a0ef8df1bd1650c501a4631` | `ae1329a922e70855757931363b18bc217421d408ec89264d8837ff74c44505ee` |
| Diabetes | `429bb46987bd2a201e382268f01be1e6bea3d12359b3c1d5648a5a092a36a82c` | `bb444140dddd104649c3b4fddd5744f32fd779a553071cb8c39dc93f2d4c7cef` |

Subsequent runner receipts additionally record and check stage-checkpoint
SHA-256 and byte length programmatically. No full dataset/provider matrix,
GPU measurement, multi-seed worker proof, or real external data asset
validation is inferred from these four bounded jobs.
