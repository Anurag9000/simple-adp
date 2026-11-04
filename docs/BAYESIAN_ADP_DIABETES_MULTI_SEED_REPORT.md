# Real Diabetes five-seed original-versus-Bayesian ADP report

**Evidence:** `BOUNDED_REAL_DATA_MULTI_SEED` — five independently completed
paired CPU jobs on one real regression dataset with an identical fixed
train/validation/test split, **not** full multi-dataset research validation.

## Execution and provenance

- **Source code revision:** `50937db2f2874efc67b0eae0d96d7a03786cec2b`
- **Successful workflow:** https://github.com/Anurag9000/simple-adp/actions/runs/41840790207
- **Five original and five Bayesian paired stage-trainings succeeded**.
  Subsequent `analyze-diabetes-seeds` job also succeeded after downloading
  and SHA-256 verifying model bundles and stage checkpoints from each seed.
- **Analysis artifact ID:** `13329960017` (finite GitHub retention).
- **Analysis JSON SHA-256:**
  `4bd52eef43bfeadfa7ffa2f9cc4b61c6408d00ad5928e6644fbd11a2cfe99057`
- **Dataset:** `sklearn.datasets.load_diabetes`, actual 442 × 10 regression
  observations, dataset SHA-256
  `5b54beae03062886d76f420a73e8a582f6b98e2d7052331b15dc79deedd5da0b`.
- **Partitions:** 264 training, 89 validation, 89 held-out testing. Same
  split identifiers, row membership, source and optimizer configuration in
  every original/Bayesian pair.
- **Experiment:** native `ADP-W-All`, matched optimizer and maximum
  architecture/training budgets, with `expansion_criterion` the only
  intended treatment difference. Seeds: 73–77.
- The source is a **direct native adapter experiment**; the coordinator's
  full `SuitePlan.build` worker path was not executed by this study.

## Verified held-out results (lower RMSE/MAE is better)

| Seed | Original RMSE | Bayesian RMSE | Original width | Bayesian width |
|---:|---:|---:|---:|---:|
| 73 | 164.998888 | 165.281662 | 6 | 3 |
| 74 | 164.508406 | 164.990059 | 6 | 3 |
| 75 | 164.796443 | 165.226444 | 6 | 3 |
| 76 | 165.000505 | 165.496839 | 6 | 3 |
| 77 | 164.534210 | 164.965394 | 6 | 3 |

## Model-seed paired bootstrap statistics

The differences are Bayesian minus original (10,000 seeded bootstrap
replicates resampling **only model seeds**).

| Metric | Paired mean difference | Exploratory 95% percentile bootstrap interval |
|---|---:|---:|
| RMSE | +0.424389 | [+0.352232, +0.480131] |
| MAE | +0.418853 | [+0.361962, +0.479722] |
| R² | −0.025546 | [−0.028925, −0.021198] |

**Interpretation:** On this one fixed split and small four-epoch
budget, the loss-growth controller selected width 6 in all five seeds;
variance-growth selected width 3 in all five. Both have very poor
regression fitness (negative R²), so these results support only
executable scientific-comparison plumbing and a limited diagnosis of
growth-trigger behavior. No cross-task superiority conclusion follows.

## Integrity evidence

The actual CI analysis checks pair and dataset identity, source SHA,
split archive SHA-256, separate model/checkpoint SHA-256 and size receipts,
the original/variance criterion, finite RMSE/MAE/R² and unique seeds.
The artifact ZIPs were additionally independently inspected: all five
`diabetes_split.npz` hashes and **20 model/checkpoint hashes** across
ten trained arms matched the saved per-seed receipts.

## Missing requirements

Formal task/adapter-wide software closure and scientific multi-seed
validation are still open. In particular, this does not exercise
forecasting/ranking objective parity, native depth/alternating search
families, source-weighted visual models, actual coordinated child-worker
restart, GPU execution, matched FLOP/capacity studies, calibration,
posterior MC cost accounting, independent dataset folds or long-budget
convergence. A bootstrap over five seeds conditioned on one split is
not a general confidence interval across datasets.
