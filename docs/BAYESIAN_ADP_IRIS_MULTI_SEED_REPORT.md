# Five-seed real Iris original-versus-shadow-BGD comparison

**Evidence gate:** bounded real dataset, seed-paired CPU training,
hash-validated per-arm model artifacts, single fixed 90/30/30 split.
This is not a cross-dataset study or a scientifically complete benchmark.

## Execution evidence

- **Source commit:** `05e1eae49f0282f043b08f71a0560bf07b121b27`.
- **Successful CI run:** https://github.com/Anurag9000/simple-adp/actions/runs/41839475995
- All five `iris-real-pair` jobs **succeeded** for seeds 73–77.
- The downstream `analyze-iris-seeds` job **succeeded** after downloading
  all five exported artifact bundles and validating each model/split SHA-256.
- **Analysis artifact ID:** `13328860033`.
- **Analysis JSON SHA-256:** `462d7f76ada92273314917abd589442129ec83b14e4d678b0980139809a45a6a`.
- **Dataset SHA-256:** `d51e090047c0dc3825bc350fe3c9282962c8d44dbf0fafba565b0328c461872b`.
- **Split:** 90 training, 30 validation, 30 held-out Iris observations.
- Original and Bayesian runs shared source data, partition indices, model
  seed per pair and unchanged optimizer/model/epoch/candidate budgets.
  Only `expansion_criterion` changed.

## Per-seed held-out accuracy

| Seed | Original accuracy | Bayesian accuracy | Original / Bayesian final width |
|---:|---:|---:|---|
| 73 | 0.666667 | 0.333333 | 6 / 3 |
| 74 | 0.533333 | 0.333333 | 6 / 3 |
| 75 | 0.366667 | 0.333333 | 6 / 3 |
| 76 | 0.333333 | 0.333333 | 3 / 3 |
| 77 | 0.200000 | 0.200000 | 3 / 3 |

## Paired model-seed bootstrap summary

Differences are always **Bayesian minus original**. Bootstrap resamples
model seeds with replacement using 10,000 seeded replicates. The
percentile bounds are conditional on the **same single Iris split**.

| Metric | Paired mean | Exploratory 95% seed-bootstrap interval |
|---|---:|---:|
| Accuracy | −0.113333 | [−0.240000, −0.006667] |
| Balanced accuracy | −0.113333 | [−0.240000, −0.006667] |
| Macro F1 | −0.090378 | [−0.169025, −0.012976] |
| Log loss (lower is better) | +0.081853 | [+0.024267, +0.139439] |

The original loss-triggered method selected additional capacity in
three seeds and kept the initial network in two. The Bayesian alternative
remained at its initial width for all five. This experiment does not
identify whether the variance threshold is miscalibrated, whether the
posterior contracts too aggressively, whether four training epochs are
insufficient, or whether the model was capacity-sufficient. These are
explicit unresolved experimental questions, not conclusions.

## Science limitations

- The cohort is one real tabular classification dataset with one
  held-out partition and five model initialization seeds; it does not
  sample datasets, shifts, folds or general-purpose architecture families.
- Both arms use their nominal original training settings, but the
  observer's extra Monte Carlo gradient work was not separately measured.
- Capacity and compute matching, predictive/posterior calibration,
  pre-proposal RNG tensor parity traces, and threshold/patience ablations
  remain unexecuted.
- The observed interval should **not** be interpreted as a population-level
  significance claim about Bayesian growth.
- Artifacts in GitHub Actions have finite retention; keep the recorded
  hashes and regenerate using the pinned commit and documented scripts.

**Status:** `BOUNDED_REAL_DATA_MULTI_SEED`, not
`MULTISEED_SCIENCE_VALIDATED` for the full repository campaign.
