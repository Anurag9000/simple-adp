# Verified bounded real-data Iris Bayesian ADP pair — 4 November 2026

**Evidence grade:** `BOUNDED_REAL_DATA_PASSED` (one dataset / one seed / one
width-growth route; **not** a full scientific comparison).

- **Source commit:** `2452f9e951d9d40a3bd1f245b1a142dc683e4f39`.
- **Successful GitHub Actions run:** https://github.com/Anurag9000/simple-adp/actions/runs/41838348200
- **Artifact ID:** `13326732823` (GitHub retention: 30 days).
- **Runner:** GitHub-hosted Ubuntu CPU, PyTorch 2.6.0,
  scikit-learn 1.9.1.
- **Data:** actual Fisher Iris dataset from `sklearn.datasets.load_iris`,
  150 records × 4 features, three species.
- **Dataset SHA-256:** `d51e090047c0dc3825bc350fe3c9282962c8d44dbf0fafba565b0328c461872b`.
- **Data split:** 90 train, 30 validation, 30 final test, stratified,
  identical for both arms.
- **Legacy split SHA-256:** `6c74fbc568ea47e0234b51b3d695ef819ac63e34e0145d00afa8cffefa3341fa`.
- **Boundary-aware split SHA-256 v2:** `84717c369b1ea9921a0557eab52339699333a1c7fe195e6bc63738d681ec4366`.
- **Model seed:** `73`; **original growth route:** `ADP-W-All`.
- **Matched settings digest (excluding trigger):**
  `1653aa2d4ac04812112624725ec3df62221d00fa4d63307eb394a9a4fb593870`.
- **Only permitted settings difference:**
  `expansion_criterion = loss` versus `variance_stagnation`.

## Observed held-out metrics

| Measurement | Original loss trigger | Shadow-BGD variance trigger | Bayesian − Original |
|---|---:|---:|---:|
| Accuracy | 0.6666667 | 0.3333333 | −0.3333333 |
| Balanced accuracy | 0.6666667 | 0.3333333 | −0.3333333 |
| Macro F1 | 0.5411255 | 0.3120680 | −0.2290575 |
| Log loss (lower better) | 0.8955997 | 1.0365060 | +0.1409063 |
| Final hidden-layer widths | [6] | [3] | distinct |
| Candidate-history entries | 4 | 1 | distinct |
| Fit time, seconds (small CPU sample; not controlled timing science) | 1.6895 | 0.2708 | −1.4187 |

The Bayesian arm remained at its initial width, while the original
loss-triggered method selected width six. These are genuine observations
for this seeded run; they must **not** be generalized to a superiority
claim for either criterion without repetitions, controls and statistics.

## Artifact and model integrity

The successful workflow emitted `pair_receipt.json`, both signed stage
checkpoints, both inference bundles and the source split archive.
The artifact ZIP was independently inspected following CI completion.
Both checkpoint dictionaries and inference bundle dictionaries were
safely loadable with `torch.load(..., weights_only=True)`.
The model and split file hashes matched the JSON receipt.

| File | SHA-256 |
|---|---|
| `original/estimator.model.pt` | `4465d5fa84ea4283fe63ba53be895c09e53a0009fd611039aaaff81bb2e99883` |
| `bayesian/estimator.model.pt` | `bed95d4218f9596bf5d80f08a112304fc1ad168ada7e341512cfbcd483128030` |
| `original/checkpoint.pt` | `1f75e480bd28a1e61794693c9f1371f02465df9689f71afa18ed4c979a800a2f` |
| `bayesian/checkpoint.pt` | `b23e4875d036ec31151bdc65d07d368bc28b0131ed7d46dcb460d715b5cc70c2` |
| `iris_split.npz` | `5278b2bd20b638e4bfb387ad2d65f77ff16bad10ff5601a5956eedfffe86ff31` |
| `pair_receipt.json` | `8e7bd729e2e674fbdc98ec86bed2ddb11e831e935d983175cd2c154b6a541b4b` |

## Exact reproduction

```bash
python -m pip install -e '.[benchmark]'
python scripts/run_bounded_iris_bayesian_pair.py \
  --seed 73 --output benchmark-results/bounded-iris-paired
```

Use a **fresh empty output directory** per execution; output overwrites are
deliberately disallowed. Source, dependency and hardware differences may
change floating-point results, so retain exact versions and checksums.

## Limitations and remaining science gates

This run does not include all native methods, visual V01–V11,
source-weight-verified V03/V04, other tabular suites, forecasting or
recommendation tasks, multiple seeds, posterior calibration, matched
capacity/compute controls, or confidence intervals. Inference-model
readability and file hashes do not independently establish every claimed
internal controller transition; detailed growth event trace audits remain
separate. Shadow Monte Carlo gradient-pass costs and energy are not
measured in this pair receipt. Scientific closure remains outstanding.
