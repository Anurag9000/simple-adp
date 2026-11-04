# Verified bounded real Diabetes regression ADP comparison

**Status:** `BOUNDED_REAL_DATA_PASSED` for one real regression dataset,
one model seed, one width-growth strategy. The full Bayesian ADP scientific
campaign remains outstanding.

## Evidence and source identity

- **GitHub Actions:** https://github.com/Anurag9000/simple-adp/actions/runs/41839748050
- **Source commit:** `d385153130d0a0d51c441f4cbfa0ed20653f2b30`
- **Downloaded artifact ID:** `13328560916`
- **Dataset:** real scikit-learn Diabetes progression regression, 442
  observations, 10 numeric features, `sklearn.datasets.load_diabetes`.
- **Dataset hash:** `5b54beae03062886d76f420a73e8a582f6b98e2d7052331b15dc79deedd5da0b`
- **Model seed:** 73
- **Split:** 264 training / 89 validation / 89 held-out test.
- **Legacy split SHA-256:** `4d6bc52b36431557579cf002e820217dea63dde9cf0ecb37bc52c5b48ce01575`
- **Partition-aware split SHA-256:** `8aa7b3fcbaf10c9dd575d7789776832b949f0d32ce6b19d8ebac8c4443d1ced6`
- **Compared architecture:** `ADP-W-All`, same initial hidden width,
  maximum width, optimizer, batch size, training epochs and candidate limit.
  Only `expansion_criterion` changes from `loss` to `variance_stagnation`.

## Actual held-out regression metrics

| Metric | Original loss trigger | Bayesian variance trigger | Bayesian minus original |
|---|---:|---:|---:|
| RMSE (lower better) | 164.998888 | 165.281662 | +0.282774 |
| MAE (lower better) | 147.462939 | 147.785729 | +0.322790 |
| R² (higher better) | −3.966856 | −3.983895 | −0.017039 |
| Final hidden-layer width | 6 | 3 | distinct |
| Recorded candidate history entries | 4 | 1 | distinct |

**Interpretation:** neither method is well trained under this deliberately
tiny four-epoch CPU budget. The Bayesian run did not grow beyond its initial
width while the loss-triggered method selected additional capacity.
These scores are a *functional execution and pairing proof*, not a
claim about practical Diabetes prediction accuracy or research superiority.

## Independently inspected artifact integrity

The GitHub Actions ZIP was downloaded; these model SHA-256s matched its
`pair_receipt.json`. Each training checkpoint and inference model bundle
also loaded safely as a dictionary using `torch.load(..., weights_only=True)`.

| Artifact | SHA-256 |
|---|---|
| `original/estimator.model.pt` | `6f036a8bfbb47a3ba03bc14af4abc9f6b24991a8423fb3ab6807dac4d3e7a5f2` |
| `bayesian/estimator.model.pt` | `4aba352e3569df16720cab37aa9bb4f8f2dd94d9063df3707992ca48e6c0f0b5` |
| `original/checkpoint.pt` | `438f8d1dbc1314d7606028ea2bcd32e3df3186010ad3f8396063e683838d4776` |
| `bayesian/checkpoint.pt` | `9fe8dd641eb207103fe36d888e1e3062a3487cc0dada981b32a1559917cc104e` |
| `diabetes_split.npz` | `caaa5b8dd67b89e68b074f6a1c36dd6a770840723a8d849754eb0c66211e3a82` |
| `pair_receipt.json` | `67f6450fcde7cd652536c34a9ca6f13e5a9cc51dbe8130c6dc056aec332d7228` |

**Unresolved:** multi-seed regression, matched compute/capacity, uncertainty
calibration, shadow posterior MC resource overhead, forecasting/ranking/
recommendation, broader datasets and full visual-family real-source studies.
