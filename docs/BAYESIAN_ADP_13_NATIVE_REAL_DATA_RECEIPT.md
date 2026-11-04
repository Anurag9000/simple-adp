# Verified bounded real Iris paired execution: all 13 native adaptive configurations

**Evidence status:** `BOUNDED_REAL_DATA_NATIVE_METHODS_PASSED` for one
packaged genuine Fisher Iris dataset, one fixed stratified split, one seed,
four short candidate-training epochs. This is **not** full scientific validation.

- **Source commit:** `10a69c2615427d4fcd6ffe9566deee955944cee3`
- **Successful GitHub Actions run:** https://github.com/Anurag9000/simple-adp/actions/runs/41842957987
- **Machine-audited artifact:** GitHub Actions artifact `13330615101`,
  `all-native-13-method-audit.json`, with 13 independently checked source/variance
  model bundles, stage checkpoints and receipt files.
- **Real data SHA-256:** `d51e090047c0dc3825bc350fe3c9282962c8d44dbf0fafba565b0328c461872b`.
- **Partition-boundary SHA-256:** `84717c369b1ea9921a0557eab52339699333a1c7fe195e6bc63738d681ec4366`.
- **Training / validation / test:** 90 / 30 / 30, identical across methods and arms.
- **Model seed:** 73; **optimizer and training/candidate budgets:** identical within each original/Bayesian pair.
- **Total:** 13 adaptive source jobs + 13 separately executed shadow-BGD jobs.
  `MLP-Fixed` is correctly excluded from the growing-pair cohort.

## Actual held-out experiment observations

| Adaptive source method | Growth family | Original selected widths | Bayesian selected widths | Original accuracy | Bayesian accuracy |
|---|---|---|---|---:|---:|
| LGND-W-Greedy | width-only | [6] | [3] | 66.67% | 33.33% |
| ADP-W-RR | width-only | [6] | [3] | 66.67% | 33.33% |
| ADP-W-All | width-only | [6] | [3] | 66.67% | 33.33% |
| ADP-WD-RR-D1 | width-to-depth | [5] | [3] | 70.00% | 36.67% |
| ADP-WD-All-Dcur | width-to-depth | [5] | [3] | 70.00% | 36.67% |
| ADP-WD-RR-Dcur | width-to-depth | [5] | [3] | 70.00% | 36.67% |
| ADP-WD-All-D1 | width-to-depth | [5] | [3] | 70.00% | 36.67% |
| ADP-DW-D1-RR | depth-to-width | [3] | [3] | 36.67% | 36.67% |
| ADP-DW-Dcur-All | depth-to-width | [3] | [3] | 36.67% | 36.67% |
| ADP-DW-Dcur-RR | depth-to-width | [3] | [3] | 36.67% | 36.67% |
| ADP-DW-D1-All | depth-to-width | [3] | [3] | 36.67% | 36.67% |
| ADP-D-D1 | depth-only | [3] | [3] | 33.33% | 33.33% |
| ADP-D-Dcur | depth-only | [3] | [3] | 33.33% | 33.33% |

The selected architecture widths represent **final accepted architectures**.
An unchanged depth-model width does not prove the controller lacked or never
attempted depth proposals; proposal/rejection traces must be analyzed
separately. The short bounded budget and small data cannot calibrate the
Bayesian posterior. Reusing the same source/test observations across 13
methods also means the 26 executions are not 26 independent dataset
replicates. No claim of broad superiority, independent confidence intervals
or matched total computation is supported.

## Still outstanding

- All 13 methods over multiple real datasets, independent splits and model seeds.
- Exhaustive original/variance real visual architecture and pretrained-weight
  model pairs, with explicit downloaded source/model checksums.
- Actual full unsliced OpenML/temporal/recommendation provider campaigns.
- Per-proposal decision and posterior diagnostic receipts, accepted vs rejected
  expansion actions, loss-before-growth alignment, observer-MC overhead.
- Matched-compute/matched-capacity studies; GPU and exact whole-run restart
  with actual user-supplied data and hardware where possible.
- Verified full-repository CI at the **latest** `main` commit.

## Reproduce a single native method

```bash
python -m pip install -e '.[benchmark]'
python scripts/run_bounded_iris_bayesian_pair.py \
  --seed 73 --model-id ADP-WD-RR-D1 \
  --output benchmark-results/iris-WD-RR-D1
```

Use a new output directory each time. For all 13, use the
`bounded-iris-all-native.yml` workflow and inspect its downstream
`audit_bounded_iris_all_native.py` SHA-validated audit artifact.
