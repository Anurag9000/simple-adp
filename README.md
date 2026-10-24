# simple-adp

Task-adaptable adaptive architecture search for tabular data using feed-forward PyTorch MLPs.

The algorithm contract is `~/Documents/ADP_algorithms.docx`. The project implementation plan is in [`docs/ADP_TABULAR_IMPLEMENTATION_PLAN.md`](docs/ADP_TABULAR_IMPLEMENTATION_PLAN.md).
The generic pressure-aware suite's operational guide and ledger-faithful behavior map are in [`docs/COORDINATED_PRESSURE_AWARE_SUITE_GUIDE.md`](docs/COORDINATED_PRESSURE_AWARE_SUITE_GUIDE.md) and [`docs/COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md`](docs/COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md).
The ledger-to-code function mapping is in [`docs/COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md`](docs/COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md); refresh or check its AST inventory with `python scripts/generate_runner_function_map.py` or `python scripts/generate_runner_function_map.py --check`.

The package exposes the guide's 14 experiment configurations through `TabularADP` and `MODEL_REGISTRY`. Classification and regression adapters are included; custom tasks can provide a `TaskAdapter`. Adaptive fitting requires caller-provided validation data so every architecture is judged on the same fixed set.

```python
from simple_adp import TabularADP, RegressionTask

model = TabularADP("ADP-W-All", task=RegressionTask())
model.fit(X_train, y_train, validation_data=(X_valid, y_valid))
predictions = model.predict(X_test)
```

`MLP-Fixed` defaults to the vanilla baseline grid: uniform-width architectures at depths 1–5, with widths 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512 (50 candidates total). Each entry is a complete hidden-width vector; for example, depth 3 at width 16 is `[16, 16, 16]`. Use `vanilla_mlp_grid()` to inspect or reuse the preset, or pass `fixed_grid=[[32, 16], [64, 32]]` to override it. Feature arrays are treated as numeric; DataFrames preserve the training column order and infer numeric versus categorical fields.

The mixed-order configurations use the generic candidate trainer (AdamW and an adapter-provided optional auxiliary loss). `generic_profile="faithful"` reproduces the guide's documented failure-counter, best-candidate, and `max_neurons` behavior. `generic_profile="corrected"` retains a true global best, resets failure counts after valid phase improvement, and enforces the aggregate unit cap. Dedicated DEN and LG-ND configurations use their separate Adam plus cosine-restart stage trainer.

## Benchmarking hard public datasets

Install the optional OpenML benchmark dependencies with `pip install -e '.[benchmark]'`. The benchmark runner defaults to all three pools: 36 TabZilla hard classification tasks, 72 broader OpenML-CC18 classification tasks, and 35 OpenML-CTR23 regression tasks. TabZilla is described as the 36 hardest classification tasks among the paper's evaluated pool; CTR23 excludes regression problems exactly solved by a linear model. Neither source alone guarantees that deeper MLPs outperform shallow ones, so the validation screen also requires measurable depth sensitivity.

For every OpenML task split, its test rows stay outside that split's training, validation, and screening. The runner makes a seeded, stratified (classification) or random (regression) validation split from the OpenML training rows, fits preprocessing on the remaining training rows, and screens 22 architectures: all 10 one-layer vanilla widths from 1 to 512, plus depths 2, 3, and 5 at widths 16, 64, 256, and 512. Each is trained with two screen seeds by default. A fold is excluded if a one-layer MLP reaches the configured ceiling repeatedly (default: 100% validation accuracy or R-squared >= 0.999 over screen seeds), or if deeper networks fail to improve validation balanced accuracy/R-squared by at least 0.02 over the best one-layer model. A dataset enters the expensive comparison only when at least 80% of its screened folds pass both gates (configurable with `--min-hard-fold-fraction`). The shallow gate removes folds already solved by simple networks; the complexity-gain gate targets folds where adding depth changes validation performance. These are screening signals, not guarantees of a test-set gain. Passing folds compare all adaptive IDs, both generic profiles, the selected 50-candidate `MLP-Fixed` grid, and each of the 50 vanilla architectures individually. Results, split-index archives, fold screens, dataset inclusion decisions, and task/provenance metadata are written under the output directory. Completed model runs and compatible screen decisions are reused on restart.

Run the screen first and inspect `screening.jsonl` for architecture metrics and `dataset-decisions.jsonl` for per-dataset inclusion. `simple-adp-benchmark-report` also writes `dataset-screening.csv` after a screen-only run. Then rerun with the same output directory and settings to launch full comparisons only for folds in eligible datasets that passed both gates; cached screen decisions are reused.

```bash
simple-adp-benchmark --suites tabzilla,ctr23 --screen-only \
  --output benchmark-results/hard-tabular
simple-adp-benchmark --suites tabzilla,ctr23 \
  --output benchmark-results/hard-tabular
```

For a small pilot, limit tasks and folds:

```bash
simple-adp-benchmark --suites tabzilla,ctr23 --task-limit 1 --fold-limit 1 \
  --max-epochs 20 --seeds 0 --output benchmark-results/pilot
```

Run the full default pool after reviewing screening and pilot cost:

```bash
simple-adp-benchmark --suites tabzilla,ctr23 --output benchmark-results/hard-tabular
```

The default full pool is expensive: it includes 22 canonical/profile-specific ADP runs (five dedicated adaptive IDs, eight generic IDs under both profiles, and `MLP-Fixed`) plus all 50 individual fixed baselines per retained fold and seed. The screen runs first so folds solved at the shallow-MLP ceiling or with no measured validation depth gain do not consume the full search budget. Use `--suites tabzilla,ctr23` to omit CC18.

Summarize finished runs with `simple-adp-benchmark-report benchmark-results/hard-tabular`. It writes per-dataset means, per-model macro means and average ranks, plus counts for completed, failed, and screened-out runs.

## Supervised temporal and recommendation adapters

The package now has adapter and deterministic framing primitives for fixed-window forecasting, per-step labeling, event classification, grouped ranking, supervised anomaly labels, and rating-derived preference labels. Enabled source providers include UCI Electricity (128 lags, next interval, all meters, stride 24 on its long series), Monash Australian Electricity Demand (horizons 1, 7, and 28), PhysioNet MIT-BIH and LTDB (AAMI beat classes, record-level splits), NAB v1.1, UCI HAPT (raw six-channel sensor data and per-step 128-sample windows with disjoint subjects), 24 individually inventoried ODDS supervised anomaly artifacts, and MovieLens 100K/1M as rating, rating-derived preference, and per-user grouped-ranking tasks. Source archives and PhysioNet record files are cached under `~/.cache/simple-adp/datasets` by default (override with `SIMPLE_ADP_DATA_CACHE`); run manifests store source digests and provenance. The Monash Australian Electricity Demand archive is CC BY 4.0; NAB files retain their individual upstream terms, so NAB is not represented as having one blanket data license. Install `simple-adp[physionet]` to load MIT-BIH/LTDB WFDB records and `simple-adp[odds]` (SciPy and h5py) to read ODDS MATLAB v5/v7 and v7.3 MAT files. ODDS does not state a separate license for its derived MAT artifacts; catalog entries record this and original-source terms separately, and this package does not redistribute those artifacts. MovieLens preference means a rating threshold of 4, not a click label. Per-user splits are chronological 80/10/10 and the ranking sampler keeps each training user's group intact. UCI forecasting splits by target position in 70/10/20 chronological order, so targets do not cross partition boundaries.

### Default coordinated comparison: one representative dataset per task

`simple-adp-suite` defaults to the task-balanced dataset selection in [`docs/REPRESENTATIVE_BENCHMARK_PROFILE.md`](docs/REPRESENTATIVE_BENCHMARK_PROFILE.md): MiniBooNE classification, Superconductivity regression, Monash Australian Electricity Demand at horizon 28, UCI HAPT sequence labeling, MIT-BIH Arrhythmia beat/event classification, MovieLens 1M for rating/preference/ranking, and ODDS Annthyroid supervised anomaly classification. This is one dataset per task family; MovieLens 1M is reused across its three rating-derived recommendation objectives. The full OpenML and provider catalogs remain available by passing `--dataset-suites` explicitly. Select one task from any multi-task provider suite with `provider:suite@task_id` (for example, `monash:aus-electricity-demand@monash-aus-electricity-demand-h28`).

Launch the selected pressure-aware comparison with:

```bash
scripts/run_representative_pressure_aware_suite.sh
```

This uses the runner's complete configured comparison matrix on each selected task: all registered ADP configurations (both generic-search profiles), the full 50-architecture vanilla MLP grid, and its configured screening routes. The pressure-aware scheduler coordinates the jobs. Add normal suite CLI arguments after the script name to set limits or override dataset selectors; the wider dataset providers remain registered and selectable.

List currently registered sources with `simple-adp-suite --list-suites`. To explicitly include the existing OpenML pools and every enabled provider suite, including all 24 individually inventoried ODDS MAT artifacts, launch with:

```bash
simple-adp-suite --set 'scheduler="pressure_aware"' --dataset-suites cc18,tabzilla,ctr23,uci_temporal:electricity,uci_sequence:hapt,movielens:ml-100k,movielens:ml-1m,nab:v1.1,monash:aus-electricity-demand,physionet_ecg:mitdb,physionet_ecg:ltdb,odds:cardio,odds:thyroid,odds:satimage-2,odds:lympho,odds:glass,odds:vowels,odds:musk,odds:letter,odds:speech,odds:satellite,odds:shuttle,odds:breastw,odds:arrhythmia,odds:ionosphere,odds:mnist,odds:optdigits,odds:http,odds:smtp,odds:mammography,odds:annthyroid,odds:pendigits,odds:ecoli,odds:wine,odds:vertebral
```

Use `--dry-run` to persist and inspect the route/split manifest without starting workers. The exact task contracts, implementation sequence, source transformations, eligibility checks, design decisions, and exclusions are documented in [`docs/SUPERVISED_TASKS_IMPLEMENTATION_PLAN.md`](docs/SUPERVISED_TASKS_IMPLEMENTATION_PLAN.md). A task adapter or framing helper alone does not mean its named source dataset has been enabled; the default experiment catalog should include only providers with an artifact loader and deterministic split tests.

Pass `checkpoint_path` to `fit` to write resumable stage state. `save(path)` writes a fitted estimator artifact for prediction, and `TabularADP.load(path)` restores that artifact. The stage trainer accepts an interrupted stage's `resume_state` together with its matching architecture and data tensors.
