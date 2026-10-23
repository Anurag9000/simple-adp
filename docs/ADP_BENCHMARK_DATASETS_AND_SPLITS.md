# ADP tabular benchmark datasets and split protocol

## Scope

Use public, documented, reusable tabular benchmark tasks to compare all 13 adaptive configurations and `MLP-Fixed` (the 50 uniform-width vanilla candidates). Keep task folds, preprocessing, random seeds, and compute budgets identical across methods. This is a benchmark plan and dataset inventory; it does not download data or launch training.

## Recommended dataset pool

### Primary classification pool: OpenML-CC18

OpenML-CC18 (all 72 tasks in study 99) is a broad discovery pool, not the default hard benchmark. It includes medium-sized classification datasets with 500–100,000 rows, at least two classes, limits on post-encoding feature count, and exclusions for grouped/time-dependent data, extremely small classes, sparse text, and trivially solvable datasets. These rules do not establish that neural architecture complexity matters, so the runner screens it before full comparison.

This suite deliberately excludes strongly imbalanced classification. That makes it suitable as the principal general-purpose classification pool, but it should not be described as covering rare-event classification. Add a separately labeled imbalance stress suite only after the primary benchmark is working.

### Classification stress pool: TabZilla Hard Datasets

Use all 36 tasks in the TabZilla benchmark (OpenML study 379) as the primary classification pool. TabZilla selected the 36 hardest classification datasets from a pool of 176 datasets evaluated across 19 algorithms. Its maintainers recommend the ten OpenML folds and their supplied validation splits. The term “hardest” refers to that study's benchmark selection; it does not promise that every dataset needs a deep MLP or that ADP will improve on a shallow MLP. Identify each dataset by OpenML dataset/task ID and report overlap with CC18 rather than counting duplicates as independent datasets.

### Primary regression pool: OpenML-CTR23

Use all 35 tasks in OpenML study 353 (`OpenML-CTR23`). This curated suite covers 500–100,000 observations, excludes sparse and non-IID/time-dependent data, and rejects regression datasets exactly solved by a linear model. That criterion supports nonlinearity, but does not prove a need for deep networks. Its task definitions provide the repeat/fold or holdout splits.

### Optional regression expansion

After the primary run, add OpenML's AutoML Benchmark Regression tasks to increase dataset diversity. Deduplicate by OpenML dataset ID against CTR23 and record task IDs separately because the same dataset may have different targets or split definitions. Do not silently merge alternate tasks.

## Split and evaluation protocol

1. Use the exact outer train/test indices from each OpenML task. The outer test fold is only used once for final scoring; never use it for early stopping, architecture selection, preprocessing fit, or checkpoint selection.
2. Carve an inner validation subset from each outer training fold: 20% for classification using stratified sampling; 20% for IID regression using a seeded random split. Keep the remaining 80% of the outer training fold as the estimator's training data. Use the same inner split for every algorithm, profile, and seed on that outer fold.
3. Fit imputation, scaling, and categorical encoders on the inner training partition only. Transform validation and outer test data with that fitted preprocessor. Preserve OpenML target definitions and categorical metadata; do not convert numeric-looking categorical fields to continuous values without documenting the decision.
4. Use task-defined grouped or temporal splits whenever those data types are intentionally added later. Never apply shuffled IID splitting to grouped observations, repeated subjects, or temporal prediction.
5. Run at least three fixed model seeds per outer split for neural initialization and data-loader order. Keep the split seed fixed across algorithms; vary only the model seed. Save seeds and row indices in the result manifest.
6. Before expensive ADP runs, train a 22-architecture screen on each split's inner training rows: all ten one-layer widths from 1 to 512, and depths 2, 3, and 5 at widths 16, 64, 256, and 512. Repeat each with two screen seeds by default. Exclude and record a fold if any one-layer model reaches a validation ceiling repeatably across screening seeds (classification accuracy 1.0; regression R-squared >= 0.999 by default), or if the best deeper architecture does not improve validation balanced accuracy/R-squared by a configured minimum (0.02 by default) over the best one-layer model. Include a dataset in the expensive comparison only if at least 80% of its screened folds pass both gates (configurable). This targets datasets with a repeatable, measurable validation architecture-complexity effect and removes folds solved by shallow networks; it does not guarantee a test-set gain. Retain every screened-out record.
7. Evaluate each trained/selected model exactly once on the outer test fold. Aggregate first over seeds within a fold, then over folds within a dataset. Report per-dataset results and macro-average/rank across datasets so large datasets do not dominate the overall score.

## Metrics

- Classification: primary accuracy and macro-F1; also report balanced accuracy and log loss when the adapter's probability outputs are valid. Keep binary and multiclass results separately visible.
- Regression: primary RMSE; also report MAE and R-squared. Compute each metric in original target units. If a target transform is used, fit it on training data and invert predictions before scoring.
- Efficiency: record total wall time, fit time, candidate/stage count, epochs, peak memory where available, and selected architecture. Report performance-versus-compute alongside best test score.
- Failures: retain explicit failed, timed-out, and unsupported records. Do not omit a method/dataset cell from ranking without showing why.

## Compared methods

- Run the 13 adaptive IDs in the registry.
- Run `MLP-Fixed` with the full 50-candidate vanilla grid: depths 1–5 crossed with uniform hidden widths 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512.
- For the eight generic-search model IDs, run both `faithful` and `corrected` profiles as separately labeled results. They are implementation behavior profiles, not extra canonical model IDs.
- Each algorithm gets the same train/validation/test rows, task adapter, metric, seed set, and declared training budget. Architecture search is allowed to use validation only. Report both the selected validation score and its one-time test score.

The runner includes all 50 vanilla architectures as separate results as well as the aggregate `MLP-Fixed` selector. For generic ADP IDs, faithful and corrected profiles produce separate result labels. Thus each retained fold/seed runs 22 canonical/profile-specific ADP configurations (13 adaptive IDs plus the fixed-grid selector, with generic IDs represented under both profiles) and 50 individual vanilla candidates.

## Feasibility and staged execution

The full matrix is large: 72 CC18 + 36 TabZilla + 35 CTR23 tasks by default, multiplied by outer folds, model seeds, profile-specific ADP configurations, and the 50 vanilla candidates. The 22-architecture screen (44 training runs at two seeds) has a cost, but avoids spending the adaptive-search budget on saturated or complexity-insensitive datasets. Start with a pilot subset spanning binary/multiclass and regression, numeric/mixed feature types, and small/medium row counts. Use the pilot to measure cost and catch unsupported data cases; do not use its test scores to change method settings. Then run the full suite with settings frozen. Consider an explicit, shared time/epoch budget per method-task-seed and record any truncation. The 50 vanilla candidates can be expensive, particularly after one-hot expansion; retain all candidates in the defined comparison and make budget-related early termination visible rather than silently dropping wide architectures.

Before launch, snapshot the downloaded task metadata and split indices, verify licenses/attributions for each dataset, and persist a manifest containing suite ID, task ID, dataset ID/name, target, row/feature counts, feature types, class counts or target range, outer split IDs, inner split seed, model seeds, and checksums. Pin data and package versions so reruns use the same inputs.

## Sources

- OpenML-CC18 suite and inclusion criteria: <https://docs.openml.org/benchmark/>; suite ID 99.
- OpenML suite tasks carry dataset, target, and train/test split definitions: <https://docs.openml.org/benchmark/>.
- OpenML-CTR23 suite: <https://www.openml.org/search?id=353&study_type=run&type=study>; suite ID 353, 35 regression tasks.
- CTR23 paper and split policy: <https://openreview.net/pdf?id=HebAOoMm94>.
- TabZilla dataset selection and recommended ten folds/validation splits: <https://github.com/naszilla/tabzilla/blob/main/docs/DATASHEET.md>.
- TabZilla benchmark paper and difficulty-selection rationale: <https://proceedings.neurips.cc/paper_files/paper/2023/file/f06d5ebd4ff40b40dd97e30cee632123-Paper-Datasets_and_Benchmarks.pdf>.
- OpenML Python task API for loading task data and exact repeat/fold/sample split indices: <https://openml.github.io/openml-python/latest/examples/Basics/simple_tasks_tutorial/> and <https://docs.openml.org/reference/tasks/>.
- UCI Machine Learning Repository for future explicitly curated additions: <https://archive.ics.uci.edu/>.
