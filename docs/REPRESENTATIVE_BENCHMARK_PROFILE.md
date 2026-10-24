# Representative one-dataset-per-task benchmark profile

## Purpose

The default `simple-adp-suite` comparison uses one fixed, literature-backed dataset for each task family currently supported by a registered provider. Every selected dataset is shared by the ADP variants and vanilla MLPs. This keeps the first coordinated comparison interpretable and makes it practical to launch as one suite. It is a representative profile, not a claim that one dataset can establish generalization across a task family.

The original full catalogs remain available. Pass explicit `--dataset-suites` selectors to request them. OpenML task IDs are pinned below; provider selectors may use `@task_id` to choose one task from a provider suite.

## Selected datasets

| Task family | Selected dataset and task | Why this is the representative | Registered task variants |
|---|---|---|---|
| Tabular classification | **MiniBooNE**, OpenML task `168335` in TabZilla (dataset ID `41150`) | TabZilla is the paper's deliberately hard classification suite: it selects cases where simple baselines and many evaluated algorithms do not reliably reach top performance. MiniBooNE is a sizable, numeric, binary task in that suite, making it a suitable single stress-test representative for a dense MLP. The paper does not identify MiniBooNE as the uniquely hardest of its 36 tasks. | classification |
| Tabular regression | **Superconductivity**, OpenML task `361242` in CTR23 | CTR23 is a curated regression benchmark designed to distinguish methods of varying complexity and excludes tasks exactly solvable by a linear model. Superconductivity is a standard, nonlinear regression task in that suite with enough rows and features to exercise the MLP architectures. The benchmark paper does not designate it as the unique hardest task. | regression |
| Time-series forecasting | **Monash Australian Electricity Demand**, horizon `28` | This is the currently integrated Monash artifact, a named series-level forecasting benchmark. Horizon 28 is selected from the provider's three supported horizons (1, 7, 28) as the longest prediction horizon. This is a profile choice, not a paper-reported ranking that it is the hardest Monash dataset. | forecasting regression |
| Sequence prediction / per-step labeling | **UCI HAPT** | This is the integrated labeled multichannel activity sequence source. The implemented task is fixed-window, per-step activity labeling with subject-disjoint splits; it is not next-step numeric forecasting. | sequence labeling / classification |
| Event / beat detection | **MIT-BIH Arrhythmia Database** | It is a long-established annotated ECG benchmark, is available through the registered PhysioNet Open Access provider, and supports supervised beat-centered event classification. The provider maps beats to AAMI classes and splits by record. | event/beat classification |
| Recommendation: ratings, preference, and ranking | **MovieLens 1M** | It is the larger of the two currently integrated GroupLens MovieLens releases and is widely used in recommendation evaluations. The same source supports all three implemented objectives, so one dataset is selected for this task family. | rating regression; preference classification using rating >= 4; grouped pairwise ranking using the same rating-derived relevance |
| Supervised anomaly classification | **ODDS Annthyroid** | Annthyroid is a commonly used tabular anomaly benchmark in the registered ODDS catalog. The repo's one-seed full-batch screen found the deeper `[512,512,512,512,512]` MLP improved test balanced accuracy and macro-F1 over `[128,128]`; that is pilot evidence, not a multi-seed or ADP result. | binary supervised anomaly classification |

## Default runner selectors

The default coordinated suite uses these selectors:

```text
openml:tabzilla@168335
openml:ctr23@361242
monash:aus-electricity-demand@monash-aus-electricity-demand-h28
uci_sequence:hapt@uci-hapt-per-step-activity
physionet_ecg:mitdb@mitdb-aami-beat-classification
movielens:ml-1m
odds:annthyroid@odds-annthyroid-supervised-anomaly
```

MovieLens `ml-1m` expands to three registered task variants because it is a single source dataset used for rating, rating-derived preference, and ranking. Every other selector expands to exactly one provider task. `SuitePlan` rejects a selector that matches zero or multiple tasks.

## Scope and interpretation

- “Hard” is evidence-backed only where the cited benchmark itself defines a hard suite (TabZilla) or filters for nonlinear regression (CTR23). No paper ranks a unique hardest dataset for each task family, so this profile does not claim that.
- The Monash choice is the most challenging horizon currently exposed for its integrated Australian Electricity artifact, not a claim that it is harder than other Monash datasets that are not registered as providers.
- The HAPT implementation is sequence labeling. It does not currently implement a separate next-step sequence-forecasting objective.
- MovieLens preference and ranking targets are derived from explicit ratings. They are not click/exposure logs or relevance judgments from a search engine.
- ODDS provides derived anomaly-labeled MAT artifacts. The provider records the artifact and original-source terms separately; this profile does not claim every derived artifact has a separate permissive license.
- The central runner's full-catalog mode remains explicit. `simple-adp-benchmark` also retains its separate all-task OpenML screening workflow; this profile changes the default for the coordinated multi-task runner only.

## Provenance

- MiniBooNE is a TabZilla task (OpenML study 379, task 168335). The TabZilla paper describes the suite as the hardest selected datasets from its evaluated pool: [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f06d5ebd4ff40b40dd97e30cee632123-Paper-Datasets_and_Benchmarks.pdf), [official datasheet](https://github.com/naszilla/tabzilla/blob/main/docs/DATASHEET.md).
- Superconductivity is CTR23 task 361242 in OpenML study 353. CTR23's criteria and rationale are described in its [paper](https://openreview.net/pdf?id=HebAOoMm94).
- Australian Electricity Demand's source/provider contract is in `src/simple_adp/runner/monash_datasets.py`.
- HAPT, MIT-BIH, MovieLens, and Annthyroid provider contracts are in `src/simple_adp/runner/sequence_datasets.py`, `physionet_datasets.py`, `public_datasets.py`, and `odds_datasets.py`, respectively.
