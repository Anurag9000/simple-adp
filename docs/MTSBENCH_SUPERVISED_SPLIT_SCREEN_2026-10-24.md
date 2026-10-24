# mTSBench screening for supervised time-series anomaly classification

**Screen date:** 2026-10-24. **Scope:** all 19 dataset configurations in the public mTSBench release. This is a candidate source screen for the repository's fixed-width tabular MLP and supervised binary anomaly task; novelty detection is excluded.

## Finding

**Cross out all 19 mTSBench configurations for this supervised task.** The benchmark's data summary lists 716 split-file records. All **349 training segments across the 19 configurations contain zero point anomalies and zero anomalous sequences**. Validation and test partitions do contain anomalies (3,248/16,962 anomalous points and 4,769/28,481 anomalous sequences, respectively), but they cannot supply the positive training class without changing the published split. That makes the official release an unsupervised detector/model-selection benchmark, not a supervised anomaly-classification dataset. A one-class/normal-only model would be novelty or unsupervised detection, which is outside the requested scope.

The data are structurally compatible with a dense MLP as timestamped numeric feature rows plus binary `is_anomaly`. Structural compatibility does not fix the missing positive training labels. I did not run MLPs because the task cannot train under the official split. This is a source-level rejection, not an empirical claim that the model saturates.

## Access and rights

The unauthenticated Hugging Face dataset API returned HTTP 200 and reports `private=false`, `gated=false`, `disabled=false`. An anonymous full GET of `CalIt2/CalIt2_traffic_train.csv` returned 24,678 bytes, parsed as 940 rows with header `timestamp,in_count,out_count,is_anomaly`, and contained no positive anomaly label. The repository tree advertises 724 files totaling 4,149,573,739 bytes; the full 4.15 GB collection was not downloaded. The hub's dataset card declares **CC BY-NC 4.0** for the aggregate release, so it does not meet the strict open-license criterion used in this audit. The project's per-source table also identifies distinct original licenses; the row-by-row labels are captured below. An original source's own permissive terms do not turn the rehosted CC BY-NC aggregate into an unrestricted release, though those original datasets can be considered separately under their own acquisition path.

## Individual configurations

All entries fail the same decisive supervised-label gate; this table ensures that no configuration was silently inferred from another. “Train segments” counts file-level training partitions, and the channel dimension is from the source manifest. Each training file has zero anomalous points and sequences in the published summary.

| Config | Feature channels | Train segments | Train rows | Original-source license stated in project | Decision |
|---|---:|---:|---:|---|---|
| CalIt2 | 3 | 1 | 940 | CC BY 4.0 | **Cross:** training labels contain zero anomalies. |
| cicids | 73 | 6 | 278,722 | Citation required | **Cross:** training labels contain zero anomalies. |
| creditcard | 30 | 1 | 56,804 | Citation required | **Cross:** training labels contain zero anomalies. |
| Daphnet | 10 | 26 | 137,565 | CC BY 4.0 | **Cross:** training labels contain zero anomalies. |
| Exathlon | 21 | 30 | 164,153 | Apache-2.0 | **Cross:** training labels contain zero anomalies. |
| GECCO | 10 | 1 | 27,560 | Citation required | **Cross:** training labels contain zero anomalies. |
| Genesis | 19 | 1 | 3,244 | CC BY-NC-SA 4.0 | **Cross:** training labels contain zero anomalies. |
| GHL | 17 | 14 | 1,645,137 | Contact authors | **Cross:** training labels contain zero anomalies. |
| GutenTAG | 21 | 30 | 30,000 | MIT | **Cross:** training labels contain zero anomalies. |
| metro | 6 | 1 | 24,072 | CC BY 4.0 | **Cross:** training labels contain zero anomalies. |
| MITDB | 3 | 47 | 5,415,890 | ODC-By 1.0 | **Cross:** training labels contain zero anomalies. |
| MSL | 56 | 26 | 57,878 | BSD-3-Clause | **Cross:** training labels contain zero anomalies. |
| room-occupancy | 6 | 2 | 2,565 | CC BY 4.0 | **Cross:** training labels contain zero anomalies. |
| OPPORTUNITY | 33 | 13 | 67,777 | CC BY 4.0 | **Cross:** training labels contain zero anomalies. |
| PSM | 27 | 1 | 132,481 | CC BY 4.0 | **Cross:** training labels contain zero anomalies. |
| SMAP | 26 | 52 | 136,202 | BSD-3-Clause | **Cross:** training labels contain zero anomalies. |
| SMD | 39 | 18 | 436,573 | MIT | **Cross:** training labels contain zero anomalies. |
| SVDB | 3 | 78 | 3,102,197 | ODC-By 1.0 | **Cross:** training labels contain zero anomalies. |
| swan | 39 | 1 | 19,672 | MIT | **Cross:** training labels contain zero anomalies. |

## Evidence and limits

The parsed official `data_summary.csv`, anonymous API response, full tree metadata, CalIt2 sample download, and per-configuration counts are retained in [`MTSBENCH_SUPERVISED_SPLIT_SCREEN_20261024.json`](data_audit_manifests/MTSBENCH_SUPERVISED_SPLIT_SCREEN_20261024.json). The source project describes mTSBench as a benchmark for anomaly detection/model selection without labels available to the selector and documents its train/validation/test CSV layout ([project page](https://plan-lab.github.io/projects/mtsbench/), [dataset card](https://huggingface.co/datasets/PLAN-Lab/mTSBench)). The data summary is a metadata inventory, not a full payload parse of every split file. Only the CalIt2 train CSV was fully downloaded as a payload validation sample. No MLP difficulty pilot was run because every official train segment is single-class.
