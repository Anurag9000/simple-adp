# ODDS supervised anomaly dataset screen

**Date:** 2026-10-24. **Scope:** each of the 24 binary-labeled ODDS MAT routes exposed by `ODDSProvider`, screened for this repository's numeric tabular feedforward MLP and supervised anomaly task. This is an audit of the named ODDS candidates, not a census of every internet dataset. Novelty detection is not included.

## Decision

**Keep only `satellite` provisionally**, under the explicit task definition “Landsat classes 2, 4, and 5 versus the other classes.” It is not a naturally observed anomaly label. It has a no-account payload, source-backed CC BY 4.0 rights, valid dense numeric data, no exact duplicate test/train feature rows, enough examples, full-batch capacity, and a useful unsaturated MLP result with an architecture gain. Cross out the other 23 routes for the individually recorded reasons below.

## How each gate was checked

- **Architecture/task fit:** every candidate is a fixed-width numeric matrix with binary labels (Lympho is the exception at source: its categorical features were training-fitted one-hot encoded). The actual `ODDSProvider.load_task` checks X/y alignment, feature count, finite inputs, both labels, row counts, and positive counts. A plain dense `TabularMLP` and the repository's `SupervisedAnomalyTask` were used where the data/license gates permitted training.
- **Anonymous access:** exact ODDS MAT payloads were fetched without credentials through the repository provider into temporary directories. Twenty were acquired as part of the common pilot, HTTP separately, and MNIST, Speech, and Mammography separately; those last three also passed provider parsing but did not proceed to training because the reuse-rights gate was unresolved. Temporary payloads were removed at process exit. The full ODDS MAT acquisition is distinct from checking original-source URLs: source-page or endpoint probes do not imply that every original-source archive was fully downloaded. The access manifests distinguish HTTP probes, range responses, and completed transfers.
- **Reuse terms:** the ODDS MAT pages do not state a separate license. For source-backed transformations, the original-source terms must be followed. The 20 trained routes plus HTTP map to UCI records that state CC BY 4.0, including KDD Cup 1999. MNIST, VUT Brno Speech, and OpenML Mammography 310 did not have a clearly compatible explicit open license in the checked primary records and were not trained. Public availability alone was not treated as an open license.
- **Difficulty:** repository stratified split with seed 42 (60/20/20); training-only standardization; training-fitted one-hot for Lympho; full-batch Adam, learning rate 1e-3, 30 epochs, validation-loss model selection; representative hidden widths `[128,128]` and `[512,512,512,512,512]`; RTX 3050 6 GB. Results are a screening pilot, not the full vanilla grid or all ADP models. Metrics are from the test split at the selected epoch.
- **Leakage:** exact float32 feature-row equality was checked across the seeded train/validation/test split. This is not a near-duplicate, patient, source-group, or repeated-time audit.
- **Resources:** HTTP's largest MLP trained full-batch at 4.59 GiB peak CUDA allocation. Other successful candidates also completed their full-batch pilot. Dataset-specific peak allocations are in the JSON pilot manifests.

## Individual outcomes

“Small → large” reports test balanced accuracy / PR-AUC for `[128,128]` → `[512]*5`; “test overlap” is the number of test rows whose exact feature vector also appears in train, out of test rows.

| Route | Shape and positives | Rights and anonymous access | Pilot or gate evidence | Decision |
|---|---|---|---|---|
| cardio | 1,831 × 21; 176 | UCI Cardiotocography CC BY 4.0; MAT and UCI artifact fetched anonymously | .936/.933 → .929/.937; overlap 0/366; widening does not improve balanced accuracy | **Cross:** small and large models already perform strongly. |
| thyroid | 3,772 × 6; 93 | UCI Thyroid CC BY 4.0; anonymous MAT; source endpoint checked | .963/.862 → .963/.853; overlap 22/753 | **Cross:** near-ceiling ranking/balanced accuracy; duplicate rows cross splits. |
| satimage-2 | 5,803 × 36; 71 | UCI Landsat CC BY 4.0; anonymous MAT; source endpoint checked | .919/.878 → .923/.843; overlap 1/1,160; majority accuracy .988 | **Cross:** severe imbalance, strong ROC scores, little useful width gain. |
| lympho | 148 × 59 one-hot; 6 | UCI Lymphography CC BY 4.0; anonymous MAT; source endpoint checked | .500/.077 → .500/.034; overlap 0/29; neither model detects test anomalies | **Cross:** six positives cannot support stable benchmark results. |
| glass | 214 × 9; 9 | UCI Glass CC BY 4.0; anonymous MAT; source endpoint checked | .939/.250 → .988/.500; overlap 0/42 | **Cross:** nine total positives make results too sample-sensitive. |
| vowels | 1,456 × 12; 50 | UCI Japanese Vowels CC BY 4.0; anonymous MAT; source endpoint checked | .875/.761 → .893/.896; overlap 1/291 | **Cross:** high ranking performance; labels select speaker identity rather than general anomaly behavior. |
| musk | 3,062 × 166; 97 | UCI Musk v2 CC BY 4.0; anonymous MAT; source endpoint checked | 1.000/1.000 → 1.000/1.000; overlap 0/612 | **Cross:** both MLPs saturate completely. |
| letter | 1,600 × 32; 100 | UCI Letter Recognition CC BY 4.0; anonymous MAT; source endpoint checked | .747/.293 → .765/.494; overlap 0/320 | **Cross:** ODDS constructs paired/concatenated artificial outliers; held-out skill is weak. |
| satellite | 6,435 × 36; 2,036 | UCI Landsat CC BY 4.0; anonymous MAT; source endpoint checked | .820/.894 → .881/.932; overlap 0/1,286; small-to-large gain | **Keep provisionally:** unsaturated with adequate sample support and no exact duplicate leakage. Report only as the explicit class-grouping task above. |
| shuttle | 49,097 × 9; 3,511 | UCI Statlog Shuttle CC BY 4.0; anonymous MAT; source endpoint checked | .988/.984 → .993/.993; overlap 0/9,819 | **Cross:** near-saturated with the small MLP. |
| breastw | 683 × 9; 239 | UCI Breast Cancer Wisconsin Original CC BY 4.0; anonymous MAT; source endpoint checked | .946/.988 → .935/.991; overlap 55/135 (40.7%) | **Cross:** diagnosis class is not an anomaly task; test/train overlap invalidates the split. |
| arrhythmia | 452 × 274; 66 | UCI Arrhythmia CC BY 4.0; anonymous MAT; source endpoint checked | .748/.693 → .760/.695; overlap 0/90; roughly 13 positives in test | **Cross:** small, high-dimensional cohort with unstable support and weak MLP skill. |
| ionosphere | 351 × 33; 126 | UCI Ionosphere CC BY 4.0; anonymous MAT; source endpoint checked | .958/.980 → .884/.957; overlap 0/70 | **Cross:** near-ceiling small-model result; wider model degrades balanced accuracy. |
| mnist | 7,603 × 100; 700 | Anonymous MAT download and provider parse passed; no explicit compatible original-source reuse license established | No pilot because rights gate failed | **Cross:** unresolved reuse terms. |
| optdigits | 5,216 × 64; 150 | UCI Optical Digits CC BY 4.0; anonymous MAT; source endpoint checked | .998/.995 → .983/.995; overlap 7/1,043 | **Cross:** effectively saturated. |
| http | 567,498 × 3; 2,211 | UCI KDD Cup 1999 CC BY 4.0; anonymous MAT; official KDD raw endpoint returned HTTP 206 | .998/.993 → .999/.988; 4.59 GiB peak; overlap 78,594/113,499 (69.2%) | **Cross:** fits capacity, but extensive exact duplicate leakage makes test performance unreliable. |
| smtp | 95,156 × 3; 30 | UCI KDD Cup 1999 CC BY 4.0; anonymous MAT; official KDD raw endpoint returned HTTP 206 | .732/.501 → .872/.503; overlap 5,831/19,031 (30.6%) | **Cross:** only 30 positives, weak PR-AUC, and major exact duplicate leakage. |
| mammography | 11,183 × 6; 260 | Anonymous MAT download and provider parse passed; OpenML 310 says only “Public” | No pilot because rights gate failed | **Cross:** “Public” does not establish an open reuse license. |
| annthyroid | 7,200 × 6; 534 | UCI Thyroid CC BY 4.0; anonymous MAT; source endpoint checked | .825/.659 → .967/.890; overlap 37/1,439 | **Cross:** wide model is near-ceiling; duplicate rows cross the split. |
| pendigits | 6,870 × 16; 156 | UCI PenDigits CC BY 4.0; anonymous MAT; source endpoint checked | .968/.925 → .967/.998; overlap 0/1,373 | **Cross:** large model nearly saturates PR ranking without a balanced-accuracy gain. |
| ecoli | 336 × 7; expected 9 | UCI Ecoli CC BY 4.0; anonymous MAT downloaded | Provider parse rejects target: all 336 labels are zero | **Cross:** artifact violates its binary-label and expected-positive-count contract. |
| wine | 129 × 13; 10 | UCI Wine CC BY 4.0; anonymous MAT; source endpoint checked | .978/1.000 → 1.000/1.000; overlap 0/25 | **Cross:** tiny sample and complete large-model saturation. |
| vertebral | 240 × 6; 30 | UCI Vertebral Column CC BY 4.0; anonymous MAT; source endpoint checked | .726/.433 → .714/.636; overlap 0/48 | **Cross:** few cases/positives, weak held-out skill, and no reliable width benefit. |
| speech | 3,686 × 400; 61 | Anonymous MAT download and provider parse passed; VUT Brno source reuse terms unresolved | No pilot because rights gate failed | **Cross:** unresolved license for reuse. |

There are 24 routes; the table includes all 24 (the order follows the provider catalog except the three license-gated routes are shown where their identities fit). Only Satellite passes this screen. Do not infer that other datasets or task families have passed because this ODDS subset did.

## Audit artifacts and primary sources

- [`ODDS_20_FULLBATCH_MLP_PILOTS_20261024.json`](data_audit_manifests/ODDS_20_FULLBATCH_MLP_PILOTS_20261024.json): 20 source-backed routes, exact split, hashes, scores, and peak allocations.
- [`ODDS_HTTP_FULLBATCH_MLP_PILOT_20261024.json`](data_audit_manifests/ODDS_HTTP_FULLBATCH_MLP_PILOT_20261024.json): HTTP full-batch pilot.
- [`ODDS_DUPLICATE_SPLIT_AUDIT_20261024.json`](data_audit_manifests/ODDS_DUPLICATE_SPLIT_AUDIT_20261024.json) and [`ODDS_HTTP_DUPLICATE_SPLIT_AUDIT_20261024.json`](data_audit_manifests/ODDS_HTTP_DUPLICATE_SPLIT_AUDIT_20261024.json): exact feature-row split overlap.
- [`ODDS_ORIGINAL_SOURCE_RIGHTS_20261024.json`](data_audit_manifests/ODDS_ORIGINAL_SOURCE_RIGHTS_20261024.json), [`ODDS_ORIGINAL_SOURCE_ACCESS_20261024.json`](data_audit_manifests/ODDS_ORIGINAL_SOURCE_ACCESS_20261024.json), [`ODDS_LICENSE_UNRESOLVED_ACCESS_20261024.json`](data_audit_manifests/ODDS_LICENSE_UNRESOLVED_ACCESS_20261024.json): source-level terms and retrieval probes/full downloads.
- [ODDS project](https://odds.cs.stonybrook.edu/); representative original-source records: [Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography), [Landsat Satellite](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite), and [KDD Cup 1999](https://archive.ics.uci.edu/dataset/130/kdd+cup+1999+data). The UCI records state CC BY 4.0. These citations establish ODDS/source context and rights; route-specific source IDs and probe data are in the rights/access manifests.

## Limits

The two pilot sizes and single seed screen saturation and rough skill; they do not prove performance for all widths, depths, random seeds, or ADP controllers. Duplicate detection checks exact feature equality only. No full ADP experiment matrix was run. Open licensing is reported from the source records and must still be honored, including attribution; ODDS's own MAT files do not state a separate license.
