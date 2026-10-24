# Alarm logs of industrial packaging machines: individual screen

Official record: [University of Padua Research Data, record 1031](https://researchdata.cab.unipd.it/1031/). The record says the data archive is visible to anyone, 8 MB, and CC BY 4.0. An anonymous full GET succeeded on 2026-10-23 (8,517,314 bytes); the ZIP passed integrity validation. Archive SHA-256: `72f556ef6742a1a64472b96d2254c3645be255254cbe4852a08e1ff79ffcbcac`. The archive's raw alarm CSV was extracted and parsed (13,180,280 bytes; SHA-256 `53bd4414a6fb5b6875a9535f1be622dfb2dfba69de407d071a52d0d304160d1a`).

## Data and MLP task

The raw CSV has 444,834 timestamped rows, 20 machine serials, and 154 distinct alarm codes, spanning 2019-02-21 through 2020-06-17. It is directly convertible into fixed-width numeric inputs and supervised labels for a fully connected MLP. The official source describes next-alarm forecasting, future-window alarms, and future alarm sequences as supervised tasks. The supplied NPZ was inspected; its actual shapes are `x_train=(41045,109)`, `y_train=(41045,556)`, `x_test=(17955,109)`, `y_test=(17955,556)`, contradicting the README's stated output width 109. The pilot therefore used raw CSV and an independently specified task.

The screened event task is: sum each alarm code over the preceding six complete hours (154 inputs) and predict whether any alarm occurs in the next hour (binary output). Hourly windows stay within each machine. Machines are split, not rows: train `[0,2,5,6,7,8,10,12,13,14,15,16]` (119,484 windows; 35.39% positive), validation `[1,4,18,19]` (25,761; 16.69%), and test `[3,9,11,17]` (31,271; 36.47%). Input means/standard deviations use train rows only. Each epoch is one full training batch.

## Full-batch CUDA screen

On the RTX 3050 Laptop 6 GiB GPU, plain `TabularMLP` models trained for 30 epochs with Adam `1e-3`, cross-entropy, one seed per architecture, and validation-loss checkpoint selection. Accuracy / balanced accuracy / macro-F1 / positive-event F1 on held-out machines were:

| Hidden widths | Accuracy | Balanced accuracy | Macro-F1 | Event F1 | Peak CUDA memory |
|---|---:|---:|---:|---:|---:|
| `[16,16]` | .6801 | .5818 | .5610 | .3324 | 0.149 GiB |
| `[128,128]` | .7651 | .7263 | .7344 | .6442 | 0.349 GiB |
| `[512,512,512,512,512]` | .7703 | .7403 | .7457 | .6666 | 1.737 GiB |

The MLP scores are not near saturation, and widening improves held-out metrics; the largest tested network fits the full batch comfortably. However, the validation group is much less event-dense than train/test. On the test machines, predicting “alarm next hour” whenever the previous hour contained any alarm gives accuracy .8491, balanced accuracy .8372, and event F1 .7932—better than every MLP. The six-hour persistence rule scores .7980/.8324/.7761. Thus this is a difficult, non-saturated stress route with a strong trivial baseline, not a demonstrated strong neural forecast.

**Disposition: KEEP PROVISIONALLY for the one-hour supervised event-occurrence task.** The record passes no-gate retrieval, CC BY 4.0, plain-MLP task shape, train-only preprocessing, group holdout, full-batch resource, and non-saturation checks. Compare every model against the previous-hour persistence baseline and report balanced accuracy/macro-F1/event-F1; keep the group-rate shift visible. The official next-alarm-type and multi-label tasks need separate conversion and pilots. The supplied fixed-sequence target mismatch must be resolved before using those routes. No repository provider integration, ADP configuration, repeated seed, or remaining vanilla grid has been tested.

Exact machine assignments, metrics, hashes, and caveats are in [`UNIPD_PACKAGING_ALARMS_SCREEN.json`](UNIPD_PACKAGING_ALARMS_SCREEN.json). The downloaded payload and pilots are in `/tmp/simple-adp-packaging-audit`, outside the repository.
