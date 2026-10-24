# OpenSTEF Liander 2024: individual 55-target MLP screen

**Audit date:** 2026-10-23. **Pinned source revision:** `ec28635e20a12927f3323a2dc18a0719a361b56d`.

All 55 load-measurement Parquet files were anonymously downloaded from the pinned public Hugging Face release (40,371,354 bytes total), SHA-256 hashed and parsed. Load-only targets use Liander open-data terms with attribution; the separately licensed weather and price tables were not needed for this screen, and consumption profiles were omitted because their upstream license is not stated. The exact per-file hashes and row counts are in the accompanying JSON/CSV manifests.

## Task and screen protocol

- One task per target location: 96 observed 15-minute load values (24 hours) to the next 96 load values (24 hours), flattened into a 96-input / 96-output `TabularMLP` regression. No recurrent, convolutional, or attention layers.
- Per-target official `train_start` to `benchmark_start` dates define development and benchmark test periods. The final 20% of the pre-benchmark span is chronological validation; the preceding 80% is training. Sliding train origins are hourly; validation/test origins are every four hours.
- Windows with missing loads or timestamp gaps are dropped. Mean/std are fitted on the training interval only. Each epoch uses the entire target training partition in one batch.
- CUDA RTX 3050 6 GB, MSE, Adam `1e-3`, 30 epochs, one fixed seed per target. Two vanilla architectures: `[16,16]` and `[128,128]`. Persistence repeats the latest observed value for all 96 outputs.
- This is a bounded vanilla difficulty screen; it does not validate all ADP variants, the full vanilla grid, multiple seeds, or pooled 55-target training.

## Aggregate result

| Targets | 55 |
|---|---:|
| Full-batch MLP pilots | 52 |
| Provisional, non-saturated targets | 51 |
| Crossed out | 4 |
| Wider MLP improved test R² | 52 / 52 pilots |
| Wider MLP beat persistence MAE | 42 / 52 pilots |
| Median test R² `[16,16]` | -0.4005 |
| Median test R² `[128,128]` | 0.1522 |
| Best test R² across tested MLPs | 0.4199 |

The tested models are far from ceiling, and width changes results consistently; absolute forecast performance remains weak. Keep these as provisional stress routes only. “Non-saturated” here describes the tested split, 24-hour horizon, preprocessing, seed, architectures and 30-epoch budget.

## Per-target results

| Target | Grid group | Train / val / test windows | Test R² `[16,16]` | Test R² `[128,128]` | Wide MAE skill vs persistence | Disposition |
|---|---|---:|---:|---:|---:|---|
| Within Stadsregio Arnhem Nijmegen_normalized | wind_park | 1105 / 67 / 1819 | -0.6270458042544347 | 0.040564867456830966 | -0.18222432866872862 | KEEP provisional |
| Within 15 kilometers of Opmeer_normalized | wind_park | 1105 / 67 / 1819 | -0.4069854816361933 | 0.09644686865363206 | -0.20396377861478654 | KEEP provisional |
| Within 20 kilometers of Leeuwarden_normalized | wind_park | 1105 / 67 / 1819 | -0.48122850843045306 | 0.17518820727477258 | -0.07162172831787283 | KEEP provisional |
| Within 15 kilometers of Alphen aan den Rijn_normalized | wind_park | 1105 / 67 / 1819 | -0.39391605169854316 | 0.18154852220304507 | -0.05549376522618621 | KEEP provisional |
| Within 15 kilometers of Dronten_normalized | wind_park | 1105 / 67 / 1819 | -0.4292987630529652 | 0.17265794905573206 | -0.07693967713217287 | KEEP provisional |
| OS Apeldoorn | transformer | 1105 / 67 / 1819 | -0.7179953439912556 | 0.11650675246167752 | 0.2134165613850334 | KEEP provisional |
| OS Middenmeer | transformer | 1105 / 67 / 1819 | -0.6283917383970705 | -0.0453668247155421 | -0.29891236211978023 | KEEP provisional |
| RS Grouw (Grou) | transformer | 1105 / 67 / 1819 | -0.5410442572958944 | 0.41994573591868 | 0.40247534695035536 | KEEP provisional |
| RS Wieringerwerf | transformer | 1105 / 67 / 1819 | 0.00480688317770972 | 0.013049857546473365 | 0.12460418572589815 | KEEP provisional |
| OS Lelystad | transformer | 1105 / 67 / 1819 | -0.05683016832391319 | 0.31638705064511685 | 0.31647351421261916 | KEEP provisional |
| RS Poederoijen | transformer | 1105 / 67 / 1819 | -0.27917020313112606 | 0.16775094622468967 | 0.29614734206364746 | KEEP provisional |
| OS Tiel | transformer | 1105 / 67 / 1819 | -0.17039051071581168 | 0.20039170771019899 | 0.22675099011849797 | KEEP provisional |
| OS Doetinchem | transformer | 1105 / 67 / 1819 | -0.23112624284702132 | 0.19774892696862556 | 0.20231612748915184 | KEEP provisional |
| OS Oterleek | transformer | 1105 / 67 / 1819 | -0.18738466045748892 | 0.3910760139182483 | 0.32763802897256744 | KEEP provisional |
| OS Nijmegen | transformer | 1105 / 67 / 1819 | -0.24944297744908606 | 0.4160888439696627 | 0.3910548990592586 | KEEP provisional |
| OS Amsterdam Hemweg | transformer | 1105 / 67 / 1819 | -0.5752273750687926 | 0.4154751379391728 | 0.2143314688960306 | KEEP provisional |
| RS Surhuisterveen | transformer | 1105 / 67 / 1819 | -2.1816501563301274 | -1.191537229933926 | -0.7696785118619431 | KEEP provisional |
| OS Ede | transformer | 1105 / 67 / 1819 | -0.5080904765817562 | 0.23962963641491009 | 0.2539229051128278 | KEEP provisional |
| RS Roodwillingen (Duiven) | transformer | 1105 / 67 / 1819 | 0.010600627186919453 | 0.11655667744622022 | 0.15911399237616242 | KEEP provisional |
| RS Minnertsga | transformer | 1105 / 67 / 1819 | -7.098321005259367 | -0.37223213073824324 | -0.2677900356483107 | KEEP provisional |
| OS Naarden | mv_feeder | 1105 / 67 / 1819 | -1.7965191533588318 | -0.7337588849377703 | -0.2700456198771086 | KEEP provisional |
| OS Leiden Noord | mv_feeder | 1105 / 67 / 1819 | -0.413474339806726 | 0.15902060841059518 | 0.21613206618950254 | KEEP provisional |
| SS Harlingen | mv_feeder | 1105 / 67 / 1819 | -0.4850067242317633 | -0.13665683062945955 | 0.17203363212412648 | KEEP provisional |
| OS Eibergen | mv_feeder | 1105 / 67 / 1819 | -0.6480328206005175 | 0.10104596167962865 | 0.12294269922408219 | KEEP provisional |
| OS Sneek | mv_feeder | 1105 / 67 / 1819 | -0.046949447197269345 | 0.17426836504817034 | 0.25224583056389316 | KEEP provisional |
| OS Weesp | mv_feeder | 1105 / 67 / 1819 | -0.8616473638737723 | 0.10885356000209379 | 0.1560829065543311 | KEEP provisional |
| SS Ureterp | mv_feeder | 1105 / 67 / 1819 | -0.5674606967034781 | -0.3224297382536052 | 0.08067244241798788 | KEEP provisional |
| OS Gorredijk | mv_feeder | 1105 / 67 / 1819 | -0.523683701052523 | 0.15016214189312516 | 0.19354240915551713 | KEEP provisional |
| OS Sassenheim | mv_feeder | 1105 / 67 / 1819 | -0.42250440430018177 | 0.15278943000223555 | 0.23432778982700964 | KEEP provisional |
| OS Waarderpolder | mv_feeder | 1105 / 67 / 1819 | -0.7374749987066589 | -0.030315616106353804 | 0.06716600172628862 | KEEP provisional |
| OS Edam | mv_feeder | 1105 / 67 / 1819 | -0.6560682619474678 | 0.27074348135249127 | 0.2482744265936916 | KEEP provisional |
| OS Watergraafsmeer | mv_feeder | 1105 / 67 / 1819 | 0.019979157943294745 | 0.14600930530901246 | 0.09087291389359542 | KEEP provisional |
| RS Angerlo | mv_feeder | 1105 / 67 / 1819 | -0.48628050679611 | 0.050472743411371535 | 0.19562127276766772 | KEEP provisional |
| SS Heerenveen Pim Mulier | mv_feeder | 1105 / 67 / 1819 | -0.37770301332749434 | 0.09084917907590673 | 0.12057885759189646 | KEEP provisional |
| RS Roodwillingen (Duiven) | mv_feeder | 1105 / 67 / 1819 | -0.3100391300406329 | 0.06540603188468885 | 0.1945029396868706 | CROSS OUT: pending source filename/target identity reconciliation |
| Within 5 kilometers of Leeuwarden_normalized | solar_park | 1105 / 67 / 1819 | -0.16700903822701196 | 0.24077114463632887 | 0.36474230360615323 | KEEP provisional |
| Within 15 kilometers of Oosterwolde_normalized | solar_park | 1105 / 67 / 1819 | -0.20065943731384372 | 0.22073745504291975 | 0.3489935279598898 | KEEP provisional |
| Within 15 kilometers of Zutphen_normalized | solar_park | 1105 / 67 / 1819 | -0.14052066099543636 | 0.3482158424187505 | 0.40494769188448887 | KEEP provisional |
| Within 20 kilometers of Rotterdam_normalized | solar_park | 1105 / 67 / 1819 | -0.12926884805469085 | 0.23508522509873675 | 0.3725403035524274 | KEEP provisional |
| Within 10 kilometers of Westwoud_normalized | solar_park | 1105 / 67 / 1819 | -0.1712492194125994 | 0.2668891492168729 | 0.3657332998192727 | KEEP provisional |
| OS Apeldoorn | station_installation | 1105 / 67 / 1819 | -0.6936323417119659 | 0.07881200769802466 | 0.1959046593249789 | KEEP provisional |
| RS Hallum | station_installation | 1105 / 67 / 1819 | -0.01224767277129346 | 0.09100138505606348 | 0.17845707398545618 | KEEP provisional |
| OS Almere | station_installation | 1105 / 67 / 1819 | -0.5775584328533918 | -0.048069755671268144 | 0.13482477102972346 | KEEP provisional |
| SS Stiens | station_installation | 0 / 0 / 1819 | — | — | — | CROSS OUT: insufficient examples after causal/availability windows |
| OS Herbayum | station_installation | 1105 / 67 / 1819 | -0.05037568130364667 | 0.08916549564989784 | 0.19765999051646677 | KEEP provisional |
| OS Westhaven | station_installation | 1105 / 67 / 1819 | -0.38078918062137146 | 0.2399623053457317 | -0.0913609378079232 | KEEP provisional |
| OS Zevenhuizen | station_installation | 1105 / 67 / 1819 | -0.2482207225537738 | 0.18333487619531952 | 0.27648369907735737 | KEEP provisional |
| RS Anklaar | station_installation | 900 / 67 / 1819 | -0.7877209373380185 | 0.1515149810062748 | 0.23293045909050325 | KEEP provisional |
| SS Noordwolde | station_installation | 874 / 67 / 1819 | -0.38150612160365216 | 0.03688742371065179 | 0.2044176278897607 | KEEP provisional |
| OS Zevenaar | station_installation | 1105 / 67 / 1819 | -0.1199883102329602 | 0.3532331157233817 | 0.3165437418150464 | KEEP provisional |
| OS Bergum | station_installation | 1105 / 67 / 1819 | -0.5563020596682478 | 0.1291693491927065 | 0.24831575449148102 | KEEP provisional |
| OS Texel | station_installation | 1105 / 67 / 1819 | -0.3024532067595518 | 0.3289766199743246 | 0.3742471083962763 | KEEP provisional |
| RS Waskemeer | station_installation | 0 / 39 / 1819 | — | — | — | CROSS OUT: insufficient examples after causal/availability windows |
| RS Wamel | station_installation | 1105 / 67 / 1819 | -0.03228112316364573 | 0.3140739881161939 | 0.24845034879135552 | KEEP provisional |
| OS Drachten | station_installation | 62 / 67 / 1819 | — | — | — | CROSS OUT: insufficient examples after causal/availability windows |
