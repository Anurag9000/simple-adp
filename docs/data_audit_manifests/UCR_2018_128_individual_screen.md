# UCR 2018: individual screen of all 128 classification problems

Source: official [UCR 2018 summary CSV](https://cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv), downloaded anonymously (11,624 bytes; SHA-256 `7fe2428588c5d513b88df2c480d71e25e4f113e8ad695e21ee2057dfac53c3ac`).
The archive access and rights decision applies to the UCR 2018 bundle route: the archive ZIP requires the published password and the archive metadata states no open reuse license. All 128 are crossed out for this route. `Best 1-NN error` is the minimum test error among the three UCR-reported Euclidean/DTW 1-NN baselines; it is context only and is not an MLP saturation test. The UCR `Default rate` is the majority-class error rate; majority accuracy below is `1 - Default rate`. Individual donor/editor names, all baseline errors, and resource estimates are in the sibling JSON/CSV files.

| Dataset | Train/test | Classes | Length | Majority accuracy | Best 1-NN error | MLP input compatibility | Decision |
|---|---:|---:|---:|---:|---:|---|---|
| Adiac | 390/391 | 37 | 176 | 0.0409 | 0.3887 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ArrowHead | 36/175 | 3 | 251 | 0.3943 | 0.2000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Beef | 30/30 | 5 | 470 | 0.2000 | 0.3333 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| BeetleFly | 20/20 | 2 | 512 | 0.5000 | 0.2500 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| BirdChicken | 20/20 | 2 | 512 | 0.5000 | 0.2500 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Car | 60/60 | 4 | 577 | 0.3167 | 0.2333 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| CBF | 30/900 | 3 | 128 | 0.3356 | 0.0033 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ChlorineConcentration | 467/3840 | 3 | 166 | 0.5326 | 0.3500 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| CinCECGTorso | 40/1380 | 4 | 1639 | 0.2536 | 0.0696 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Coffee | 28/28 | 2 | 286 | 0.5357 | 0.0000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Computers | 250/250 | 2 | 720 | 0.5000 | 0.3000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| CricketX | 390/390 | 12 | 300 | 0.1026 | 0.2282 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| CricketY | 390/390 | 12 | 300 | 0.0949 | 0.2410 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| CricketZ | 390/390 | 12 | 300 | 0.1026 | 0.2462 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| DiatomSizeReduction | 16/306 | 4 | 345 | 0.3072 | 0.0327 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| DistalPhalanxOutlineAgeGroup | 400/139 | 3 | 80 | 0.4676 | 0.2302 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| DistalPhalanxOutlineCorrect | 600/276 | 2 | 80 | 0.5833 | 0.2754 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| DistalPhalanxTW | 400/139 | 6 | 80 | 0.3022 | 0.3669 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Earthquakes | 322/139 | 2 | 512 | 0.7482 | 0.2734 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ECG200 | 100/100 | 2 | 96 | 0.6400 | 0.1200 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ECG5000 | 500/4500 | 5 | 140 | 0.5838 | 0.0749 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ECGFiveDays | 23/861 | 2 | 136 | 0.5029 | 0.2033 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ElectricDevices | 8926/7711 | 7 | 96 | 0.2537 | 0.3806 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FaceAll | 560/1690 | 14 | 131 | 0.1698 | 0.1917 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FaceFour | 24/88 | 4 | 350 | 0.2955 | 0.1136 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FacesUCR | 200/2050 | 14 | 131 | 0.1434 | 0.0878 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FiftyWords | 450/455 | 50 | 270 | 0.1253 | 0.2418 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Fish | 175/175 | 7 | 463 | 0.1657 | 0.1543 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FordA | 3601/1320 | 2 | 500 | 0.5159 | 0.3091 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FordB | 3636/810 | 2 | 500 | 0.5049 | 0.3802 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| GunPoint | 50/150 | 2 | 150 | 0.5067 | 0.0867 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Ham | 109/105 | 2 | 431 | 0.5143 | 0.4000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| HandOutlines | 1000/370 | 2 | 2709 | 0.6405 | 0.1189 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Haptics | 155/308 | 5 | 1092 | 0.2175 | 0.5877 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Herring | 64/64 | 2 | 512 | 0.5937 | 0.4688 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| InlineSkate | 100/550 | 7 | 1882 | 0.1836 | 0.6127 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| InsectWingbeatSound | 220/1980 | 11 | 256 | 0.0909 | 0.4152 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ItalyPowerDemand | 67/1029 | 2 | 24 | 0.5015 | 0.0447 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| LargeKitchenAppliances | 375/375 | 3 | 720 | 0.3333 | 0.2053 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Lightning2 | 60/61 | 2 | 637 | 0.5410 | 0.1311 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Lightning7 | 70/73 | 7 | 319 | 0.2603 | 0.2740 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Mallat | 55/2345 | 8 | 1024 | 0.1271 | 0.0661 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Meat | 60/60 | 3 | 448 | 0.3333 | 0.0667 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MedicalImages | 381/760 | 10 | 99 | 0.5145 | 0.2526 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MiddlePhalanxOutlineAgeGroup | 400/154 | 3 | 80 | 0.5714 | 0.4805 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MiddlePhalanxOutlineCorrect | 600/291 | 2 | 80 | 0.5704 | 0.2337 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MiddlePhalanxTW | 399/154 | 6 | 80 | 0.2857 | 0.4870 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MoteStrain | 20/1252 | 2 | 84 | 0.5391 | 0.1214 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| NonInvasiveFetalECGThorax1 | 1800/1965 | 42 | 750 | 0.0295 | 0.1710 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| NonInvasiveFetalECGThorax2 | 1800/1965 | 42 | 750 | 0.0295 | 0.1201 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| OliveOil | 30/30 | 4 | 570 | 0.4000 | 0.1333 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| OSULeaf | 200/242 | 6 | 427 | 0.2273 | 0.3884 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| PhalangesOutlinesCorrect | 1800/858 | 2 | 80 | 0.6131 | 0.2389 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Phoneme | 214/1896 | 39 | 1024 | 0.1129 | 0.7716 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Plane | 105/105 | 7 | 144 | 0.2000 | 0.0000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ProximalPhalanxOutlineAgeGroup | 400/205 | 3 | 80 | 0.4878 | 0.1951 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ProximalPhalanxOutlineCorrect | 600/291 | 2 | 80 | 0.6838 | 0.1924 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ProximalPhalanxTW | 400/205 | 6 | 80 | 0.3512 | 0.2439 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| RefrigerationDevices | 375/375 | 3 | 720 | 0.3333 | 0.5360 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ScreenType | 375/375 | 3 | 720 | 0.3333 | 0.5893 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ShapeletSim | 20/180 | 2 | 500 | 0.5000 | 0.3000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ShapesAll | 600/600 | 60 | 512 | 0.0167 | 0.1980 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SmallKitchenAppliances | 375/375 | 3 | 720 | 0.3333 | 0.3280 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SonyAIBORobotSurface1 | 20/601 | 2 | 70 | 0.5707 | 0.2745 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SonyAIBORobotSurface2 | 27/953 | 2 | 65 | 0.6170 | 0.1406 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| StarLightCurves | 1000/8236 | 3 | 1024 | 0.5772 | 0.0934 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Strawberry | 613/370 | 2 | 235 | 0.6432 | 0.0541 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SwedishLeaf | 500/625 | 15 | 128 | 0.0784 | 0.1536 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Symbols | 25/995 | 6 | 398 | 0.1789 | 0.0503 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SyntheticControl | 300/300 | 6 | 60 | 0.1667 | 0.0067 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ToeSegmentation1 | 40/228 | 2 | 277 | 0.5263 | 0.2281 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ToeSegmentation2 | 36/130 | 2 | 343 | 0.8154 | 0.0923 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Trace | 100/100 | 4 | 275 | 0.2900 | 0.0000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| TwoLeadECG | 23/1139 | 2 | 82 | 0.5004 | 0.0957 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| TwoPatterns | 1000/4000 | 4 | 128 | 0.2588 | 0.0000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| UWaveGestureLibraryAll | 896/3582 | 8 | 945 | 0.1284 | 0.0343 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| UWaveGestureLibraryX | 896/3582 | 8 | 315 | 0.1284 | 0.2267 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| UWaveGestureLibraryY | 896/3582 | 8 | 315 | 0.1284 | 0.3009 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| UWaveGestureLibraryZ | 896/3582 | 8 | 315 | 0.1284 | 0.3222 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Wafer | 1000/6164 | 2 | 152 | 0.8921 | 0.0045 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Wine | 57/54 | 2 | 234 | 0.5000 | 0.3889 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| WordSynonyms | 267/638 | 25 | 270 | 0.2194 | 0.2618 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Worms | 181/77 | 5 | 900 | 0.4286 | 0.4156 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| WormsTwoClass | 181/77 | 2 | 900 | 0.5714 | 0.3766 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Yoga | 300/3000 | 2 | 426 | 0.5357 | 0.1560 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ACSF1 | 100/100 | 10 | 1460 | 0.1000 | 0.3600 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| AllGestureWiimoteX | 300/700 | 10 | Vary | 0.1000 | 0.2829 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| AllGestureWiimoteY | 300/700 | 10 | Vary | 0.1000 | 0.2700 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| AllGestureWiimoteZ | 300/700 | 10 | Vary | 0.1000 | 0.3486 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| BME | 30/150 | 3 | 128 | 0.3333 | 0.0200 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Chinatown | 20/343 | 2 | 24 | 0.7259 | 0.0437 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Crop | 7200/16800 | 24 | 46 | 0.0417 | 0.2883 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| DodgerLoopDay | 78/80 | 7 | 288 | 0.1625 | 0.4125 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| DodgerLoopGame | 20/138 | 2 | 288 | 0.5217 | 0.0725 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| DodgerLoopWeekend | 20/138 | 2 | 288 | 0.7391 | 0.0145 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| EOGHorizontalSignal | 362/362 | 12 | 1250 | 0.0856 | 0.4972 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| EOGVerticalSignal | 362/362 | 12 | 1250 | 0.0856 | 0.5249 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| EthanolLevel | 504/500 | 4 | 1751 | 0.2520 | 0.7180 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FreezerRegularTrain | 150/2850 | 2 | 301 | 0.5000 | 0.0930 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| FreezerSmallTrain | 28/2850 | 2 | 301 | 0.5000 | 0.2411 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Fungi | 18/186 | 18 | 201 | 0.1022 | 0.1613 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| GestureMidAirD1 | 208/130 | 26 | Vary | 0.0385 | 0.3615 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| GestureMidAirD2 | 208/130 | 26 | Vary | 0.0385 | 0.3923 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| GestureMidAirD3 | 208/130 | 26 | Vary | 0.0385 | 0.6231 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| GesturePebbleZ1 | 132/172 | 6 | Vary | 0.1860 | 0.1744 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| GesturePebbleZ2 | 146/158 | 6 | Vary | 0.1899 | 0.2215 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| GunPointAgeSpan | 135/316 | 2 | 150 | 0.5063 | 0.0348 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| GunPointMaleVersusFemale | 135/316 | 2 | 150 | 0.5253 | 0.0032 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| GunPointOldVersusYoung | 136/315 | 2 | 150 | 0.5238 | 0.0349 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| HouseTwenty | 40/119 | 2 | 2000 | 0.5798 | 0.0588 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| InsectEPGRegularTrain | 62/249 | 3 | 601 | 0.4739 | 0.1285 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| InsectEPGSmallTrain | 17/249 | 3 | 601 | 0.4739 | 0.2651 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MelbournePedestrian | 1194/2439 | 10 | 24 | 0.1005 | 0.1525 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MixedShapesRegularTrain | 500/2425 | 5 | 1024 | 0.2697 | 0.0911 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| MixedShapesSmallTrain | 100/2425 | 5 | 1024 | 0.2697 | 0.1645 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| PickupGestureWiimoteZ | 50/50 | 10 | Vary | 0.1000 | 0.3400 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| PigAirwayPressure | 104/208 | 52 | 2000 | 0.0192 | 0.8942 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| PigArtPressure | 104/208 | 52 | 2000 | 0.0192 | 0.7548 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| PigCVP | 104/208 | 52 | 2000 | 0.0192 | 0.8413 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| PLAID | 537/537 | 11 | Vary | 0.1620 | 0.1639 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| PowerCons | 180/180 | 2 | 144 | 0.5000 | 0.0667 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| Rock | 20/50 | 4 | 2844 | 0.4200 | 0.1600 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SemgHandGenderCh2 | 300/600 | 2 | 1500 | 0.6500 | 0.1550 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SemgHandMovementCh2 | 450/450 | 6 | 1500 | 0.1667 | 0.3622 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| SemgHandSubjectCh2 | 450/450 | 5 | 1500 | 0.2000 | 0.2000 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| ShakeGestureWiimoteZ | 50/50 | 10 | Vary | 0.1000 | 0.1400 | requires padding/truncation | Cross out: password-required bundle; explicit open license absent |
| SmoothSubspace | 150/150 | 3 | 15 | 0.3333 | 0.0533 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
| UMD | 36/144 | 3 | 150 | 0.3333 | 0.0069 | fixed-width numeric vector | Cross out: password-required bundle; explicit open license absent |
