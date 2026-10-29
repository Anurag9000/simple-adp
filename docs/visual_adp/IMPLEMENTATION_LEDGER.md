# Visual ADP implementation ledger

States distinguish source/static implementation from executed tests and science.

| Item | State | Evidence / missing |
|---|---|---|
| Legacy 14 model identities | preserved | registry.py unchanged |
| V01 frozen image feature -> native ADP MLP head | runtime-tested (bounded CPU) | src/simple_adp/visual.py |
| V01 actual candidate growth | runtime-tested (bounded CPU) | delegates to native TabularADP; tests/test_visual_v01.py |
| V01 fixed non-growing control | runtime-tested (bounded CPU) | MLP-Fixed native route, tested in new test |
| V01 pretrained ResNet/ViT/ConvNeXt concrete model loaders | missing | encoder supplied externally; no weights auto-download |
| V01 packaged estimator export | static-complete | src/simple_adp/__init__.py |
| CIFAR-10/CIFAR-100/Fashion-MNIST optional loader | runtime-tested API on offline synthetic arrays (not real datasets) | vision_data.py, torchvision optional; no real dataset execution | 
| V01 suite runner routes / DatasetRegistry provider | missing | no central vision plan entries |
| V02/V03/V04 | missing | requires composite controller, fixed-dimensional local mutations |
| V05/V06a/V06b/V07 | missing | operator/model integration, mutation and resume |
| V08/V09 | missing | multistage/bank routing, isolation and candidate scheduling |
| V10/V11 | missing | ResMLP/GFNet operators and full integration |
| Real image scientific mini-runs and multi-seed benchmarks | missing | do not infer from synthetic CI |
| CPU/CUDA complete regression | partial | 30 CPU smoke/core/image-data tests passed on second run, plus variance contract workflow; CUDA/full suite not executed |

Next: implement native inner-MLP adapter and snapshot codec; connect StageTrainer
to composite image models; implement V02-V04 first. Add architecture and data
providers to ModelRegistry/DatasetRegistry and only then generate signed job
identities without disturbing existing tabular routes. Add loss vs variance
criterion pairs only where the corresponding native controller really implements
the criterion.
